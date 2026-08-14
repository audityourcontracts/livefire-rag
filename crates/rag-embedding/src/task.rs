use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufReader, Read},
    path::Path,
    sync::Arc,
    time::Duration,
};

use sha2::{Digest as _, Sha256};
use tokio::task::JoinSet;

use crate::{
    AtomicFilePublication, AtomicPublishOutcome, EmbeddingError, EmbeddingProfile, EmbeddingShard,
    EmbeddingShardExpectation, EmbeddingShardMetadata, EmbeddingShardWriter, IdentifiedEmbedder,
    Result, RetryClass, validate_embedding_profile, validate_vector,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPolicy {
    /// Total attempts, including the first request.
    pub max_attempts: usize,
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 4,
            initial_backoff: Duration::from_millis(250),
            max_backoff: Duration::from_secs(8),
        }
    }
}

impl RetryPolicy {
    #[must_use]
    pub fn delay_for_retry(&self, retry_number: usize, jitter_seed: u64) -> Duration {
        let exponent = u32::try_from(retry_number.saturating_sub(1)).unwrap_or(u32::MAX);
        let multiplier = 2_u32.checked_pow(exponent.min(31)).unwrap_or(u32::MAX);
        let base = self
            .initial_backoff
            .saturating_mul(multiplier)
            .min(self.max_backoff);
        if base.is_zero() {
            return base;
        }
        // Deterministic 0..25% jitter avoids a new RNG dependency and keeps
        // retry timing exactly reproducible in tests and receipts.
        let quarter_nanos = base.as_nanos() / 4;
        let jitter_nanos = if quarter_nanos == 0 {
            0
        } else {
            u128::from(jitter_seed.wrapping_mul(0x9e37_79b9_7f4a_7c15)) % quarter_nanos
        };
        base.saturating_add(Duration::from_nanos(
            u64::try_from(jitter_nanos).unwrap_or(u64::MAX),
        ))
        .min(self.max_backoff)
    }

    fn validate(self) -> Result<()> {
        if self.max_attempts == 0
            || self.max_attempts > 32
            || self.max_backoff < self.initial_backoff
        {
            return Err(EmbeddingError::Invalid("embedding retry policy"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddingTaskOptions {
    pub batch_size: usize,
    pub max_in_flight: usize,
    pub retry: RetryPolicy,
}

impl Default for EmbeddingTaskOptions {
    fn default() -> Self {
        Self {
            batch_size: 16,
            max_in_flight: 1,
            retry: RetryPolicy::default(),
        }
    }
}

impl EmbeddingTaskOptions {
    fn validate(self) -> Result<()> {
        if self.batch_size == 0
            || self.batch_size > 32
            || self.max_in_flight == 0
            || self.max_in_flight > 256
        {
            return Err(EmbeddingError::Invalid("embedding task concurrency"));
        }
        self.retry.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingTaskStats {
    pub rows: u64,
    pub requests: usize,
    pub retries: usize,
    pub skipped: bool,
    pub returned_model: String,
}

struct CompletedBatch {
    ordinal: usize,
    vectors: Vec<Vec<f32>>,
    attempts: usize,
    returned_model: String,
}

/// Embed ordered task texts with bounded Tokio concurrency and publish one
/// portable `LFREMB01` result. A pre-existing part is never trusted on its own:
/// the task is executed again and the new bytes must match exactly. A higher
/// layer may skip this function only after validating a bound task receipt.
pub async fn execute_embedding_task<E>(
    embedder: Arc<E>,
    profile: &EmbeddingProfile,
    texts: &[String],
    output_path: &Path,
    order_sha256: [u8; 32],
    options: EmbeddingTaskOptions,
) -> Result<EmbeddingTaskStats>
where
    E: IdentifiedEmbedder + Send + 'static,
{
    validate_embedding_profile(profile)?;
    options.validate()?;
    if texts.is_empty() {
        return Err(EmbeddingError::Invalid("empty embedding task"));
    }
    let row_count = u64::try_from(texts.len())
        .map_err(|_| EmbeddingError::Invalid("embedding task row count"))?;
    let expected = EmbeddingShardExpectation {
        row_count,
        dimensions: profile.dimensions,
        order_sha256,
    };
    let dimensions = usize::try_from(profile.dimensions)
        .map_err(|_| EmbeddingError::Invalid("profile dimensions"))?;
    let publication = AtomicFilePublication::new(output_path)?;
    let metadata = EmbeddingShardMetadata::from(expected);
    let mut writer = EmbeddingShardWriter::create(publication.staging_path(), metadata)?;
    let total_batches = texts.len().div_ceil(options.batch_size);
    let mut join_set = JoinSet::new();
    let mut next_spawn = 0;
    let mut next_write = 0;
    let mut active = 0;
    let mut completed = BTreeMap::new();
    let mut requests = 0;
    let mut retries = 0;
    let mut returned_model = None;

    while active < options.max_in_flight && next_spawn < total_batches {
        spawn_batch(
            &mut join_set,
            Arc::clone(&embedder),
            texts,
            next_spawn,
            options,
        );
        active += 1;
        next_spawn += 1;
    }

    while next_write < total_batches {
        let joined = join_set
            .join_next()
            .await
            .ok_or(EmbeddingError::Invalid("embedding task join set"))?
            .map_err(|error| EmbeddingError::Task(error.to_string()))??;
        active -= 1;
        if completed.insert(joined.ordinal, joined).is_some() {
            return Err(EmbeddingError::Invalid("duplicate embedding batch"));
        }

        while let Some(batch) = completed.remove(&next_write) {
            if batch.returned_model != profile.model
                || returned_model
                    .as_ref()
                    .is_some_and(|model| model != &batch.returned_model)
            {
                return Err(EmbeddingError::Invalid("embedding response model"));
            }
            returned_model = Some(batch.returned_model.clone());
            requests += batch.attempts;
            retries += batch.attempts.saturating_sub(1);
            for vector in batch.vectors {
                validate_vector(&vector, dimensions, &profile.normalization)?;
                writer.write_vector(&vector)?;
            }
            next_write += 1;
        }

        // Active requests plus this reorder buffer are bounded to at most
        // twice max_in_flight. Stop feeding work if an early batch is slow.
        while active < options.max_in_flight
            && next_spawn < total_batches
            && completed.len() < options.max_in_flight
        {
            spawn_batch(
                &mut join_set,
                Arc::clone(&embedder),
                texts,
                next_spawn,
                options,
            );
            active += 1;
            next_spawn += 1;
        }
    }

    writer.finish()?;
    let staged_sha256 = file_digest(publication.staging_path())?;
    match publication.commit()? {
        AtomicPublishOutcome::Published => {}
        AtomicPublishOutcome::AlreadyExists => {
            if file_digest(output_path)? != staged_sha256 {
                return Err(EmbeddingError::Invalid(
                    "existing embedding shard differs from executed result",
                ));
            }
        }
    }
    EmbeddingShard::open_expected(output_path, expected)?
        .validate_normalization(&profile.normalization)?;
    Ok(EmbeddingTaskStats {
        rows: row_count,
        requests,
        retries,
        skipped: false,
        returned_model: returned_model
            .ok_or(EmbeddingError::Invalid("missing embedding response model"))?,
    })
}

fn spawn_batch<E>(
    join_set: &mut JoinSet<Result<CompletedBatch>>,
    embedder: Arc<E>,
    texts: &[String],
    ordinal: usize,
    options: EmbeddingTaskOptions,
) where
    E: IdentifiedEmbedder + Send + 'static,
{
    let start = ordinal * options.batch_size;
    let end = (start + options.batch_size).min(texts.len());
    let batch = texts[start..end].to_vec();
    join_set.spawn(async move {
        let mut attempts = 0;
        loop {
            attempts += 1;
            match embedder.embed_identified(&batch).await {
                Ok(response) => {
                    if response.vectors.len() != batch.len() {
                        return Err(EmbeddingError::Invalid("batch cardinality"));
                    }
                    return Ok(CompletedBatch {
                        ordinal,
                        vectors: response.vectors,
                        attempts,
                        returned_model: response.returned_model,
                    });
                }
                Err(error)
                    if error.retry_class() == RetryClass::Temporary
                        && attempts < options.retry.max_attempts =>
                {
                    let delay = options
                        .retry
                        .delay_for_retry(attempts, ordinal as u64 ^ attempts as u64);
                    tokio::time::sleep(delay).await;
                }
                Err(error) => return Err(error),
            }
        }
    });
}

fn file_digest(path: &Path) -> Result<[u8; 32]> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().into())
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use tempfile::tempdir;

    use super::*;
    use crate::{Embedder, IdentifiedEmbeddingBatch};

    struct DelayedFake {
        calls: AtomicUsize,
    }

    impl Embedder for DelayedFake {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let number = texts[0].parse::<u64>().unwrap();
            tokio::time::sleep(Duration::from_millis((4 - number) * 5)).await;
            Ok(texts
                .iter()
                .map(|text| vec![text.parse::<f32>().unwrap(), 1.0])
                .collect())
        }
    }

    impl IdentifiedEmbedder for DelayedFake {
        async fn embed_identified(&self, texts: &[String]) -> Result<IdentifiedEmbeddingBatch> {
            Ok(IdentifiedEmbeddingBatch {
                vectors: self.embed(texts).await?,
                returned_model: "fake".into(),
            })
        }
    }

    fn profile() -> EmbeddingProfile {
        EmbeddingProfile {
            id: "test".into(),
            version: "1".into(),
            sha256: "a".repeat(64),
            model: "fake".into(),
            dimensions: 2,
            normalization: "none".into(),
            query_instruction: None,
            query_composition: None,
        }
    }

    #[tokio::test]
    async fn out_of_order_responses_are_ordered_and_orphan_parts_are_reexecuted() {
        let root = tempdir().unwrap();
        let output = root.path().join("part.f32");
        let embedder = Arc::new(DelayedFake {
            calls: AtomicUsize::new(0),
        });
        let texts = ["1".into(), "2".into(), "3".into()];
        let options = EmbeddingTaskOptions {
            batch_size: 1,
            max_in_flight: 3,
            retry: RetryPolicy {
                max_attempts: 1,
                initial_backoff: Duration::ZERO,
                max_backoff: Duration::ZERO,
            },
        };
        let first = execute_embedding_task(
            Arc::clone(&embedder),
            &profile(),
            &texts,
            &output,
            [9; 32],
            options,
        )
        .await
        .unwrap();
        assert_eq!(first.requests, 3);
        assert!(!first.skipped);
        let vectors = EmbeddingShard::open(&output)
            .unwrap()
            .vectors()
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(
            vectors,
            vec![vec![1.0, 1.0], vec![2.0, 1.0], vec![3.0, 1.0]]
        );

        let second = execute_embedding_task(
            Arc::clone(&embedder),
            &profile(),
            &texts,
            &output,
            [9; 32],
            options,
        )
        .await
        .unwrap();
        assert!(!second.skipped);
        assert_eq!(second.returned_model, "fake");
        assert_eq!(embedder.calls.load(Ordering::Relaxed), 6);
    }

    struct RetryingFake {
        calls: AtomicUsize,
    }

    impl Embedder for RetryingFake {
        async fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
            if self.calls.fetch_add(1, Ordering::Relaxed) == 0 {
                Err(EmbeddingError::Temporary("try again".into()))
            } else {
                Ok(vec![vec![1.0, 0.0]])
            }
        }
    }

    impl IdentifiedEmbedder for RetryingFake {
        async fn embed_identified(&self, texts: &[String]) -> Result<IdentifiedEmbeddingBatch> {
            Ok(IdentifiedEmbeddingBatch {
                vectors: self.embed(texts).await?,
                returned_model: "fake".into(),
            })
        }
    }

    #[tokio::test]
    async fn retries_only_temporary_failures_and_counts_attempts() {
        let root = tempdir().unwrap();
        let stats = execute_embedding_task(
            Arc::new(RetryingFake {
                calls: AtomicUsize::new(0),
            }),
            &profile(),
            &["one".into()],
            &root.path().join("part.f32"),
            [1; 32],
            EmbeddingTaskOptions {
                retry: RetryPolicy {
                    max_attempts: 2,
                    initial_backoff: Duration::ZERO,
                    max_backoff: Duration::ZERO,
                },
                ..EmbeddingTaskOptions::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(stats.requests, 2);
        assert_eq!(stats.retries, 1);
    }

    struct InvalidFake(Vec<f32>);

    impl Embedder for InvalidFake {
        async fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(vec![self.0.clone()])
        }
    }

    impl IdentifiedEmbedder for InvalidFake {
        async fn embed_identified(&self, texts: &[String]) -> Result<IdentifiedEmbeddingBatch> {
            Ok(IdentifiedEmbeddingBatch {
                vectors: self.embed(texts).await?,
                returned_model: "fake".into(),
            })
        }
    }

    #[tokio::test]
    async fn wrong_dimension_and_non_finite_outputs_are_never_published() {
        for (name, vector) in [("shape", vec![1.0]), ("nan", vec![f32::NAN, 0.0])] {
            let root = tempdir().unwrap();
            let output = root.path().join(format!("{name}.f32"));
            let result = execute_embedding_task(
                Arc::new(InvalidFake(vector)),
                &profile(),
                &["one".into()],
                &output,
                [1; 32],
                EmbeddingTaskOptions::default(),
            )
            .await;
            assert!(result.is_err());
            assert!(!output.exists());
            assert!(fs::read_dir(root.path()).unwrap().next().is_none());
        }
    }

    struct ModelFake(&'static str, Vec<f32>);

    impl Embedder for ModelFake {
        async fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(vec![self.1.clone()])
        }
    }

    impl IdentifiedEmbedder for ModelFake {
        async fn embed_identified(&self, texts: &[String]) -> Result<IdentifiedEmbeddingBatch> {
            Ok(IdentifiedEmbeddingBatch {
                vectors: self.embed(texts).await?,
                returned_model: self.0.into(),
            })
        }
    }

    #[tokio::test]
    async fn wrong_response_model_and_different_existing_vectors_are_rejected() {
        let root = tempdir().unwrap();
        let output = root.path().join("part.f32");
        let texts = ["one".into()];
        let wrong_model = execute_embedding_task(
            Arc::new(ModelFake("other", vec![1.0, 0.0])),
            &profile(),
            &texts,
            &output,
            [1; 32],
            EmbeddingTaskOptions::default(),
        )
        .await;
        assert!(wrong_model.is_err());
        assert!(!output.exists());

        execute_embedding_task(
            Arc::new(ModelFake("fake", vec![1.0, 0.0])),
            &profile(),
            &texts,
            &output,
            [1; 32],
            EmbeddingTaskOptions::default(),
        )
        .await
        .unwrap();
        let different = execute_embedding_task(
            Arc::new(ModelFake("fake", vec![0.0, 1.0])),
            &profile(),
            &texts,
            &output,
            [1; 32],
            EmbeddingTaskOptions::default(),
        )
        .await;
        assert!(different.is_err());
        assert_eq!(
            EmbeddingShard::open(&output)
                .unwrap()
                .vectors()
                .unwrap()
                .next()
                .unwrap()
                .unwrap(),
            vec![1.0, 0.0]
        );
    }
}

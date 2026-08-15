use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufReader, Read},
    ops::Range,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
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

/// A checked selection over an ordered list of embedding tasks. Ranges are
/// start-inclusive and end-exclusive, matching Rust slices and plan ordinals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum TaskSelection {
    All,
    Range { start: usize, end: usize },
}

impl TaskSelection {
    pub fn resolve(self, task_count: usize) -> Result<Range<usize>> {
        match self {
            Self::All => Ok(0..task_count),
            Self::Range { start, end } if start < end && end <= task_count => Ok(start..end),
            Self::Range { .. } => Err(EmbeddingError::Invalid("embedding task selection")),
        }
    }

    pub fn select<T>(self, tasks: &[T]) -> Result<&[T]> {
        let range = self.resolve(tasks.len())?;
        tasks
            .get(range)
            .ok_or(EmbeddingError::Invalid("embedding task selection"))
    }
}

/// Sanitized attempt outcome. It deliberately carries no backend error text,
/// request content, URL, or response body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingAttemptOutcome {
    Success,
    TemporaryFailure,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingAttemptReport {
    pub attempt: usize,
    pub input_rows: u64,
    /// Sum of UTF-8 text lengths, excluding JSON and transport framing.
    pub input_text_bytes: u64,
    /// Decoded f32 vector payload bytes, excluding JSON and transport framing.
    pub vector_bytes: u64,
    pub elapsed_micros: u64,
    pub backoff_micros: u64,
    pub outcome: EmbeddingAttemptOutcome,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingBatchReport {
    pub batch_ordinal: usize,
    pub row_start: u64,
    pub row_end: u64,
    pub input_text_bytes: u64,
    pub vector_bytes: u64,
    pub elapsed_micros: u64,
    pub backoff_micros: u64,
    pub attempts: Vec<EmbeddingAttemptReport>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingTaskReport {
    pub rows: u64,
    pub batches: usize,
    pub attempts: usize,
    pub retries: usize,
    /// UTF-8 bytes in the task's unique inputs before retries.
    pub unique_input_text_bytes: u64,
    /// UTF-8 bytes submitted across every attempt, including retries.
    pub sent_input_text_bytes: u64,
    /// Decoded f32 vector payload bytes from successful requests.
    pub vector_bytes: u64,
    /// Complete published `LFREMB01` bytes, including its 64-byte header.
    pub shard_bytes: u64,
    pub elapsed_micros: u64,
    /// Sum of request durations. This can exceed wall time under concurrency.
    pub request_elapsed_micros: u64,
    pub retry_backoff_micros: u64,
    pub peak_in_flight: usize,
    pub batch_reports: Vec<EmbeddingBatchReport>,
}

struct CompletedBatch {
    ordinal: usize,
    vectors: Vec<Vec<f32>>,
    attempts: usize,
    returned_model: String,
    report: EmbeddingBatchReport,
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
    execute_embedding_task_reported(embedder, profile, texts, output_path, order_sha256, options)
        .await
        .map(|(stats, _)| stats)
}

/// Execute an embedding task and return aggregate plus per-attempt counters
/// suitable for a sanitized operational report. Text, URLs, response bodies,
/// and backend error messages are never retained in the report.
pub async fn execute_embedding_task_reported<E>(
    embedder: Arc<E>,
    profile: &EmbeddingProfile,
    texts: &[String],
    output_path: &Path,
    order_sha256: [u8; 32],
    options: EmbeddingTaskOptions,
) -> Result<(EmbeddingTaskStats, EmbeddingTaskReport)>
where
    E: IdentifiedEmbedder + Send + 'static,
{
    let task_started = Instant::now();
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
    let mut peak_in_flight = 0;
    let mut batch_reports = Vec::with_capacity(total_batches);

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
        peak_in_flight = peak_in_flight.max(active);
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
            batch_reports.push(batch.report);
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
            peak_in_flight = peak_in_flight.max(active);
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
    let returned_model =
        returned_model.ok_or(EmbeddingError::Invalid("missing embedding response model"))?;
    let unique_input_text_bytes = texts
        .iter()
        .fold(0_u64, |total, text| total.saturating_add(text.len() as u64));
    let sent_input_text_bytes = batch_reports.iter().fold(0_u64, |total, batch| {
        batch.attempts.iter().fold(total, |total, attempt| {
            total.saturating_add(attempt.input_text_bytes)
        })
    });
    let vector_bytes = batch_reports.iter().fold(0_u64, |total, batch| {
        total.saturating_add(batch.vector_bytes)
    });
    let request_elapsed_micros = batch_reports.iter().fold(0_u64, |total, batch| {
        batch.attempts.iter().fold(total, |total, attempt| {
            total.saturating_add(attempt.elapsed_micros)
        })
    });
    let retry_backoff_micros = batch_reports.iter().fold(0_u64, |total, batch| {
        total.saturating_add(batch.backoff_micros)
    });
    let shard_bytes = File::open(output_path)?.metadata()?.len();
    let stats = EmbeddingTaskStats {
        rows: row_count,
        requests,
        retries,
        skipped: false,
        returned_model,
    };
    let report = EmbeddingTaskReport {
        rows: row_count,
        batches: total_batches,
        attempts: requests,
        retries,
        unique_input_text_bytes,
        sent_input_text_bytes,
        vector_bytes,
        shard_bytes,
        elapsed_micros: duration_micros(task_started.elapsed()),
        request_elapsed_micros,
        retry_backoff_micros,
        peak_in_flight,
        batch_reports,
    };
    Ok((stats, report))
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
        let mut attempt_reports = Vec::new();
        let input_rows = batch.len() as u64;
        let input_text_bytes = batch
            .iter()
            .fold(0_u64, |total, text| total.saturating_add(text.len() as u64));
        let batch_started = Instant::now();
        let mut backoff_micros = 0_u64;
        loop {
            attempts += 1;
            let attempt_started = Instant::now();
            match embedder.embed_identified(&batch).await {
                Ok(response) => {
                    if response.vectors.len() != batch.len() {
                        return Err(EmbeddingError::Invalid("batch cardinality"));
                    }
                    let vector_bytes = response.vectors.iter().fold(0_u64, |total, vector| {
                        total.saturating_add((vector.len() as u64).saturating_mul(4))
                    });
                    attempt_reports.push(EmbeddingAttemptReport {
                        attempt: attempts,
                        input_rows,
                        input_text_bytes,
                        vector_bytes,
                        elapsed_micros: duration_micros(attempt_started.elapsed()),
                        backoff_micros: 0,
                        outcome: EmbeddingAttemptOutcome::Success,
                    });
                    return Ok(CompletedBatch {
                        ordinal,
                        vectors: response.vectors,
                        attempts,
                        returned_model: response.returned_model,
                        report: EmbeddingBatchReport {
                            batch_ordinal: ordinal,
                            row_start: start as u64,
                            row_end: end as u64,
                            input_text_bytes,
                            vector_bytes,
                            elapsed_micros: duration_micros(batch_started.elapsed()),
                            backoff_micros,
                            attempts: attempt_reports,
                        },
                    });
                }
                Err(error)
                    if error.retry_class() == RetryClass::Temporary
                        && attempts < options.retry.max_attempts =>
                {
                    let delay = options
                        .retry
                        .delay_for_retry(attempts, ordinal as u64 ^ attempts as u64);
                    let attempt_report_index = attempt_reports.len();
                    attempt_reports.push(EmbeddingAttemptReport {
                        attempt: attempts,
                        input_rows,
                        input_text_bytes,
                        vector_bytes: 0,
                        elapsed_micros: duration_micros(attempt_started.elapsed()),
                        backoff_micros: 0,
                        outcome: EmbeddingAttemptOutcome::TemporaryFailure,
                    });
                    let backoff_started = Instant::now();
                    tokio::time::sleep(delay).await;
                    let elapsed = duration_micros(backoff_started.elapsed());
                    attempt_reports[attempt_report_index].backoff_micros = elapsed;
                    backoff_micros = backoff_micros.saturating_add(elapsed);
                }
                Err(error) => return Err(error),
            }
        }
    });
}

fn duration_micros(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
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
        io::{Read, Write},
        net::TcpListener,
        sync::atomic::{AtomicUsize, Ordering},
        thread,
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
            vector_derivation: None,
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
        let (first, report) = execute_embedding_task_reported(
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
        assert_eq!(report.peak_in_flight, 3);
        assert_eq!(
            report
                .batch_reports
                .iter()
                .map(|batch| batch.batch_ordinal)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
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
        let (stats, report) = execute_embedding_task_reported(
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
        assert_eq!(report.rows, 1);
        assert_eq!(report.batches, 1);
        assert_eq!(report.attempts, 2);
        assert_eq!(report.retries, 1);
        assert_eq!(report.unique_input_text_bytes, 3);
        assert_eq!(report.sent_input_text_bytes, 6);
        assert_eq!(report.vector_bytes, 8);
        assert_eq!(report.shard_bytes, 72);
        assert_eq!(report.peak_in_flight, 1);
        assert_eq!(report.batch_reports[0].attempts.len(), 2);
        assert_eq!(
            report.batch_reports[0].attempts[0].outcome,
            EmbeddingAttemptOutcome::TemporaryFailure
        );
        assert_eq!(
            report.batch_reports[0].attempts[1].outcome,
            EmbeddingAttemptOutcome::Success
        );
        let json = serde_json::to_string(&report).unwrap();
        assert!(!json.contains("one"));
        assert!(!json.contains("try again"));
    }

    #[test]
    fn task_selection_is_checked_and_end_exclusive() {
        let tasks = [10, 20, 30, 40];
        assert_eq!(TaskSelection::All.select(&tasks).unwrap(), &tasks);
        assert_eq!(
            TaskSelection::Range { start: 1, end: 3 }
                .select(&tasks)
                .unwrap(),
            &[20, 30]
        );
        for selection in [
            TaskSelection::Range { start: 1, end: 1 },
            TaskSelection::Range { start: 3, end: 2 },
            TaskSelection::Range { start: 0, end: 5 },
        ] {
            assert!(selection.select(&tasks).is_err());
        }
    }

    fn read_http_request(connection: &mut std::net::TcpStream) {
        let mut request = Vec::new();
        let mut buffer = [0_u8; 1024];
        loop {
            let read = connection.read(&mut buffer).unwrap();
            request.extend_from_slice(&buffer[..read]);
            let Some(header_end) = request.windows(4).position(|part| part == b"\r\n\r\n") else {
                continue;
            };
            let headers = String::from_utf8_lossy(&request[..header_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
                .unwrap();
            if request.len() >= header_end + 4 + content_length {
                break;
            }
        }
    }

    fn retry_server(
        first_status: Option<u16>,
        first_delay: Duration,
    ) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let requests = usize::from(first_status.is_some() || !first_delay.is_zero()) + 1;
        let server = thread::spawn(move || {
            for request in 0..requests {
                let (mut connection, _) = listener.accept().unwrap();
                read_http_request(&mut connection);
                if request == 0 && !first_delay.is_zero() {
                    thread::sleep(first_delay);
                }
                if request == 0
                    && let Some(status) = first_status
                {
                    let body = "SECRET_BACKEND_BODY";
                    let _ = write!(
                        connection,
                        "HTTP/1.1 {status} retry\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                } else if request == 0 && !first_delay.is_zero() {
                    let _ = write!(
                        connection,
                        "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    );
                } else {
                    let body = r#"{"model":"fake","data":[{"index":0,"embedding":[1.0,0.0]}]}"#;
                    write!(
                        connection,
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    )
                    .unwrap();
                    connection.flush().unwrap();
                }
            }
        });
        (format!("http://{address}"), server)
    }

    async fn execute_http_retry_case(
        first_status: Option<u16>,
        first_delay: Duration,
        timeout: Duration,
    ) -> EmbeddingTaskReport {
        let root = tempdir().unwrap();
        let (endpoint, server) = retry_server(first_status, first_delay);
        let embedder =
            Arc::new(crate::LmStudioEmbedder::with_timeout(&endpoint, "fake", timeout).unwrap());
        let (_, report) = execute_embedding_task_reported(
            embedder,
            &profile(),
            &["one".into()],
            &root.path().join("part.f32"),
            [7; 32],
            EmbeddingTaskOptions {
                retry: RetryPolicy {
                    max_attempts: 2,
                    initial_backoff: if first_delay.is_zero() {
                        Duration::ZERO
                    } else {
                        timeout
                    },
                    max_backoff: if first_delay.is_zero() {
                        Duration::ZERO
                    } else {
                        timeout
                    },
                },
                ..EmbeddingTaskOptions::default()
            },
        )
        .await
        .unwrap();
        server.join().unwrap();
        report
    }

    #[tokio::test]
    async fn retries_408_429_and_all_5xx_without_exposing_response_bodies() {
        for status in [408, 429, 500, 501, 503, 599] {
            let report =
                execute_http_retry_case(Some(status), Duration::ZERO, Duration::from_secs(1)).await;
            assert_eq!(report.attempts, 2, "status {status}");
            assert_eq!(report.retries, 1, "status {status}");
            let json = serde_json::to_string(&report).unwrap();
            assert!(!json.contains("SECRET_BACKEND_BODY"));
        }
    }

    #[tokio::test]
    async fn retries_request_timeout_and_reports_only_timing_and_counts() {
        let report =
            execute_http_retry_case(None, Duration::from_millis(30), Duration::from_millis(20))
                .await;
        assert_eq!(report.attempts, 2);
        assert_eq!(report.retries, 1);
        assert!(report.request_elapsed_micros >= 20_000);
        assert!(report.retry_backoff_micros >= 20_000);
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

    #[tokio::test]
    async fn non_unit_vectors_fail_l2_normalization_and_are_never_published() {
        let root = tempdir().unwrap();
        let output = root.path().join("norm.f32");
        let mut normalized_profile = profile();
        normalized_profile.normalization = "l2".into();
        let result = execute_embedding_task(
            Arc::new(InvalidFake(vec![2.0, 0.0])),
            &normalized_profile,
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

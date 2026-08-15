use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest as _, Sha256};
use tokenizers::Tokenizer;

use super::{
    ComponentRef, DatasetIdentity, Digest, EMBEDDING_PLAN_V2_SCHEMA, EmbeddingProfileRef,
    EmbeddingResultSetManifest, PipelineError, PreparedCorpusManifest, PreparedDocumentObject,
    PreparedDocumentRow, Result, SafeRelativePath, VectorResultReceipt, atomic_write,
    canonical_digest, component_digest, digest_bytes, embedding_input_order_digest,
    require_safe_u64, require_text, resolve_existing_artifact, resolve_output_artifact,
    validate_prepared_documents,
};

const DOCUMENT_TOKEN_COUNTS_DOMAIN: &[u8] = b"livefire.rag.document-token-counts/1\0";
pub const DOCUMENT_TOKEN_COUNTS_PATH: &str = "token-counts/document-token-counts.u32le";

#[derive(Clone, Copy)]
struct TaskIdentityContext<'a> {
    prepared_corpus_sha256: &'a Digest,
    embedding_profile_sha256: &'a Digest,
    tokenizer_sha256: &'a Digest,
}

/// The executable tokenizer representation consumed by this crate.
///
/// Qwen tokenizer execution is intentionally bound to frozen Hugging Face
/// `tokenizer.json` bytes. The artifact's component version records the
/// tokenizer source revision; `model_revision` separately records the target
/// GGUF revision whose tokenizer compatibility the packager asserted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerArtifactFormat {
    HuggingFaceTokenizerJson,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutableTokenizerRef {
    pub artifact: ComponentRef,
    pub format: TokenizerArtifactFormat,
    pub model_revision: String,
    /// Logical tokenizer identity declared by the embedding profile. This
    /// prevents a tokenizer for another runtime from being reused merely
    /// because both runtimes load the same model revision.
    pub target_tokenizer: ComponentRef,
    pub add_special_tokens: bool,
    pub maximum_input_bytes: u64,
}

impl ExecutableTokenizerRef {
    pub fn validate(&self) -> Result<()> {
        self.artifact.validate()?;
        self.target_tokenizer.validate()?;
        require_safe_u64(self.maximum_input_bytes)?;
        if self.model_revision.is_empty() || self.maximum_input_bytes == 0 {
            return Err(PipelineError::Invalid("executable tokenizer identity"));
        }
        Ok(())
    }

    pub fn validate_for_profile(&self, profile: &EmbeddingProfileRef) -> Result<()> {
        self.validate()?;
        if self.model_revision != profile.model_artifact.version {
            return Err(PipelineError::Invalid("tokenizer model revision binding"));
        }
        if self.target_tokenizer != profile.tokenizer {
            return Err(PipelineError::Invalid("tokenizer profile binding"));
        }
        Ok(())
    }
}

/// A loaded tokenizer whose executable bytes match the durable artifact
/// digest. Truncation and padding are rejected because either would make a
/// reported count differ from the model input token sequence.
pub struct ExactTokenizer {
    reference: ExecutableTokenizerRef,
    tokenizer: Tokenizer,
}

impl ExactTokenizer {
    pub fn from_bytes(reference: ExecutableTokenizerRef, bytes: &[u8]) -> Result<Self> {
        reference.validate()?;
        if super::digest_bytes(bytes) != reference.artifact.sha256 {
            return Err(PipelineError::Invalid("tokenizer artifact byte digest"));
        }
        let tokenizer = Tokenizer::from_bytes(bytes)
            .map_err(|error| PipelineError::Tokenizer(error.to_string()))?;
        if tokenizer.get_truncation().is_some() || tokenizer.get_padding().is_some() {
            return Err(PipelineError::Invalid(
                "tokenizer artifact truncation or padding",
            ));
        }
        Ok(Self {
            reference,
            tokenizer,
        })
    }

    #[must_use]
    pub fn reference(&self) -> &ExecutableTokenizerRef {
        &self.reference
    }

    pub fn count(&self, input: &str) -> Result<u64> {
        let token_ids = self.token_ids(input)?;
        let count =
            u64::try_from(token_ids.len()).map_err(|_| PipelineError::Invalid("token count"))?;
        require_safe_u64(count)?;
        Ok(count)
    }

    /// Return the exact token IDs used for planning. This is primarily used
    /// to prove that the portable tokenizer matches the tokenizer embedded in
    /// the pinned GGUF before a large embedding run begins.
    pub fn token_ids(&self, input: &str) -> Result<Vec<u32>> {
        let input_bytes = u64::try_from(input.len())
            .map_err(|_| PipelineError::Invalid("tokenizer input byte length"))?;
        require_safe_u64(input_bytes)?;
        if input_bytes > self.reference.maximum_input_bytes {
            return Err(PipelineError::Invalid("tokenizer input byte limit"));
        }
        let encoding = self
            .tokenizer
            .encode(input, self.reference.add_special_tokens)
            .map_err(|error| PipelineError::Tokenizer(error.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }
}

pub fn format_document_input_exact(format: &str, semantic_text: &str) -> Result<String> {
    rag_contracts::format_document_input(format, semantic_text)
        .map_err(|_| PipelineError::Invalid("document input format"))
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingInputSliceV2 {
    pub path: SafeRelativePath,
    pub object_sha256: Digest,
    pub row_offset: u64,
    pub rows: u64,
    pub embedding_input_order_sha256: Digest,
    pub token_count: u64,
    pub maximum_document_tokens: u32,
    pub document_token_counts_sha256: Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingTaskV2 {
    pub task_id: String,
    pub ordinal_start: u64,
    pub ordinal_end: u64,
    pub input_slices: Vec<EmbeddingInputSliceV2>,
    pub embedding_input_order_sha256: Digest,
    pub token_count: u64,
    pub maximum_document_tokens: u32,
    pub document_token_counts_sha256: Digest,
    pub result_path: SafeRelativePath,
    pub receipt_path: SafeRelativePath,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DocumentTokenCountsObject {
    pub path: SafeRelativePath,
    pub rows: u64,
    pub bytes: u64,
    pub sha256: Digest,
    pub document_token_counts_sha256: Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenStatistics {
    pub total_tokens: u64,
    pub p50_tokens: u32,
    pub p90_tokens: u32,
    pub p95_tokens: u32,
    pub p99_tokens: u32,
    pub maximum_tokens: u32,
}

impl EmbeddingTaskV2 {
    #[must_use]
    pub fn row_count(&self) -> u64 {
        self.ordinal_end.saturating_sub(self.ordinal_start)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingPlanV2 {
    pub schema_version: String,
    pub component_sha256: Digest,
    pub prepared_corpus_sha256: Digest,
    pub dataset: DatasetIdentity,
    pub embedding_profile: EmbeddingProfileRef,
    pub executable_tokenizer: ExecutableTokenizerRef,
    pub document_count: u64,
    pub document_order_sha256: Digest,
    pub document_token_counts_sha256: Digest,
    pub document_token_counts_object: DocumentTokenCountsObject,
    pub token_statistics: TokenStatistics,
    pub maximum_task_tokens: u64,
    pub maximum_task_documents: u32,
    pub tasks: Vec<EmbeddingTaskV2>,
}

impl EmbeddingPlanV2 {
    pub fn validate(&self) -> Result<()> {
        require_safe_u64(self.document_count)?;
        require_safe_u64(self.maximum_task_tokens)?;
        if self.schema_version != EMBEDDING_PLAN_V2_SCHEMA
            || self.maximum_task_tokens == 0
            || self.maximum_task_documents == 0
        {
            return Err(PipelineError::Invalid("embedding plan v2 schema or limits"));
        }
        self.dataset.validate()?;
        self.embedding_profile.validate()?;
        format_document_input_exact(&self.embedding_profile.document_format, "validation")?;
        self.executable_tokenizer
            .validate_for_profile(&self.embedding_profile)?;
        validate_token_count_metadata(self)?;
        validate_v2_tasks(
            &self.tasks,
            self.document_count,
            self.maximum_task_tokens,
            self.maximum_task_documents,
            self.embedding_profile.maximum_input_tokens,
            TaskIdentityContext {
                prepared_corpus_sha256: &self.prepared_corpus_sha256,
                embedding_profile_sha256: &self.embedding_profile.component.sha256,
                tokenizer_sha256: &self.executable_tokenizer.artifact.sha256,
            },
        )?;
        if self.component_sha256 != component_digest(self)? {
            return Err(PipelineError::Invalid("embedding plan v2 component digest"));
        }
        Ok(())
    }

    pub fn seal(&mut self) -> Result<()> {
        self.component_sha256 = component_digest(self)?;
        self.validate()
    }

    #[must_use]
    pub fn task(&self, task_id: &str) -> Option<&EmbeddingTaskV2> {
        self.tasks.iter().find(|task| task.task_id == task_id)
    }

    pub fn validate_manifest_binding(&self, prepared: &PreparedCorpusManifest) -> Result<()> {
        self.validate()?;
        prepared.validate()?;
        if self.prepared_corpus_sha256 != prepared.component_sha256
            || self.dataset != prepared.dataset
            || self.document_count != prepared.document_count
            || self.document_order_sha256 != prepared.document_order_sha256
        {
            return Err(PipelineError::Invalid("embedding plan v2 corpus binding"));
        }
        validate_v2_slices_against_manifest(&self.tasks, &prepared.documents)
    }

    /// Re-tokenizes the exact formatted document inputs and proves that every
    /// durable token total, maximum, digest, and greedy task boundary matches.
    pub fn validate_with_tokenizer(
        &self,
        prepared: &PreparedCorpusManifest,
        documents: &[PreparedDocumentRow],
        tokenizer_bytes: &[u8],
    ) -> Result<()> {
        self.validate_manifest_binding(prepared)?;
        let expected = build_token_balanced_plan(
            prepared,
            documents,
            self.embedding_profile.clone(),
            self.executable_tokenizer.clone(),
            tokenizer_bytes,
            TokenBalanceOptions {
                maximum_task_tokens: self.maximum_task_tokens,
                maximum_task_documents: self.maximum_task_documents,
            },
        )?;
        if &expected != self {
            return Err(PipelineError::Invalid(
                "embedding plan v2 exact token binding",
            ));
        }
        Ok(())
    }

    /// Validate one count per absolute document ordinal against every plan,
    /// task, and slice total and percentile.
    pub fn validate_document_token_counts(&self, counts: &[u32]) -> Result<()> {
        self.validate()?;
        validate_counts_against_plan(self, counts)
    }

    /// Atomically write the plan's count object at its fixed relative path.
    pub fn write_document_token_counts(&self, plan_root: &Path, counts: &[u32]) -> Result<PathBuf> {
        self.validate_document_token_counts(counts)?;
        let path = resolve_output_artifact(plan_root, &self.document_token_counts_object.path)?;
        atomic_write(&path, &encode_document_token_counts(counts))?;
        Ok(path)
    }

    /// Safely resolve, read, and fully validate the bound count object.
    pub fn read_document_token_counts(&self, plan_root: &Path) -> Result<Vec<u32>> {
        self.validate()?;
        let path = resolve_existing_artifact(plan_root, &self.document_token_counts_object.path)?;
        let metadata = fs::metadata(&path)?;
        if !metadata.is_file() || metadata.len() != self.document_token_counts_object.bytes {
            return Err(PipelineError::Invalid(
                "document token count object metadata",
            ));
        }
        let bytes = fs::read(path)?;
        if digest_bytes(&bytes) != self.document_token_counts_object.sha256 {
            return Err(PipelineError::Invalid(
                "document token count object file digest",
            ));
        }
        let counts = decode_document_token_counts(&bytes)?;
        self.validate_document_token_counts(&counts)?;
        Ok(counts)
    }
}

/// Rebind an existing token-balanced plan to a post-processing profile. Token
/// counts and input slices stay unchanged; task identities and output paths
/// change because the vector profile is part of their identity.
pub fn derive_embedding_plan_v2(
    source: &EmbeddingPlanV2,
    embedding_profile: EmbeddingProfileRef,
) -> Result<EmbeddingPlanV2> {
    source.validate()?;
    embedding_profile.validate()?;
    if source.embedding_profile.model_artifact != embedding_profile.model_artifact
        || source.embedding_profile.tokenizer != embedding_profile.tokenizer
        || source.embedding_profile.maximum_input_tokens != embedding_profile.maximum_input_tokens
        || source.embedding_profile.pooling != embedding_profile.pooling
        || source.embedding_profile.document_format != embedding_profile.document_format
        || source.embedding_profile.dtype != embedding_profile.dtype
        || source.embedding_profile.dimensions <= embedding_profile.dimensions
        || embedding_profile.normalization != "l2"
    {
        return Err(PipelineError::Invalid("derived plan profile"));
    }
    let mut plan = source.clone();
    plan.embedding_profile = embedding_profile;
    for task in &mut plan.tasks {
        let task_id = embedding_task_v2_id(
            TaskIdentityContext {
                prepared_corpus_sha256: &plan.prepared_corpus_sha256,
                embedding_profile_sha256: &plan.embedding_profile.component.sha256,
                tokenizer_sha256: &plan.executable_tokenizer.artifact.sha256,
            },
            task.ordinal_start,
            task.ordinal_end,
            &task.embedding_input_order_sha256,
            &task.document_token_counts_sha256,
            task.token_count,
        )?
        .to_string();
        task.task_id = task_id.clone();
        task.result_path = SafeRelativePath::new(format!("parts/{task_id}.f32"))?;
        task.receipt_path = SafeRelativePath::new(format!("receipts/{task_id}.json"))?;
    }
    plan.seal()?;
    Ok(plan)
}

impl VectorResultReceipt {
    /// Validate an existing receipt against a token-balanced v2 plan. The
    /// receipt's v1 wire schema remains valid because its plan reference was
    /// deliberately version-neutral.
    pub fn validate_against_v2(&self, plan: &EmbeddingPlanV2) -> Result<()> {
        plan.validate()?;
        for value in [
            self.ordinal_start,
            self.ordinal_end,
            self.vector.rows,
            self.vector.bytes,
            self.executor.requests,
            self.executor.retries,
            self.executor.input_bytes_upper_bound,
            self.executor.elapsed_ms,
        ] {
            require_safe_u64(value)?;
        }
        if !matches!(
            (self.schema_version.as_str(), self.derived()),
            (super::VECTOR_RECEIPT_SCHEMA, false) | (super::DERIVED_VECTOR_RECEIPT_SCHEMA, true)
        ) {
            return Err(PipelineError::Invalid("vector receipt schema"));
        }
        let task = plan
            .task(&self.task_id)
            .ok_or(PipelineError::Invalid("receipt task"))?;
        if self.plan_sha256 != plan.component_sha256
            || self.prepared_corpus_sha256 != plan.prepared_corpus_sha256
            || self.embedding_profile_sha256 != plan.embedding_profile.component.sha256
            || self.ordinal_start != task.ordinal_start
            || self.ordinal_end != task.ordinal_end
            || self.embedding_input_order_sha256 != task.embedding_input_order_sha256
            || self.vector.path != task.result_path
            || self.vector.rows != task.row_count()
            || self.vector.dimensions != plan.embedding_profile.dimensions
            || self.vector.dtype != plan.embedding_profile.dtype
            || self.vector.embedding_input_order_sha256 != task.embedding_input_order_sha256
        {
            return Err(PipelineError::Invalid("receipt plan binding"));
        }
        if !self.finite_values_validated || !self.normalization_validated {
            return Err(PipelineError::Invalid("receipt validation flags"));
        }
        self.executor.implementation.validate()?;
        self.executor.runtime.validate()?;
        require_text(&self.executor.returned_model)?;
        if let Some(derivation) = &self.derivation {
            derivation.validate(self.vector.dimensions)?;
        }
        let invalid_executor = if self.test_only() {
            self.executor.requests != 0
                || self.executor.retries != 0
                || self.executor.conformance_passed
        } else if self.derived() {
            self.executor.implementation.id != super::DERIVED_VECTOR_EXECUTOR_ID
                || self.executor.requests != 0
                || self.executor.retries != 0
                || self.executor.conformance_passed
        } else {
            !self.executor.conformance_passed
                || self.executor.retries > self.executor.requests
                || self.executor.requests == 0
        };
        if invalid_executor {
            return Err(PipelineError::Invalid("receipt executor validation"));
        }
        let expected_bytes = 64_u64
            .checked_add(
                self.vector
                    .rows
                    .checked_mul(u64::from(self.vector.dimensions))
                    .and_then(|values| values.checked_mul(4))
                    .ok_or(PipelineError::Invalid("receipt vector byte length"))?,
            )
            .ok_or(PipelineError::Invalid("receipt vector byte length"))?;
        if self.vector.bytes != expected_bytes {
            return Err(PipelineError::Invalid("receipt vector byte length"));
        }
        if self.component_sha256 != component_digest(self)? {
            return Err(PipelineError::Invalid("receipt component digest"));
        }
        Ok(())
    }
}

impl EmbeddingResultSetManifest {
    /// Validate complete receipt coverage for a token-balanced v2 plan.
    pub fn validate_v2(
        &self,
        plan: &EmbeddingPlanV2,
        loaded: &[VectorResultReceipt],
    ) -> Result<()> {
        plan.validate()?;
        require_safe_u64(self.document_count)?;
        if !matches!(
            (
                self.schema_version.as_str(),
                self.test_only,
                self.derivation.is_some()
            ),
            (super::RESULT_SET_SCHEMA, false, false)
                | (super::TEST_RESULT_SET_SCHEMA, true, false)
                | (super::DERIVED_RESULT_SET_SCHEMA, false, true)
        ) || self.plan_sha256 != plan.component_sha256
            || self.prepared_corpus_sha256 != plan.prepared_corpus_sha256
            || self.embedding_profile_sha256 != plan.embedding_profile.component.sha256
            || self.document_count != plan.document_count
            || self.document_order_sha256 != plan.document_order_sha256
        {
            return Err(PipelineError::Invalid("result set plan binding"));
        }
        if self.receipts.len() != plan.tasks.len() || loaded.len() != plan.tasks.len() {
            return Err(PipelineError::Invalid("result set coverage"));
        }
        if let Some(derivation) = &self.derivation {
            derivation.validate(plan.embedding_profile.dimensions)?;
        }
        let entries: BTreeMap<_, _> = self
            .receipts
            .iter()
            .map(|entry| (entry.task_id.as_str(), entry))
            .collect();
        let values: BTreeMap<_, _> = loaded
            .iter()
            .map(|receipt| (receipt.task_id.as_str(), receipt))
            .collect();
        if entries.len() != self.receipts.len() || values.len() != loaded.len() {
            return Err(PipelineError::Invalid("duplicate result task"));
        }
        let mut executor_implementation = None;
        for task in &plan.tasks {
            let entry = entries
                .get(task.task_id.as_str())
                .ok_or(PipelineError::Invalid("missing result task"))?;
            let receipt = values
                .get(task.task_id.as_str())
                .ok_or(PipelineError::Invalid("missing receipt"))?;
            receipt.validate_against_v2(plan)?;
            if executor_implementation
                .as_ref()
                .is_some_and(|expected| expected != &receipt.executor.implementation)
            {
                return Err(PipelineError::Invalid(
                    "result executor implementation differs",
                ));
            }
            executor_implementation.get_or_insert_with(|| receipt.executor.implementation.clone());
            if receipt.test_only() != self.test_only {
                return Err(PipelineError::Invalid("result set test-only binding"));
            }
            if receipt.derivation.as_ref().map(|value| {
                (
                    &value.parent_embedding_profile_sha256,
                    &value.parent_result_set_sha256,
                    value.parent_dimensions,
                    value.transformation.as_str(),
                )
            }) != self.derivation.as_ref().map(|value| {
                (
                    &value.parent_embedding_profile_sha256,
                    &value.parent_result_set_sha256,
                    value.parent_dimensions,
                    value.transformation.as_str(),
                )
            }) {
                return Err(PipelineError::Invalid("result derivation binding"));
            }
            if entry.path != task.receipt_path || entry.sha256 != receipt.component_sha256 {
                return Err(PipelineError::Invalid("result receipt binding"));
            }
        }
        if self.component_sha256 != component_digest(self)? {
            return Err(PipelineError::Invalid("result set component digest"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenBalanceOptions {
    pub maximum_task_tokens: u64,
    pub maximum_task_documents: u32,
}

pub fn document_token_counts_digest(start_ordinal: u64, counts: &[u32]) -> Result<Digest> {
    let mut hasher = Sha256::new();
    hasher.update(DOCUMENT_TOKEN_COUNTS_DOMAIN);
    for (offset, count) in counts.iter().enumerate() {
        let offset =
            u64::try_from(offset).map_err(|_| PipelineError::Invalid("token count ordinal"))?;
        let ordinal = start_ordinal
            .checked_add(offset)
            .ok_or(PipelineError::Invalid("token count ordinal"))?;
        require_safe_u64(ordinal)?;
        hasher.update(ordinal.to_le_bytes());
        hasher.update(count.to_le_bytes());
    }
    Ok(Digest(format!("{:x}", hasher.finalize())))
}

#[must_use]
pub fn encode_document_token_counts(counts: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(counts.len().saturating_mul(4));
    for count in counts {
        bytes.extend_from_slice(&count.to_le_bytes());
    }
    bytes
}

pub fn decode_document_token_counts(bytes: &[u8]) -> Result<Vec<u32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(PipelineError::Invalid(
            "document token count object byte length",
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect())
}

pub fn token_statistics(counts: &[u32]) -> Result<TokenStatistics> {
    if counts.contains(&0) {
        return Err(PipelineError::Invalid("zero document token count"));
    }
    if counts.is_empty() {
        return Ok(TokenStatistics {
            total_tokens: 0,
            p50_tokens: 0,
            p90_tokens: 0,
            p95_tokens: 0,
            p99_tokens: 0,
            maximum_tokens: 0,
        });
    }
    let mut sorted = counts.to_vec();
    sorted.sort_unstable();
    let percentile = |percent: usize| {
        let rank = (sorted.len() / 100) * percent + ((sorted.len() % 100) * percent).div_ceil(100);
        sorted[rank.saturating_sub(1)]
    };
    Ok(TokenStatistics {
        total_tokens: sum_counts(counts)?,
        p50_tokens: percentile(50),
        p90_tokens: percentile(90),
        p95_tokens: percentile(95),
        p99_tokens: percentile(99),
        maximum_tokens: *sorted.last().expect("non-empty counts"),
    })
}

pub fn build_token_balanced_plan(
    prepared: &PreparedCorpusManifest,
    documents: &[PreparedDocumentRow],
    embedding_profile: EmbeddingProfileRef,
    executable_tokenizer: ExecutableTokenizerRef,
    tokenizer_bytes: &[u8],
    options: TokenBalanceOptions,
) -> Result<EmbeddingPlanV2> {
    Ok(build_token_balanced_plan_with_counts(
        prepared,
        documents,
        embedding_profile,
        executable_tokenizer,
        tokenizer_bytes,
        options,
    )?
    .0)
}

pub fn build_token_balanced_plan_with_counts(
    prepared: &PreparedCorpusManifest,
    documents: &[PreparedDocumentRow],
    embedding_profile: EmbeddingProfileRef,
    executable_tokenizer: ExecutableTokenizerRef,
    tokenizer_bytes: &[u8],
    options: TokenBalanceOptions,
) -> Result<(EmbeddingPlanV2, Vec<u32>)> {
    prepared.validate()?;
    validate_prepared_documents(prepared, documents)?;
    embedding_profile.validate()?;
    executable_tokenizer.validate_for_profile(&embedding_profile)?;
    require_safe_u64(options.maximum_task_tokens)?;
    if options.maximum_task_tokens == 0 || options.maximum_task_documents == 0 {
        return Err(PipelineError::Invalid("token balance limits"));
    }
    let tokenizer = ExactTokenizer::from_bytes(executable_tokenizer.clone(), tokenizer_bytes)?;
    let mut counts = Vec::with_capacity(documents.len());
    for document in documents {
        let input = format_document_input_exact(
            &embedding_profile.document_format,
            &document.semantic_text,
        )?;
        let count = tokenizer.count(&input)?;
        let count = u32::try_from(count).map_err(|_| PipelineError::Invalid("document tokens"))?;
        if count == 0
            || count > embedding_profile.maximum_input_tokens
            || u64::from(count) > options.maximum_task_tokens
        {
            return Err(PipelineError::Invalid("overlength embedding document"));
        }
        counts.push(count);
    }

    let maximum_task_documents = usize::try_from(options.maximum_task_documents)
        .map_err(|_| PipelineError::Invalid("token balance document limit"))?;
    let mut ranges = Vec::new();
    let mut start = 0_usize;
    while start < counts.len() {
        let mut end = start;
        let mut tokens = 0_u64;
        while end < counts.len()
            && end - start < maximum_task_documents
            && tokens
                .checked_add(u64::from(counts[end]))
                .is_some_and(|next| next <= options.maximum_task_tokens)
        {
            tokens += u64::from(counts[end]);
            end += 1;
        }
        if end == start {
            return Err(PipelineError::Invalid("overlength embedding task"));
        }
        ranges.push((start, end));
        start = end;
    }

    let mut tasks = Vec::with_capacity(ranges.len());
    for (start, end) in ranges {
        let ordinal_start =
            u64::try_from(start).map_err(|_| PipelineError::Invalid("task ordinal"))?;
        let ordinal_end = u64::try_from(end).map_err(|_| PipelineError::Invalid("task ordinal"))?;
        let task_counts = &counts[start..end];
        let token_count = sum_counts(task_counts)?;
        let maximum_document_tokens = maximum_count(task_counts)?;
        let token_digest = document_token_counts_digest(ordinal_start, task_counts)?;
        let order_digest = embedding_input_order_digest(&documents[start..end]);
        let input_slices = build_v2_slices(
            &prepared.documents,
            documents,
            &counts,
            ordinal_start,
            ordinal_end,
        )?;
        let task_id = embedding_task_v2_id(
            TaskIdentityContext {
                prepared_corpus_sha256: &prepared.component_sha256,
                embedding_profile_sha256: &embedding_profile.component.sha256,
                tokenizer_sha256: &executable_tokenizer.artifact.sha256,
            },
            ordinal_start,
            ordinal_end,
            &order_digest,
            &token_digest,
            token_count,
        )?
        .to_string();
        tasks.push(EmbeddingTaskV2 {
            task_id: task_id.clone(),
            ordinal_start,
            ordinal_end,
            input_slices,
            embedding_input_order_sha256: order_digest,
            token_count,
            maximum_document_tokens,
            document_token_counts_sha256: token_digest,
            result_path: SafeRelativePath::new(format!("parts/{task_id}.f32"))?,
            receipt_path: SafeRelativePath::new(format!("receipts/{task_id}.json"))?,
        });
    }
    let counts_bytes = encode_document_token_counts(&counts);
    let counts_digest = document_token_counts_digest(0, &counts)?;
    let mut plan = EmbeddingPlanV2 {
        schema_version: EMBEDDING_PLAN_V2_SCHEMA.into(),
        component_sha256: Digest::new("0".repeat(64))?,
        prepared_corpus_sha256: prepared.component_sha256.clone(),
        dataset: prepared.dataset.clone(),
        embedding_profile,
        executable_tokenizer,
        document_count: prepared.document_count,
        document_order_sha256: prepared.document_order_sha256.clone(),
        document_token_counts_sha256: counts_digest.clone(),
        document_token_counts_object: DocumentTokenCountsObject {
            path: SafeRelativePath::new(DOCUMENT_TOKEN_COUNTS_PATH)?,
            rows: prepared.document_count,
            bytes: u64::try_from(counts_bytes.len())
                .map_err(|_| PipelineError::Invalid("document token count object bytes"))?,
            sha256: digest_bytes(&counts_bytes),
            document_token_counts_sha256: counts_digest,
        },
        token_statistics: token_statistics(&counts)?,
        maximum_task_tokens: options.maximum_task_tokens,
        maximum_task_documents: options.maximum_task_documents,
        tasks,
    };
    plan.seal()?;
    Ok((plan, counts))
}

fn build_v2_slices(
    objects: &[PreparedDocumentObject],
    documents: &[PreparedDocumentRow],
    counts: &[u32],
    task_start: u64,
    task_end: u64,
) -> Result<Vec<EmbeddingInputSliceV2>> {
    let mut slices = Vec::new();
    let mut object_start = 0_u64;
    for object in objects {
        let object_end = object_start
            .checked_add(object.object.rows)
            .ok_or(PipelineError::Invalid("document object range"))?;
        let start = task_start.max(object_start);
        let end = task_end.min(object_end);
        if start < end {
            let start_usize = usize::try_from(start)
                .map_err(|_| PipelineError::Invalid("document slice range"))?;
            let end_usize =
                usize::try_from(end).map_err(|_| PipelineError::Invalid("document slice range"))?;
            let slice_counts = counts
                .get(start_usize..end_usize)
                .ok_or(PipelineError::Invalid("document slice token range"))?;
            slices.push(EmbeddingInputSliceV2 {
                path: object.object.path.clone(),
                object_sha256: object.object.sha256.clone(),
                row_offset: start - object_start,
                rows: end - start,
                embedding_input_order_sha256: embedding_input_order_digest(
                    documents
                        .get(start_usize..end_usize)
                        .ok_or(PipelineError::Invalid("document slice range"))?,
                ),
                token_count: sum_counts(slice_counts)?,
                maximum_document_tokens: maximum_count(slice_counts)?,
                document_token_counts_sha256: document_token_counts_digest(start, slice_counts)?,
            });
        }
        object_start = object_end;
    }
    if slices.is_empty() {
        return Err(PipelineError::Invalid("empty document slices"));
    }
    Ok(slices)
}

fn sum_counts(counts: &[u32]) -> Result<u64> {
    counts.iter().try_fold(0_u64, |total, count| {
        total
            .checked_add(u64::from(*count))
            .ok_or(PipelineError::Invalid("token total"))
    })
}

fn maximum_count(counts: &[u32]) -> Result<u32> {
    counts
        .iter()
        .copied()
        .max()
        .ok_or(PipelineError::Invalid("empty token counts"))
}

fn validate_token_count_metadata(plan: &EmbeddingPlanV2) -> Result<()> {
    for value in [
        plan.document_token_counts_object.rows,
        plan.document_token_counts_object.bytes,
        plan.token_statistics.total_tokens,
    ] {
        require_safe_u64(value)?;
    }
    let expected_bytes = plan
        .document_count
        .checked_mul(4)
        .ok_or(PipelineError::Invalid("document token count object bytes"))?;
    let task_total = plan.tasks.iter().try_fold(0_u64, |total, task| {
        total
            .checked_add(task.token_count)
            .ok_or(PipelineError::Invalid("plan token total"))
    })?;
    let task_maximum = plan
        .tasks
        .iter()
        .map(|task| task.maximum_document_tokens)
        .max()
        .unwrap_or(0);
    let statistics = &plan.token_statistics;
    if plan.document_token_counts_object.path.as_str() != DOCUMENT_TOKEN_COUNTS_PATH
        || plan.document_token_counts_object.rows != plan.document_count
        || plan.document_token_counts_object.bytes != expected_bytes
        || plan
            .document_token_counts_object
            .document_token_counts_sha256
            != plan.document_token_counts_sha256
        || statistics.total_tokens != task_total
        || statistics.maximum_tokens != task_maximum
        || statistics.maximum_tokens > plan.embedding_profile.maximum_input_tokens
        || statistics.p50_tokens > statistics.p90_tokens
        || statistics.p90_tokens > statistics.p95_tokens
        || statistics.p95_tokens > statistics.p99_tokens
        || statistics.p99_tokens > statistics.maximum_tokens
        || (plan.document_count == 0)
            != (statistics.total_tokens == 0
                && statistics.p50_tokens == 0
                && statistics.p90_tokens == 0
                && statistics.p95_tokens == 0
                && statistics.p99_tokens == 0
                && statistics.maximum_tokens == 0)
    {
        return Err(PipelineError::Invalid("document token count metadata"));
    }
    Ok(())
}

fn validate_counts_against_plan(plan: &EmbeddingPlanV2, counts: &[u32]) -> Result<()> {
    if u64::try_from(counts.len()).ok() != Some(plan.document_count)
        || counts
            .iter()
            .any(|count| *count == 0 || *count > plan.embedding_profile.maximum_input_tokens)
    {
        return Err(PipelineError::Invalid("document token count coverage"));
    }
    let encoded = encode_document_token_counts(counts);
    if u64::try_from(encoded.len()).ok() != Some(plan.document_token_counts_object.bytes)
        || digest_bytes(&encoded) != plan.document_token_counts_object.sha256
        || document_token_counts_digest(0, counts)? != plan.document_token_counts_sha256
        || token_statistics(counts)? != plan.token_statistics
    {
        return Err(PipelineError::Invalid(
            "document token count object binding",
        ));
    }
    for task in &plan.tasks {
        let start = usize::try_from(task.ordinal_start)
            .map_err(|_| PipelineError::Invalid("task token range"))?;
        let end = usize::try_from(task.ordinal_end)
            .map_err(|_| PipelineError::Invalid("task token range"))?;
        let task_counts = counts
            .get(start..end)
            .ok_or(PipelineError::Invalid("task token range"))?;
        if sum_counts(task_counts)? != task.token_count
            || maximum_count(task_counts)? != task.maximum_document_tokens
            || document_token_counts_digest(task.ordinal_start, task_counts)?
                != task.document_token_counts_sha256
        {
            return Err(PipelineError::Invalid("task token count binding"));
        }
        let mut slice_start = start;
        for slice in &task.input_slices {
            let rows = usize::try_from(slice.rows)
                .map_err(|_| PipelineError::Invalid("slice token range"))?;
            let slice_end = slice_start
                .checked_add(rows)
                .ok_or(PipelineError::Invalid("slice token range"))?;
            let slice_counts = counts
                .get(slice_start..slice_end)
                .ok_or(PipelineError::Invalid("slice token range"))?;
            let absolute_start = u64::try_from(slice_start)
                .map_err(|_| PipelineError::Invalid("slice token range"))?;
            if sum_counts(slice_counts)? != slice.token_count
                || maximum_count(slice_counts)? != slice.maximum_document_tokens
                || document_token_counts_digest(absolute_start, slice_counts)?
                    != slice.document_token_counts_sha256
            {
                return Err(PipelineError::Invalid("slice token count binding"));
            }
            slice_start = slice_end;
        }
        if slice_start != end {
            return Err(PipelineError::Invalid("slice token count coverage"));
        }
    }
    Ok(())
}

fn validate_v2_tasks(
    tasks: &[EmbeddingTaskV2],
    document_count: u64,
    maximum_task_tokens: u64,
    maximum_task_documents: u32,
    maximum_input_tokens: u32,
    identity: TaskIdentityContext<'_>,
) -> Result<()> {
    let mut next = 0_u64;
    let mut ids = BTreeSet::new();
    let mut paths = BTreeSet::new();
    for task in tasks {
        require_safe_u64(task.ordinal_start)?;
        require_safe_u64(task.ordinal_end)?;
        require_safe_u64(task.token_count)?;
        let rows = task.row_count();
        let slice_rows = task.input_slices.iter().try_fold(0_u64, |sum, slice| {
            for value in [slice.row_offset, slice.rows, slice.token_count] {
                require_safe_u64(value)?;
            }
            if slice.rows == 0
                || slice.token_count == 0
                || slice.token_count < slice.rows
                || slice.maximum_document_tokens == 0
                || slice.maximum_document_tokens > maximum_input_tokens
                || u64::from(slice.maximum_document_tokens) > slice.token_count
            {
                return Err(PipelineError::Invalid("embedding task v2 slice tokens"));
            }
            sum.checked_add(slice.rows)
                .ok_or(PipelineError::Invalid("embedding task v2 slice rows"))
        })?;
        let slice_tokens = task.input_slices.iter().try_fold(0_u64, |sum, slice| {
            sum.checked_add(slice.token_count)
                .ok_or(PipelineError::Invalid("embedding task v2 slice tokens"))
        })?;
        let slice_max = task
            .input_slices
            .iter()
            .map(|slice| slice.maximum_document_tokens)
            .max()
            .ok_or(PipelineError::Invalid("embedding task v2 slices"))?;
        let expected_task_id = embedding_task_v2_id(
            identity,
            task.ordinal_start,
            task.ordinal_end,
            &task.embedding_input_order_sha256,
            &task.document_token_counts_sha256,
            task.token_count,
        )?;
        if task.task_id != expected_task_id.as_str()
            || task.ordinal_start != next
            || task.ordinal_end <= task.ordinal_start
            || rows != slice_rows
            || rows > u64::from(maximum_task_documents)
            || task.input_slices.is_empty()
            || task.token_count == 0
            || task.token_count < rows
            || task.token_count > maximum_task_tokens
            || task.token_count != slice_tokens
            || task.maximum_document_tokens != slice_max
            || task.maximum_document_tokens > maximum_input_tokens
            || u64::from(task.maximum_document_tokens) > task.token_count
            || !ids.insert(&task.task_id)
            || !paths.insert(task.result_path.clone())
            || !paths.insert(task.receipt_path.clone())
        {
            return Err(PipelineError::Invalid("embedding task v2 coverage"));
        }
        next = task.ordinal_end;
    }
    if next != document_count || (document_count > 0) != !tasks.is_empty() {
        return Err(PipelineError::Invalid("embedding plan v2 coverage"));
    }
    Ok(())
}

fn embedding_task_v2_id(
    identity: TaskIdentityContext<'_>,
    ordinal_start: u64,
    ordinal_end: u64,
    embedding_input_order_sha256: &Digest,
    document_token_counts_sha256: &Digest,
    token_count: u64,
) -> Result<Digest> {
    canonical_digest(&json!({
        "schema_version": "livefire.rag.embedding-task/2",
        "prepared_corpus_sha256": identity.prepared_corpus_sha256,
        "embedding_profile_sha256": identity.embedding_profile_sha256,
        "tokenizer_sha256": identity.tokenizer_sha256,
        "ordinal_start": ordinal_start,
        "ordinal_end": ordinal_end,
        "embedding_input_order_sha256": embedding_input_order_sha256,
        "document_token_counts_sha256": document_token_counts_sha256,
        "token_count": token_count,
    }))
}

fn validate_v2_slices_against_manifest(
    tasks: &[EmbeddingTaskV2],
    objects: &[PreparedDocumentObject],
) -> Result<()> {
    let mut object_starts = BTreeMap::new();
    let mut object_start = 0_u64;
    for object in objects {
        object_starts.insert(object.object.path.clone(), (object_start, object));
        object_start = object_start
            .checked_add(object.object.rows)
            .ok_or(PipelineError::Invalid("document object ordinal"))?;
    }
    for task in tasks {
        let mut next = task.ordinal_start;
        for slice in &task.input_slices {
            let (start, object) = object_starts
                .get(&slice.path)
                .ok_or(PipelineError::Invalid("task v2 input object"))?;
            let slice_end = slice
                .row_offset
                .checked_add(slice.rows)
                .ok_or(PipelineError::Invalid("task v2 input slice"))?;
            let global_start = start
                .checked_add(slice.row_offset)
                .ok_or(PipelineError::Invalid("task v2 input slice"))?;
            let global_end = start
                .checked_add(slice_end)
                .ok_or(PipelineError::Invalid("task v2 input slice"))?;
            if slice.object_sha256 != object.object.sha256
                || slice_end > object.object.rows
                || global_start != next
                || global_end > task.ordinal_end
            {
                return Err(PipelineError::Invalid("task v2 input slice binding"));
            }
            next = global_end;
        }
        if next != task.ordinal_end {
            return Err(PipelineError::Invalid("task v2 input slice coverage"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DocumentKind, EmbeddingResultSetManifest, ExecutorReceipt, ObjectEntry,
        PreparedDocumentObject, ReceiptEntry, RelationAccounting, VectorObject,
        VectorResultReceipt, document_order_digest,
    };
    use serde::Deserialize;

    const TEST_TOKENIZER: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"WhitespaceSplit"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"a":0,"b":1,"é":2,"👩‍💻":3,"<unk>":4},"unk_token":"<unk>"}}"#;

    fn digest(character: char) -> Digest {
        Digest::new(character.to_string().repeat(64)).unwrap()
    }

    fn component(id: &str, version: &str, sha256: Digest) -> ComponentRef {
        ComponentRef {
            id: id.into(),
            version: version.into(),
            sha256,
        }
    }

    fn tokenizer_ref(maximum_input_bytes: u64) -> ExecutableTokenizerRef {
        ExecutableTokenizerRef {
            artifact: component(
                "hf.example.tokenizer-json",
                "tokenizer-revision-z",
                super::super::digest_bytes(TEST_TOKENIZER.as_bytes()),
            ),
            format: TokenizerArtifactFormat::HuggingFaceTokenizerJson,
            model_revision: "revision-a".into(),
            target_tokenizer: component("logical-tokenizer", "revision-a", digest('c')),
            add_special_tokens: false,
            maximum_input_bytes,
        }
    }

    fn profile(maximum_input_tokens: u32) -> EmbeddingProfileRef {
        EmbeddingProfileRef {
            component: component("profile", "1", digest('a')),
            model_artifact: component("model", "revision-a", digest('b')),
            tokenizer: component("logical-tokenizer", "revision-a", digest('c')),
            maximum_input_tokens,
            pooling: "last_token".into(),
            normalization: "l2".into(),
            dimensions: 4,
            dtype: "f32le".into(),
            document_format: "{semantic_text}".into(),
        }
    }

    fn corpus(texts: &[&str]) -> (PreparedCorpusManifest, Vec<PreparedDocumentRow>) {
        let rows = texts
            .iter()
            .enumerate()
            .map(|(ordinal, text)| PreparedDocumentRow {
                document_ordinal: ordinal as u64,
                document_id: format!("doc-{ordinal:02}"),
                document_sha256: digest(
                    char::from_digit((ordinal % 10) as u32, 16).expect("hex digit"),
                ),
                semantic_text_sha256: super::super::digest_bytes(text.as_bytes()),
                semantic_text: (*text).into(),
                document_kind: DocumentKind::Activity,
                primary_relation: "events".into(),
                facets_json: "{}".into(),
                relations_json: "[\"events\"]".into(),
                occurrence_count: 1,
            })
            .collect::<Vec<_>>();
        let object_sha = digest('9');
        let mut manifest = PreparedCorpusManifest {
            schema_version: crate::PREPARED_CORPUS_SCHEMA.into(),
            component_sha256: digest('0'),
            dataset: DatasetIdentity {
                id: "dataset".into(),
                version: "1".into(),
                source_snapshot: component("snapshot", "1", digest('1')),
                mapping: component("mapping", "1", digest('2')),
                included_relations: vec!["events".into()],
                excluded_relations: vec![],
                structured_only_relations: vec![],
            },
            projection_policy: component("projection", "1", digest('3')),
            document_schema: component("document-schema", "1", digest('4')),
            occurrence_schema: component("occurrence-schema", "1", digest('5')),
            preparation_implementation: component("prepare", "1", digest('6')),
            document_count: rows.len() as u64,
            occurrence_count: rows.len() as u64,
            document_order_sha256: document_order_digest(
                rows.iter().map(|row| row.document_id.as_str()),
            ),
            embedding_input_order_sha256: embedding_input_order_digest(&rows),
            documents: vec![PreparedDocumentObject {
                object: ObjectEntry {
                    path: SafeRelativePath::new("documents/part.parquet").unwrap(),
                    rows: rows.len() as u64,
                    bytes: 1,
                    sha256: object_sha,
                    logical_order_sha256: canonical_digest(&rows).unwrap(),
                },
                ordinal: 0,
                first_document_id: rows.first().unwrap().document_id.clone(),
                last_document_id: rows.last().unwrap().document_id.clone(),
                embedding_input_order_sha256: embedding_input_order_digest(&rows),
            }],
            occurrences: vec![crate::PreparedOccurrenceObject {
                object: ObjectEntry {
                    path: SafeRelativePath::new("occurrences/events/part.parquet").unwrap(),
                    rows: rows.len() as u64,
                    bytes: 1,
                    sha256: digest('7'),
                    logical_order_sha256: digest('8'),
                },
                ordinal: 0,
                relation: "events".into(),
            }],
            relation_accounting: BTreeMap::from([(
                "events".into(),
                RelationAccounting {
                    source_rows: rows.len() as u64,
                    searchable_occurrences: rows.len() as u64,
                    selected_occurrences: rows.len() as u64,
                    excluded_rows: 0,
                },
            )]),
        };
        manifest.seal().unwrap();
        (manifest, rows)
    }

    #[test]
    fn exact_tokenizer_binds_bytes_and_counts_hostile_unicode() {
        let reference = tokenizer_ref(128);
        let tokenizer =
            ExactTokenizer::from_bytes(reference.clone(), TEST_TOKENIZER.as_bytes()).unwrap();
        assert_eq!(tokenizer.count("a é 👩‍💻").unwrap(), 3);
        assert_eq!(tokenizer.count("e\u{301} \0 b").unwrap(), 3);

        let mut wrong = TEST_TOKENIZER.as_bytes().to_vec();
        wrong.push(b' ');
        assert!(ExactTokenizer::from_bytes(reference, &wrong).is_err());
    }

    #[test]
    fn tokenizer_rejects_byte_limit_before_encoding() {
        let tokenizer =
            ExactTokenizer::from_bytes(tokenizer_ref(4), TEST_TOKENIZER.as_bytes()).unwrap();
        assert_eq!(tokenizer.count("éé").unwrap(), 1);
        assert!(tokenizer.count("ééa").is_err());
    }

    #[test]
    fn token_balanced_tasks_are_deterministic_consecutive_and_exact() {
        let (prepared, rows) = corpus(&["a b", "a", "é 👩‍💻", "b b", "a"]);
        let options = TokenBalanceOptions {
            maximum_task_tokens: 3,
            maximum_task_documents: 3,
        };
        let first = build_token_balanced_plan(
            &prepared,
            &rows,
            profile(4),
            tokenizer_ref(128),
            TEST_TOKENIZER.as_bytes(),
            options,
        )
        .unwrap();
        let second = build_token_balanced_plan(
            &prepared,
            &rows,
            profile(4),
            tokenizer_ref(128),
            TEST_TOKENIZER.as_bytes(),
            options,
        )
        .unwrap();
        assert_eq!(first, second);
        assert_eq!(first.tasks.len(), 3);
        assert_eq!(
            (first.tasks[0].ordinal_start, first.tasks[0].ordinal_end),
            (0, 2)
        );
        assert_eq!(
            (first.tasks[1].ordinal_start, first.tasks[1].ordinal_end),
            (2, 3)
        );
        assert_eq!(
            (first.tasks[2].ordinal_start, first.tasks[2].ordinal_end),
            (3, 5)
        );
        assert_eq!(
            first
                .tasks
                .iter()
                .map(|task| task.token_count)
                .collect::<Vec<_>>(),
            vec![3, 2, 3]
        );
        first
            .validate_with_tokenizer(&prepared, &rows, TEST_TOKENIZER.as_bytes())
            .unwrap();
    }

    #[test]
    fn overlength_is_rejected_before_a_plan_can_be_sealed() {
        let (prepared, rows) = corpus(&["a b a"]);
        assert!(
            build_token_balanced_plan(
                &prepared,
                &rows,
                profile(2),
                tokenizer_ref(128),
                TEST_TOKENIZER.as_bytes(),
                TokenBalanceOptions {
                    maximum_task_tokens: 10,
                    maximum_task_documents: 10,
                },
            )
            .is_err()
        );

        let mut wrong_target = tokenizer_ref(128);
        wrong_target.target_tokenizer = component("other-tokenizer", "revision-a", digest('d'));
        assert!(
            build_token_balanced_plan(
                &prepared,
                &rows,
                profile(4),
                wrong_target,
                TEST_TOKENIZER.as_bytes(),
                TokenBalanceOptions {
                    maximum_task_tokens: 4,
                    maximum_task_documents: 4,
                },
            )
            .is_err()
        );
    }

    #[test]
    fn derived_plan_keeps_ranges_but_gets_new_stable_task_identities() {
        let (prepared, rows) = corpus(&["a", "a b"]);
        let source = build_token_balanced_plan(
            &prepared,
            &rows,
            profile(4),
            tokenizer_ref(128),
            TEST_TOKENIZER.as_bytes(),
            TokenBalanceOptions {
                maximum_task_tokens: 4,
                maximum_task_documents: 4,
            },
        )
        .unwrap();
        let mut reduced_profile = source.embedding_profile.clone();
        reduced_profile.component.sha256 = digest('f');
        reduced_profile.dimensions = 2;
        let first = derive_embedding_plan_v2(&source, reduced_profile.clone()).unwrap();
        let second = derive_embedding_plan_v2(&source, reduced_profile).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.tasks[0].ordinal_start, source.tasks[0].ordinal_start);
        assert_eq!(first.tasks[0].ordinal_end, source.tasks[0].ordinal_end);
        assert_ne!(first.tasks[0].task_id, source.tasks[0].task_id);
        assert_ne!(first.component_sha256, source.component_sha256);
    }

    #[test]
    fn exact_validation_rejects_mutated_token_totals_and_revision() {
        let (prepared, rows) = corpus(&["a", "a b"]);
        let mut plan = build_token_balanced_plan(
            &prepared,
            &rows,
            profile(4),
            tokenizer_ref(128),
            TEST_TOKENIZER.as_bytes(),
            TokenBalanceOptions {
                maximum_task_tokens: 4,
                maximum_task_documents: 4,
            },
        )
        .unwrap();
        plan.tasks[0].token_count += 1;
        plan.seal().unwrap_err();

        let mut consistently_mutated = build_token_balanced_plan(
            &prepared,
            &rows,
            profile(4),
            tokenizer_ref(128),
            TEST_TOKENIZER.as_bytes(),
            TokenBalanceOptions {
                maximum_task_tokens: 4,
                maximum_task_documents: 4,
            },
        )
        .unwrap();
        consistently_mutated.tasks[0].token_count += 1;
        consistently_mutated.tasks[0].input_slices[0].token_count += 1;
        consistently_mutated.token_statistics.total_tokens += 1;
        let task = &consistently_mutated.tasks[0];
        consistently_mutated.tasks[0].task_id = embedding_task_v2_id(
            TaskIdentityContext {
                prepared_corpus_sha256: &consistently_mutated.prepared_corpus_sha256,
                embedding_profile_sha256: &consistently_mutated.embedding_profile.component.sha256,
                tokenizer_sha256: &consistently_mutated.executable_tokenizer.artifact.sha256,
            },
            task.ordinal_start,
            task.ordinal_end,
            &task.embedding_input_order_sha256,
            &task.document_token_counts_sha256,
            task.token_count,
        )
        .unwrap()
        .to_string();
        consistently_mutated.seal().unwrap();
        assert!(
            consistently_mutated
                .validate_with_tokenizer(&prepared, &rows, TEST_TOKENIZER.as_bytes())
                .is_err()
        );

        let mut wrong_revision = tokenizer_ref(128);
        wrong_revision.model_revision = "other".into();
        assert!(
            build_token_balanced_plan(
                &prepared,
                &rows,
                profile(4),
                wrong_revision,
                TEST_TOKENIZER.as_bytes(),
                TokenBalanceOptions {
                    maximum_task_tokens: 4,
                    maximum_task_documents: 4,
                },
            )
            .is_err()
        );
    }

    #[test]
    fn tokenizer_limits_obey_json_safe_integer_boundaries() {
        tokenizer_ref(9_007_199_254_740_991).validate().unwrap();
        assert!(tokenizer_ref(9_007_199_254_740_992).validate().is_err());

        let (prepared, rows) = corpus(&["a"]);
        assert!(
            build_token_balanced_plan(
                &prepared,
                &rows,
                profile(4),
                tokenizer_ref(128),
                TEST_TOKENIZER.as_bytes(),
                TokenBalanceOptions {
                    maximum_task_tokens: 9_007_199_254_740_992,
                    maximum_task_documents: 1,
                },
            )
            .is_err()
        );
    }

    #[test]
    fn token_count_object_round_trip_binds_every_ordinal_and_statistics() {
        let (prepared, rows) = corpus(&["a b", "a", "é 👩‍💻", "b b", "a"]);
        let (plan, counts) = build_token_balanced_plan_with_counts(
            &prepared,
            &rows,
            profile(4),
            tokenizer_ref(128),
            TEST_TOKENIZER.as_bytes(),
            TokenBalanceOptions {
                maximum_task_tokens: 3,
                maximum_task_documents: 3,
            },
        )
        .unwrap();
        assert_eq!(counts, vec![2, 1, 2, 2, 1]);
        assert_eq!(
            plan.token_statistics,
            TokenStatistics {
                total_tokens: 8,
                p50_tokens: 2,
                p90_tokens: 2,
                p95_tokens: 2,
                p99_tokens: 2,
                maximum_tokens: 2,
            }
        );
        assert_eq!(
            plan.document_token_counts_object.path.as_str(),
            DOCUMENT_TOKEN_COUNTS_PATH
        );
        assert_eq!(plan.document_token_counts_object.rows, 5);
        assert_eq!(plan.document_token_counts_object.bytes, 20);

        let root = tempfile::tempdir().unwrap();
        let path = plan
            .write_document_token_counts(root.path(), &counts)
            .unwrap();
        assert_eq!(
            path,
            fs::canonicalize(root.path())
                .unwrap()
                .join(DOCUMENT_TOKEN_COUNTS_PATH)
        );
        assert_eq!(
            plan.read_document_token_counts(root.path()).unwrap(),
            counts
        );
    }

    #[test]
    fn token_count_object_rejects_tampering_and_wrong_statistics() {
        let (prepared, rows) = corpus(&["a", "a b"]);
        let (plan, counts) = build_token_balanced_plan_with_counts(
            &prepared,
            &rows,
            profile(4),
            tokenizer_ref(128),
            TEST_TOKENIZER.as_bytes(),
            TokenBalanceOptions {
                maximum_task_tokens: 4,
                maximum_task_documents: 4,
            },
        )
        .unwrap();
        assert!(plan.validate_document_token_counts(&[1, 1]).is_err());

        let mut wrong_statistics = plan.clone();
        wrong_statistics.token_statistics.p50_tokens = 2;
        wrong_statistics.component_sha256 = component_digest(&wrong_statistics).unwrap();
        assert!(
            wrong_statistics
                .validate_document_token_counts(&counts)
                .is_err()
        );

        let root = tempfile::tempdir().unwrap();
        let path = plan
            .write_document_token_counts(root.path(), &counts)
            .unwrap();
        fs::write(path, encode_document_token_counts(&[1, 1])).unwrap();
        assert!(plan.read_document_token_counts(root.path()).is_err());
    }

    #[test]
    fn tasks_cross_object_boundaries_with_exact_per_slice_counts() {
        let (mut prepared, rows) = corpus(&["a", "a b", "é"]);
        prepared.documents = [0..1, 1..3]
            .into_iter()
            .enumerate()
            .map(|(ordinal, range)| {
                let shard = &rows[range.clone()];
                PreparedDocumentObject {
                    object: ObjectEntry {
                        path: SafeRelativePath::new(format!("documents/part-{ordinal}.parquet"))
                            .unwrap(),
                        rows: shard.len() as u64,
                        bytes: 1,
                        sha256: digest(char::from_digit(ordinal as u32 + 1, 16).unwrap()),
                        logical_order_sha256: canonical_digest(&shard).unwrap(),
                    },
                    ordinal: ordinal as u32,
                    first_document_id: shard.first().unwrap().document_id.clone(),
                    last_document_id: shard.last().unwrap().document_id.clone(),
                    embedding_input_order_sha256: embedding_input_order_digest(shard),
                }
            })
            .collect();
        prepared.seal().unwrap();

        let plan = build_token_balanced_plan(
            &prepared,
            &rows,
            profile(4),
            tokenizer_ref(128),
            TEST_TOKENIZER.as_bytes(),
            TokenBalanceOptions {
                maximum_task_tokens: 8,
                maximum_task_documents: 8,
            },
        )
        .unwrap();
        assert_eq!(plan.tasks.len(), 1);
        assert_eq!(plan.tasks[0].input_slices.len(), 2);
        assert_eq!(plan.tasks[0].input_slices[0].token_count, 1);
        assert_eq!(plan.tasks[0].input_slices[1].token_count, 3);
        plan.validate_with_tokenizer(&prepared, &rows, TEST_TOKENIZER.as_bytes())
            .unwrap();
    }

    #[test]
    fn v1_receipt_wire_contract_closes_a_v2_plan() {
        let (prepared, rows) = corpus(&["a", "a b"]);
        let plan = build_token_balanced_plan(
            &prepared,
            &rows,
            profile(4),
            tokenizer_ref(128),
            TEST_TOKENIZER.as_bytes(),
            TokenBalanceOptions {
                maximum_task_tokens: 4,
                maximum_task_documents: 4,
            },
        )
        .unwrap();
        let task = &plan.tasks[0];
        let vector_bytes = 64 + task.row_count() * u64::from(plan.embedding_profile.dimensions) * 4;
        let mut receipt = VectorResultReceipt {
            schema_version: crate::VECTOR_RECEIPT_SCHEMA.into(),
            component_sha256: digest('0'),
            plan_sha256: plan.component_sha256.clone(),
            prepared_corpus_sha256: plan.prepared_corpus_sha256.clone(),
            embedding_profile_sha256: plan.embedding_profile.component.sha256.clone(),
            task_id: task.task_id.clone(),
            ordinal_start: task.ordinal_start,
            ordinal_end: task.ordinal_end,
            embedding_input_order_sha256: task.embedding_input_order_sha256.clone(),
            vector: VectorObject {
                path: task.result_path.clone(),
                rows: task.row_count(),
                bytes: vector_bytes,
                sha256: digest('d'),
                dimensions: plan.embedding_profile.dimensions,
                dtype: plan.embedding_profile.dtype.clone(),
                embedding_input_order_sha256: task.embedding_input_order_sha256.clone(),
            },
            executor: ExecutorReceipt {
                implementation: component("executor", "1", digest('e')),
                runtime: component("runtime", "1", digest('f')),
                returned_model: "qwen".into(),
                requests: 1,
                retries: 0,
                input_bytes_upper_bound: 8,
                elapsed_ms: 1,
                conformance_passed: true,
            },
            derivation: None,
            finite_values_validated: true,
            normalization_validated: true,
        };
        receipt.seal().unwrap();
        receipt.validate_against_v2(&plan).unwrap();

        let mut result_set = EmbeddingResultSetManifest {
            schema_version: crate::RESULT_SET_SCHEMA.into(),
            component_sha256: digest('0'),
            plan_sha256: plan.component_sha256.clone(),
            prepared_corpus_sha256: plan.prepared_corpus_sha256.clone(),
            embedding_profile_sha256: plan.embedding_profile.component.sha256.clone(),
            document_count: plan.document_count,
            document_order_sha256: plan.document_order_sha256.clone(),
            receipts: vec![ReceiptEntry {
                task_id: task.task_id.clone(),
                path: task.receipt_path.clone(),
                sha256: receipt.component_sha256.clone(),
            }],
            derivation: None,
            test_only: false,
        };
        result_set.seal().unwrap();
        result_set.validate_v2(&plan, &[receipt]).unwrap();
    }

    #[test]
    fn result_set_rejects_mixed_executor_implementations() {
        let (prepared, rows) = corpus(&["a", "a b"]);
        let plan = build_token_balanced_plan(
            &prepared,
            &rows,
            profile(4),
            tokenizer_ref(128),
            TEST_TOKENIZER.as_bytes(),
            TokenBalanceOptions {
                maximum_task_tokens: 4,
                maximum_task_documents: 1,
            },
        )
        .unwrap();
        assert_eq!(plan.tasks.len(), 2);
        let receipt_for = |task: &EmbeddingTaskV2, implementation_digest| {
            let mut receipt = VectorResultReceipt {
                schema_version: crate::VECTOR_RECEIPT_SCHEMA.into(),
                component_sha256: digest('0'),
                plan_sha256: plan.component_sha256.clone(),
                prepared_corpus_sha256: plan.prepared_corpus_sha256.clone(),
                embedding_profile_sha256: plan.embedding_profile.component.sha256.clone(),
                task_id: task.task_id.clone(),
                ordinal_start: task.ordinal_start,
                ordinal_end: task.ordinal_end,
                embedding_input_order_sha256: task.embedding_input_order_sha256.clone(),
                vector: VectorObject {
                    path: task.result_path.clone(),
                    rows: task.row_count(),
                    bytes: 64 + task.row_count() * u64::from(plan.embedding_profile.dimensions) * 4,
                    sha256: digest('d'),
                    dimensions: plan.embedding_profile.dimensions,
                    dtype: plan.embedding_profile.dtype.clone(),
                    embedding_input_order_sha256: task.embedding_input_order_sha256.clone(),
                },
                executor: ExecutorReceipt {
                    implementation: component("executor", "1", digest(implementation_digest)),
                    runtime: component("runtime", "1", digest('f')),
                    returned_model: "qwen".into(),
                    requests: 1,
                    retries: 0,
                    input_bytes_upper_bound: 8,
                    elapsed_ms: 1,
                    conformance_passed: true,
                },
                derivation: None,
                finite_values_validated: true,
                normalization_validated: true,
            };
            receipt.seal().unwrap();
            receipt.validate_against_v2(&plan).unwrap();
            receipt
        };
        let receipts = vec![
            receipt_for(&plan.tasks[0], 'e'),
            receipt_for(&plan.tasks[1], 'a'),
        ];
        let mut result_set = EmbeddingResultSetManifest {
            schema_version: crate::RESULT_SET_SCHEMA.into(),
            component_sha256: digest('0'),
            plan_sha256: plan.component_sha256.clone(),
            prepared_corpus_sha256: plan.prepared_corpus_sha256.clone(),
            embedding_profile_sha256: plan.embedding_profile.component.sha256.clone(),
            document_count: plan.document_count,
            document_order_sha256: plan.document_order_sha256.clone(),
            receipts: plan
                .tasks
                .iter()
                .zip(&receipts)
                .map(|(task, receipt)| ReceiptEntry {
                    task_id: task.task_id.clone(),
                    path: task.receipt_path.clone(),
                    sha256: receipt.component_sha256.clone(),
                })
                .collect(),
            derivation: None,
            test_only: false,
        };
        result_set.seal().unwrap();
        assert!(matches!(
            result_set.validate_v2(&plan, &receipts),
            Err(PipelineError::Invalid(
                "result executor implementation differs"
            ))
        ));

        let matching = vec![
            receipt_for(&plan.tasks[0], 'e'),
            receipt_for(&plan.tasks[1], 'e'),
        ];
        result_set.receipts = plan
            .tasks
            .iter()
            .zip(&matching)
            .map(|(task, receipt)| ReceiptEntry {
                task_id: task.task_id.clone(),
                path: task.receipt_path.clone(),
                sha256: receipt.component_sha256.clone(),
            })
            .collect();
        result_set.seal().unwrap();
        result_set.validate_v2(&plan, &matching).unwrap();
    }

    #[test]
    fn test_only_result_requires_test_executor_and_v2_manifest() {
        let (prepared, rows) = corpus(&["a"]);
        let plan = build_token_balanced_plan(
            &prepared,
            &rows,
            profile(32),
            tokenizer_ref(4_096),
            TEST_TOKENIZER.as_bytes(),
            TokenBalanceOptions {
                maximum_task_tokens: 32,
                maximum_task_documents: 8,
            },
        )
        .unwrap();
        let task = &plan.tasks[0];
        let mut receipt = VectorResultReceipt {
            schema_version: crate::VECTOR_RECEIPT_SCHEMA.into(),
            component_sha256: digest('0'),
            plan_sha256: plan.component_sha256.clone(),
            prepared_corpus_sha256: plan.prepared_corpus_sha256.clone(),
            embedding_profile_sha256: plan.embedding_profile.component.sha256.clone(),
            task_id: task.task_id.clone(),
            ordinal_start: task.ordinal_start,
            ordinal_end: task.ordinal_end,
            embedding_input_order_sha256: task.embedding_input_order_sha256.clone(),
            vector: VectorObject {
                path: task.result_path.clone(),
                rows: task.row_count(),
                bytes: 64 + task.row_count() * u64::from(plan.embedding_profile.dimensions) * 4,
                sha256: digest('9'),
                dimensions: plan.embedding_profile.dimensions,
                dtype: "f32le".into(),
                embedding_input_order_sha256: task.embedding_input_order_sha256.clone(),
            },
            executor: ExecutorReceipt {
                implementation: component(crate::TEST_VECTOR_EXECUTOR_ID, "1", digest('8')),
                runtime: component("runtime", "1", digest('7')),
                returned_model: "qwen".into(),
                requests: 0,
                retries: 0,
                input_bytes_upper_bound: 1,
                elapsed_ms: 0,
                conformance_passed: false,
            },
            derivation: None,
            finite_values_validated: true,
            normalization_validated: true,
        };
        receipt.seal().unwrap();
        receipt.validate_against_v2(&plan).unwrap();
        let mut result = EmbeddingResultSetManifest {
            schema_version: crate::TEST_RESULT_SET_SCHEMA.into(),
            component_sha256: digest('0'),
            plan_sha256: plan.component_sha256.clone(),
            prepared_corpus_sha256: plan.prepared_corpus_sha256.clone(),
            embedding_profile_sha256: plan.embedding_profile.component.sha256.clone(),
            document_count: plan.document_count,
            document_order_sha256: plan.document_order_sha256.clone(),
            receipts: vec![ReceiptEntry {
                task_id: task.task_id.clone(),
                path: task.receipt_path.clone(),
                sha256: receipt.component_sha256.clone(),
            }],
            derivation: None,
            test_only: true,
        };
        result.seal().unwrap();
        result.validate_v2(&plan, &[receipt.clone()]).unwrap();

        result.test_only = false;
        result.schema_version = crate::RESULT_SET_SCHEMA.into();
        result.seal().unwrap();
        assert!(result.validate_v2(&plan, &[receipt]).is_err());
    }

    #[derive(Deserialize)]
    struct TokenizerParityFixture {
        cases: Vec<TokenizerParityCase>,
        generated_cases: Vec<GeneratedTokenizerParityCase>,
    }

    #[derive(Deserialize)]
    struct TokenizerParityCase {
        name: String,
        input: String,
        token_ids: Vec<u32>,
    }

    #[derive(Deserialize)]
    struct GeneratedTokenizerParityCase {
        name: String,
        repeat: String,
        count: usize,
        token_count: usize,
        token_ids_u32le_sha256: String,
    }

    #[test]
    #[ignore = "requires the pinned 11 MB Qwen tokenizer.json artifact"]
    fn pinned_qwen_tokenizer_matches_llama_cpp_token_ids() {
        let tokenizer_path = std::env::var("LIVEFIRE_QWEN_TOKENIZER_JSON")
            .expect("LIVEFIRE_QWEN_TOKENIZER_JSON must name the pinned tokenizer.json");
        let tokenizer_bytes = std::fs::read(tokenizer_path).unwrap();
        let reference: ExecutableTokenizerRef = serde_json::from_slice(include_bytes!(
            "../../../profiles/qwen3-embedding-8b-gguf-q4-k-m-tokenizer.ref.json"
        ))
        .unwrap();
        let tokenizer = ExactTokenizer::from_bytes(reference, &tokenizer_bytes).unwrap();
        let fixture: TokenizerParityFixture = serde_json::from_slice(include_bytes!(
            "../../../fixtures/qwen3-embedding-8b-tokenizer-parity.v1.json"
        ))
        .unwrap();

        for case in fixture.cases {
            assert_eq!(
                tokenizer.token_ids(&case.input).unwrap(),
                case.token_ids,
                "{}",
                case.name
            );
        }
        for case in fixture.generated_cases {
            let input = case.repeat.repeat(case.count);
            let token_ids = tokenizer.token_ids(&input).unwrap();
            assert_eq!(token_ids.len(), case.token_count, "{}", case.name);
            let bytes = token_ids
                .iter()
                .flat_map(|token| token.to_le_bytes())
                .collect::<Vec<_>>();
            assert_eq!(
                super::super::digest_bytes(&bytes).as_str(),
                case.token_ids_u32le_sha256,
                "{}",
                case.name
            );
        }
    }
}

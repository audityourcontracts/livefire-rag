//! Sealed query vectors for exact, offline dense and fused search.
//!
//! A set is bound to the byte-exact frozen JSONL query plan that produced it,
//! the embedding profile and policy, and the complete cloud execution
//! identity. Search callers select a vector by query ID and must also supply
//! the original and composed query strings; vectors are never accepted in a
//! search request.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest as ShaDigest, Sha256};

use super::{
    AtomicDirectory, ComponentRef, Digest, PipelineError, Result, RunpodExecutionIdentity,
    SafeRelativePath, canonical_digest, component_digest, digest_bytes, read_json,
    resolve_existing_artifact, write_canonical_json,
};

pub const QUERY_VECTOR_SET_SCHEMA: &str = "livefire.rag.query-vector-set/1";
pub const QUERY_VECTOR_SET_MANIFEST: &str = "manifest.json";
pub const QUERY_VECTOR_SET_PLAN: &str = "queries.jsonl";
pub const QUERY_VECTOR_SET_VECTORS: &str = "vectors.f32le";
const MAX_QUERY_PLAN_ROWS: usize = 10_000;
const MAX_QUERY_ID_BYTES: usize = 128;
const MAX_QUERY_BYTES: usize = 8_192;
const MAX_QUERY_PLAN_BYTES: u64 = 128 * 1024 * 1024;
const MAX_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
const MAX_VECTOR_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueryVectorArtifact {
    pub path: SafeRelativePath,
    pub bytes: u64,
    pub sha256: Digest,
}

impl QueryVectorArtifact {
    fn validate(&self, expected_path: &str) -> Result<()> {
        if self.path.as_str() != expected_path
            || self.bytes == 0
            || self.bytes > MAX_QUERY_PLAN_BYTES
        {
            return Err(PipelineError::Invalid("query vector set artifact"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackedQueryVectors {
    pub path: SafeRelativePath,
    pub bytes: u64,
    pub sha256: Digest,
    pub rows: u32,
    pub dimensions: u32,
    pub dtype: String,
    pub normalization: String,
    pub order_sha256: Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueryVectorExecutionBinding {
    pub embedding_profile: ComponentRef,
    pub embedding_policy: ComponentRef,
    pub execution_identity_sha256: Digest,
    pub execution: RunpodExecutionIdentity,
    pub executor_image_build_receipt: ComponentRef,
}

impl QueryVectorExecutionBinding {
    pub fn validate(&self) -> Result<()> {
        self.embedding_profile.validate()?;
        self.embedding_policy.validate()?;
        self.executor_image_build_receipt.validate()?;
        for component in [
            &self.execution.executor_image,
            &self.execution.executor_image_build,
            &self.execution.runtime,
            &self.execution.worker_binary,
            &self.execution.model_artifact,
            &self.execution.embedding_profile,
        ] {
            component.validate()?;
        }
        for value in [
            self.execution.accelerator.provider.as_str(),
            self.execution.accelerator.model.as_str(),
            self.execution.accelerator.architecture.as_str(),
            self.execution.accelerator.compute_capability.as_str(),
            self.execution.returned_model.as_str(),
        ] {
            if value.trim().is_empty() {
                return Err(PipelineError::Invalid("query vector execution identity"));
            }
        }
        if self.execution.accelerator.count != 1
            || self.execution.embedding_profile != self.embedding_profile
            || self.execution.executor_image_build != self.executor_image_build_receipt
            || self.execution_identity_sha256 != canonical_digest(&self.execution)?
        {
            return Err(PipelineError::Invalid("query vector execution binding"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueryVectorRow {
    pub ordinal: u32,
    pub query_id: String,
    pub raw_query_sha256: Digest,
    pub composed_query_sha256: Digest,
    pub vector_sha256: Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueryVectorSetManifest {
    pub schema_version: String,
    pub component_sha256: Digest,
    pub status: String,
    pub query_plan: QueryVectorArtifact,
    pub request_rows: u32,
    pub semantic_request_rows: u32,
    pub execution: QueryVectorExecutionBinding,
    pub vectors: PackedQueryVectors,
    pub queries: Vec<QueryVectorRow>,
}

impl QueryVectorSetManifest {
    pub fn validate(&self) -> Result<()> {
        self.query_plan.validate(QUERY_VECTOR_SET_PLAN)?;
        self.execution.validate()?;
        if self.schema_version != QUERY_VECTOR_SET_SCHEMA
            || self.status != "complete"
            || self.request_rows == 0
            || self.semantic_request_rows == 0
            || self.semantic_request_rows > self.request_rows
            || self.queries.is_empty()
            || usize::try_from(self.vectors.rows).ok() != Some(self.queries.len())
            || self.vectors.path.as_str() != QUERY_VECTOR_SET_VECTORS
            || self.vectors.dtype != "f32le"
            || !matches!(self.vectors.normalization.as_str(), "l2" | "none")
            || self.vectors.dimensions == 0
            || self.vectors.bytes == 0
            || self.vectors.bytes > MAX_VECTOR_BYTES
            || self.vectors.bytes
                != u64::from(self.vectors.rows)
                    .checked_mul(u64::from(self.vectors.dimensions))
                    .and_then(|values| values.checked_mul(4))
                    .ok_or(PipelineError::Invalid("query vector byte count"))?
        {
            return Err(PipelineError::Invalid("query vector set manifest"));
        }
        let mut ids = BTreeSet::new();
        for (ordinal, row) in self.queries.iter().enumerate() {
            if usize::try_from(row.ordinal).ok() != Some(ordinal)
                || !valid_query_id(&row.query_id)
                || !ids.insert(row.query_id.as_str())
            {
                return Err(PipelineError::Invalid("query vector row order"));
            }
        }
        if self.vectors.order_sha256 != query_vector_order_digest(&self.queries)
            || self.component_sha256 != component_digest(self)?
        {
            return Err(PipelineError::Invalid("query vector set binding"));
        }
        Ok(())
    }

    pub fn seal(&mut self) -> Result<()> {
        self.component_sha256 = component_digest(self)?;
        self.validate()
    }
}

/// One vector and its exact query surfaces supplied by the producer.
pub struct QueryVectorSetInput<'a> {
    pub query_id: &'a str,
    pub raw_query: &'a str,
    pub composed_query: &'a str,
    pub vector: &'a [f32],
}

/// Unique semantic queries in first-occurrence order from a validated frozen
/// request plan. Dense and fused rows sharing one query ID appear once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryVectorPlanQuery {
    pub query_id: String,
    pub query: String,
}

pub fn query_vector_plan_queries(bytes: &[u8]) -> Result<Vec<QueryVectorPlanQuery>> {
    Ok(parse_query_plan(bytes)?
        .unique_semantic_queries
        .into_iter()
        .map(|query| QueryVectorPlanQuery {
            query_id: query.query_id,
            query: query.query,
        })
        .collect())
}

/// A fully verified sealed set. Vector slices remain bound to the manifest's
/// immutable row order.
#[derive(Debug)]
pub struct SealedQueryVectorSet {
    pub manifest: QueryVectorSetManifest,
    root: PathBuf,
    vectors: Vec<f32>,
    by_query_id: BTreeMap<String, usize>,
    request_surfaces: BTreeSet<(String, String, u64, Vec<String>)>,
}

impl SealedQueryVectorSet {
    /// Open and validate the complete artifact tree against the profile used
    /// by the target index or catalogue.
    pub fn open(
        root: &Path,
        expected_profile: &ComponentRef,
        expected_returned_model: &str,
        expected_dimensions: u32,
        expected_normalization: &str,
        expected_query_plan: Option<&Path>,
    ) -> Result<Self> {
        let root = fs::canonicalize(root)?;
        let manifest_path =
            resolve_existing_artifact(&root, &SafeRelativePath::new(QUERY_VECTOR_SET_MANIFEST)?)?;
        let manifest_metadata = fs::metadata(&manifest_path)?;
        if !manifest_metadata.is_file()
            || manifest_metadata.len() == 0
            || manifest_metadata.len() > MAX_MANIFEST_BYTES
        {
            return Err(PipelineError::Invalid("query vector set manifest file"));
        }
        let manifest: QueryVectorSetManifest = read_json(&manifest_path)?;
        manifest.validate()?;
        if &manifest.execution.embedding_profile != expected_profile
            || manifest.execution.execution.returned_model != expected_returned_model
            || manifest.vectors.dimensions != expected_dimensions
            || manifest.vectors.normalization != expected_normalization
        {
            return Err(PipelineError::Invalid("query vector profile binding"));
        }
        let plan_path = validate_artifact(&root, &manifest.query_plan)?;
        if let Some(expected) = expected_query_plan {
            let expected = fs::canonicalize(expected)?;
            if fs::metadata(&expected)?.len() != manifest.query_plan.bytes
                || file_digest(&expected)? != manifest.query_plan.sha256
            {
                return Err(PipelineError::Invalid("query vector query plan binding"));
            }
        }
        let plan = parse_query_plan(&fs::read(plan_path)?)?;
        if plan.request_rows != manifest.request_rows
            || plan.semantic_request_rows != manifest.semantic_request_rows
            || plan.unique_semantic_queries.len() != manifest.queries.len()
        {
            return Err(PipelineError::Invalid("query vector query plan coverage"));
        }
        for (planned, row) in plan.unique_semantic_queries.iter().zip(&manifest.queries) {
            if planned.query_id != row.query_id
                || digest_bytes(planned.query.as_bytes()) != row.raw_query_sha256
            {
                return Err(PipelineError::Invalid("query vector query plan order"));
            }
        }
        let vector_path = resolve_existing_artifact(&root, &manifest.vectors.path)?;
        let vector_metadata = fs::metadata(&vector_path)?;
        if !vector_metadata.is_file()
            || vector_metadata.len() != manifest.vectors.bytes
            || file_digest(&vector_path)? != manifest.vectors.sha256
        {
            return Err(PipelineError::Invalid("query vector packed object"));
        }
        let vector_bytes = fs::read(vector_path)?;
        let vectors = decode_vectors(&vector_bytes)?;
        let dimensions = usize::try_from(manifest.vectors.dimensions)
            .map_err(|_| PipelineError::Invalid("query vector dimensions"))?;
        for (ordinal, values) in vectors.chunks_exact(dimensions).enumerate() {
            validate_vector(values, &manifest.vectors.normalization)?;
            if vector_digest(values) != manifest.queries[ordinal].vector_sha256 {
                return Err(PipelineError::Invalid("query vector row digest"));
            }
        }
        let by_query_id = manifest
            .queries
            .iter()
            .enumerate()
            .map(|(ordinal, row)| (row.query_id.clone(), ordinal))
            .collect();
        Ok(Self {
            manifest,
            root,
            vectors,
            by_query_id,
            request_surfaces: plan.surfaces,
        })
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Return a sealed vector only when the query ID, raw query, and composed
    /// query all match the manifest row exactly.
    fn vector_for<'a>(
        &'a self,
        query_id: &str,
        raw_query: &str,
        composed_query: &str,
    ) -> Result<&'a [f32]> {
        let ordinal = *self
            .by_query_id
            .get(query_id)
            .ok_or(PipelineError::Invalid("query is not in sealed vector set"))?;
        let row = &self.manifest.queries[ordinal];
        if row.raw_query_sha256 != digest_bytes(raw_query.as_bytes())
            || row.composed_query_sha256 != digest_bytes(composed_query.as_bytes())
        {
            return Err(PipelineError::Invalid("sealed query surface binding"));
        }
        let dimensions = usize::try_from(self.manifest.vectors.dimensions)
            .map_err(|_| PipelineError::Invalid("query vector dimensions"))?;
        let start = ordinal
            .checked_mul(dimensions)
            .ok_or(PipelineError::Invalid("query vector offset"))?;
        let end = start
            .checked_add(dimensions)
            .ok_or(PipelineError::Invalid("query vector offset"))?;
        self.vectors
            .get(start..end)
            .ok_or(PipelineError::Invalid("query vector coverage"))
    }

    /// Select a vector only for one complete request surface present in the
    /// frozen JSONL plan. Relations must use the plan's sorted order.
    pub fn vector_for_request<'a>(
        &'a self,
        query_id: &str,
        raw_query: &str,
        composed_query: &str,
        mode: &str,
        top_n: usize,
        relations: &[String],
    ) -> Result<&'a [f32]> {
        let top_n = u64::try_from(top_n)
            .map_err(|_| PipelineError::Invalid("sealed query request surface"))?;
        if !self.request_surfaces.contains(&(
            query_id.to_owned(),
            mode.to_owned(),
            top_n,
            relations.to_vec(),
        )) {
            return Err(PipelineError::Invalid("sealed query request surface"));
        }
        self.vector_for(query_id, raw_query, composed_query)
    }

    /// Select a vector without accepting a client-supplied query ID. Exactly
    /// one frozen semantic query must match the raw/composed query hashes and
    /// the complete request surface. This is the provider boundary: a caller
    /// can name ordinary search arguments, but cannot choose a vector row.
    pub fn vector_for_unique_request<'a>(
        &'a self,
        raw_query: &str,
        composed_query: &str,
        mode: &str,
        top_n: usize,
        relations: &[String],
    ) -> Result<&'a [f32]> {
        let top_n = u64::try_from(top_n)
            .map_err(|_| PipelineError::Invalid("sealed query request surface"))?;
        let raw_digest = digest_bytes(raw_query.as_bytes());
        let composed_digest = digest_bytes(composed_query.as_bytes());
        let mut ordinal = None;
        for (candidate, row) in self.manifest.queries.iter().enumerate() {
            if row.raw_query_sha256 == raw_digest
                && row.composed_query_sha256 == composed_digest
                && self.request_surfaces.contains(&(
                    row.query_id.clone(),
                    mode.to_owned(),
                    top_n,
                    relations.to_vec(),
                ))
                && ordinal.replace(candidate).is_some()
            {
                return Err(PipelineError::Invalid("ambiguous sealed query request"));
            }
        }
        let ordinal = ordinal.ok_or(PipelineError::Invalid("query is not in sealed vector set"))?;
        let dimensions = usize::try_from(self.manifest.vectors.dimensions)
            .map_err(|_| PipelineError::Invalid("query vector dimensions"))?;
        let start = ordinal
            .checked_mul(dimensions)
            .ok_or(PipelineError::Invalid("query vector offset"))?;
        let end = start
            .checked_add(dimensions)
            .ok_or(PipelineError::Invalid("query vector offset"))?;
        self.vectors
            .get(start..end)
            .ok_or(PipelineError::Invalid("query vector coverage"))
    }
}

pub fn write_query_vector_set(
    destination: &Path,
    query_plan: &Path,
    execution: QueryVectorExecutionBinding,
    dimensions: u32,
    normalization: &str,
    inputs: &[QueryVectorSetInput<'_>],
) -> Result<QueryVectorSetManifest> {
    execution.validate()?;
    if inputs.is_empty()
        || inputs.len() > MAX_QUERY_PLAN_ROWS
        || dimensions == 0
        || !matches!(normalization, "l2" | "none")
    {
        return Err(PipelineError::Invalid("query vector set input"));
    }
    let query_plan_bytes = fs::read(query_plan)?;
    if query_plan_bytes.len() > usize::try_from(MAX_QUERY_PLAN_BYTES).unwrap_or(usize::MAX) {
        return Err(PipelineError::Invalid("query vector query plan size"));
    }
    let parsed = parse_query_plan(&query_plan_bytes)?;
    if parsed.unique_semantic_queries.len() != inputs.len() {
        return Err(PipelineError::Invalid("query vector query plan coverage"));
    }
    let staging = AtomicDirectory::new(destination)?;
    let plan_path = staging.path().join(QUERY_VECTOR_SET_PLAN);
    fs::write(&plan_path, &query_plan_bytes)?;
    let vector_path = staging.path().join(QUERY_VECTOR_SET_VECTORS);
    let mut vector_writer = BufWriter::new(File::create(&vector_path)?);
    let mut rows = Vec::with_capacity(inputs.len());
    for (ordinal, (planned, input)) in parsed
        .unique_semantic_queries
        .iter()
        .zip(inputs)
        .enumerate()
    {
        if planned.query_id != input.query_id || planned.query != input.raw_query {
            return Err(PipelineError::Invalid("query vector query plan order"));
        }
        if input.composed_query.is_empty()
            || input.composed_query.len() > 16_384
            || input.vector.len() != usize::try_from(dimensions).unwrap_or(usize::MAX)
        {
            return Err(PipelineError::Invalid("query vector set input"));
        }
        validate_vector(input.vector, normalization)?;
        for value in input.vector {
            vector_writer.write_all(&value.to_le_bytes())?;
        }
        rows.push(QueryVectorRow {
            ordinal: u32::try_from(ordinal)
                .map_err(|_| PipelineError::Invalid("query vector ordinal"))?,
            query_id: input.query_id.to_owned(),
            raw_query_sha256: digest_bytes(input.raw_query.as_bytes()),
            composed_query_sha256: digest_bytes(input.composed_query.as_bytes()),
            vector_sha256: vector_digest(input.vector),
        });
    }
    vector_writer.flush()?;
    drop(vector_writer);
    let vector_bytes = fs::metadata(&vector_path)?.len();
    if vector_bytes == 0 || vector_bytes > MAX_VECTOR_BYTES {
        return Err(PipelineError::Invalid("query vector byte count"));
    }
    let mut manifest = QueryVectorSetManifest {
        schema_version: QUERY_VECTOR_SET_SCHEMA.into(),
        component_sha256: Digest::new("0".repeat(64))?,
        status: "complete".into(),
        query_plan: QueryVectorArtifact {
            path: SafeRelativePath::new(QUERY_VECTOR_SET_PLAN)?,
            bytes: u64::try_from(query_plan_bytes.len())
                .map_err(|_| PipelineError::Invalid("query plan byte count"))?,
            sha256: digest_bytes(&query_plan_bytes),
        },
        request_rows: parsed.request_rows,
        semantic_request_rows: parsed.semantic_request_rows,
        execution,
        vectors: PackedQueryVectors {
            path: SafeRelativePath::new(QUERY_VECTOR_SET_VECTORS)?,
            bytes: vector_bytes,
            sha256: file_digest(&vector_path)?,
            rows: u32::try_from(rows.len())
                .map_err(|_| PipelineError::Invalid("query vector row count"))?,
            dimensions,
            dtype: "f32le".into(),
            normalization: normalization.into(),
            order_sha256: query_vector_order_digest(&rows),
        },
        queries: rows,
    };
    manifest.seal()?;
    write_canonical_json(&staging.path().join(QUERY_VECTOR_SET_MANIFEST), &manifest)?;
    staging.publish()?;
    Ok(manifest)
}

#[derive(Debug)]
struct PlannedQuery {
    query_id: String,
    query: String,
}

struct ParsedQueryPlan {
    request_rows: u32,
    semantic_request_rows: u32,
    unique_semantic_queries: Vec<PlannedQuery>,
    surfaces: BTreeSet<(String, String, u64, Vec<String>)>,
}

fn parse_query_plan(bytes: &[u8]) -> Result<ParsedQueryPlan> {
    if bytes.is_empty() || bytes.last() != Some(&b'\n') {
        return Err(PipelineError::Invalid("query vector query plan"));
    }
    let text = std::str::from_utf8(bytes)
        .map_err(|_| PipelineError::Invalid("query vector query plan"))?;
    let mut request_rows = 0_u32;
    let mut semantic_request_rows = 0_u32;
    let mut query_by_id = BTreeMap::<String, (String, u64, Vec<String>)>::new();
    let mut mode_surfaces = BTreeSet::<(String, String)>::new();
    let mut surfaces = BTreeSet::<(String, String, u64, Vec<String>)>::new();
    let mut unique_semantic_queries = Vec::new();
    for line in text.lines() {
        request_rows = request_rows
            .checked_add(1)
            .ok_or(PipelineError::Invalid("query vector query plan rows"))?;
        if usize::try_from(request_rows).unwrap_or(usize::MAX) > MAX_QUERY_PLAN_ROWS {
            return Err(PipelineError::Invalid("query vector query plan rows"));
        }
        let row: Value = serde_json::from_str(line)?;
        let object = row
            .as_object()
            .ok_or(PipelineError::Invalid("query vector query plan row"))?;
        let query_id = object
            .get("query_id")
            .and_then(Value::as_str)
            .ok_or(PipelineError::Invalid("query vector query ID"))?;
        let query = object
            .get("query")
            .and_then(Value::as_str)
            .ok_or(PipelineError::Invalid("query vector raw query"))?;
        let mode = object
            .get("mode")
            .and_then(Value::as_str)
            .ok_or(PipelineError::Invalid("query vector query mode"))?;
        let top_n = object
            .get("top_n")
            .and_then(Value::as_u64)
            .ok_or(PipelineError::Invalid("query vector query top_n"))?;
        let relations = object
            .get("relations")
            .and_then(Value::as_array)
            .ok_or(PipelineError::Invalid("query vector query relations"))?
            .iter()
            .map(|relation| {
                relation
                    .as_str()
                    .map(str::to_owned)
                    .ok_or(PipelineError::Invalid("query vector query relation"))
            })
            .collect::<Result<Vec<_>>>()?;
        if !valid_query_id(query_id)
            || query.trim().is_empty()
            || query.len() > MAX_QUERY_BYTES
            || !matches!(mode, "dense" | "lexical" | "fused")
            || top_n == 0
            || top_n > 100
            || object.len() != 5
            || relations.len() > 64
            || relations
                .iter()
                .any(|relation| relation.trim().is_empty() || relation.len() > 256)
            || !relations
                .windows(2)
                .all(|pair| pair[0].as_str() < pair[1].as_str())
            || !mode_surfaces.insert((query_id.to_owned(), mode.to_owned()))
        {
            return Err(PipelineError::Invalid("query vector query plan row"));
        }
        surfaces.insert((
            query_id.to_owned(),
            mode.to_owned(),
            top_n,
            relations.clone(),
        ));
        if let Some(existing) = query_by_id.get(query_id) {
            if existing != &(query.to_owned(), top_n, relations.clone()) {
                return Err(PipelineError::Invalid("query vector query ID reuse"));
            }
        } else {
            query_by_id.insert(query_id.to_owned(), (query.to_owned(), top_n, relations));
        }
        if matches!(mode, "dense" | "fused") {
            semantic_request_rows = semantic_request_rows
                .checked_add(1)
                .ok_or(PipelineError::Invalid("query vector query plan rows"))?;
            if !unique_semantic_queries
                .iter()
                .any(|planned: &PlannedQuery| planned.query_id == query_id)
            {
                unique_semantic_queries.push(PlannedQuery {
                    query_id: query_id.to_owned(),
                    query: query.to_owned(),
                });
            }
        }
    }
    if request_rows == 0 || semantic_request_rows == 0 {
        return Err(PipelineError::Invalid("query vector query plan rows"));
    }
    Ok(ParsedQueryPlan {
        request_rows,
        semantic_request_rows,
        unique_semantic_queries,
        surfaces,
    })
}

fn valid_query_id(value: &str) -> bool {
    !value.trim().is_empty() && value.len() <= MAX_QUERY_ID_BYTES
}

fn validate_artifact(root: &Path, artifact: &QueryVectorArtifact) -> Result<PathBuf> {
    let path = resolve_existing_artifact(root, &artifact.path)?;
    let metadata = fs::metadata(&path)?;
    if !metadata.is_file()
        || metadata.len() != artifact.bytes
        || file_digest(&path)? != artifact.sha256
    {
        return Err(PipelineError::Invalid("query vector set artifact binding"));
    }
    Ok(path)
}

fn file_digest(path: &Path) -> Result<Digest> {
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
    Digest::new(format!("{:x}", hasher.finalize()))
}

fn decode_vectors(bytes: &[u8]) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(PipelineError::Invalid("query vector packed object"));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn validate_vector(vector: &[f32], normalization: &str) -> Result<()> {
    if vector.is_empty() || vector.iter().any(|value| !value.is_finite()) {
        return Err(PipelineError::Invalid("query vector values"));
    }
    if normalization == "l2" {
        let norm = vector
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            .sqrt();
        if (norm - 1.0).abs() > 1.0e-4 {
            return Err(PipelineError::Invalid("query vector normalization"));
        }
    }
    Ok(())
}

fn vector_digest(vector: &[f32]) -> Digest {
    let mut hasher = Sha256::new();
    for value in vector {
        hasher.update(value.to_le_bytes());
    }
    Digest::new(format!("{:x}", hasher.finalize())).expect("SHA-256 is valid")
}

fn query_vector_order_digest(rows: &[QueryVectorRow]) -> Digest {
    let mut hasher = Sha256::new();
    hasher.update(b"livefire.rag.query-vector-order/1\0");
    for row in rows {
        hasher.update(row.ordinal.to_le_bytes());
        hasher.update(row.query_id.as_bytes());
        hasher.update([0]);
        hasher.update(row.raw_query_sha256.as_str().as_bytes());
        hasher.update(row.composed_query_sha256.as_str().as_bytes());
        hasher.update(row.vector_sha256.as_str().as_bytes());
    }
    Digest::new(format!("{:x}", hasher.finalize())).expect("SHA-256 is valid")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RunpodAcceleratorIdentity;
    use tempfile::tempdir;

    fn digest(byte: char) -> Digest {
        Digest::new(byte.to_string().repeat(64)).unwrap()
    }

    fn component(id: &str, byte: char) -> ComponentRef {
        ComponentRef {
            id: id.into(),
            version: "1".into(),
            sha256: digest(byte),
        }
    }

    fn binding() -> QueryVectorExecutionBinding {
        let profile = component("profile", '1');
        let build = component("build", '2');
        let execution = RunpodExecutionIdentity {
            executor_image: component("executor", '3'),
            executor_image_build: build.clone(),
            runtime: component("runtime", '4'),
            worker_binary: component("worker", '5'),
            model_artifact: component("model", '6'),
            embedding_profile: profile.clone(),
            accelerator: RunpodAcceleratorIdentity {
                provider: "runpod".into(),
                model: "A100".into(),
                architecture: "ampere".into(),
                compute_capability: "8.0".into(),
                count: 1,
            },
            returned_model: "qwen".into(),
        };
        QueryVectorExecutionBinding {
            embedding_profile: profile,
            embedding_policy: component("policy", '7'),
            execution_identity_sha256: canonical_digest(&execution).unwrap(),
            execution,
            executor_image_build_receipt: build,
        }
    }

    fn write_plan(path: &Path) {
        fs::write(
            path,
            concat!(
                "{\"query_id\":\"q1\",\"query\":\"one\",\"mode\":\"dense\",\"top_n\":5,\"relations\":[\"events\",\"process\"]}\n",
                "{\"query_id\":\"q1\",\"query\":\"one\",\"mode\":\"fused\",\"top_n\":5,\"relations\":[\"events\",\"process\"]}\n",
                "{\"query_id\":\"q2\",\"query\":\"two\",\"mode\":\"lexical\",\"top_n\":5,\"relations\":[]}\n",
            ),
        )
        .unwrap();
    }

    #[test]
    fn sealed_set_round_trips_and_requires_exact_query_surfaces() {
        let temp = tempdir().unwrap();
        let plan = temp.path().join("source.jsonl");
        write_plan(&plan);
        let out = temp.path().join("set");
        let binding = binding();
        let vector = [1.0_f32, 0.0];
        let manifest = write_query_vector_set(
            &out,
            &plan,
            binding.clone(),
            2,
            "l2",
            &[QueryVectorSetInput {
                query_id: "q1",
                raw_query: "one",
                composed_query: "Instruct: retrieve\nQuery: one",
                vector: &vector,
            }],
        )
        .unwrap();
        assert_eq!(manifest.request_rows, 3);
        assert_eq!(manifest.semantic_request_rows, 2);
        let opened = SealedQueryVectorSet::open(
            &out,
            &binding.embedding_profile,
            "qwen",
            2,
            "l2",
            Some(&plan),
        )
        .unwrap();
        assert_eq!(
            opened
                .vector_for("q1", "one", "Instruct: retrieve\nQuery: one")
                .unwrap(),
            vector
        );
        assert_eq!(
            opened
                .vector_for_request(
                    "q1",
                    "one",
                    "Instruct: retrieve\nQuery: one",
                    "dense",
                    5,
                    &["events".into(), "process".into()],
                )
                .unwrap(),
            vector
        );
        assert_eq!(
            opened
                .vector_for_unique_request(
                    "one",
                    "Instruct: retrieve\nQuery: one",
                    "dense",
                    5,
                    &["events".into(), "process".into()],
                )
                .unwrap(),
            vector
        );
        assert!(
            opened
                .vector_for_request(
                    "q1",
                    "one",
                    "Instruct: retrieve\nQuery: one",
                    "dense",
                    6,
                    &["events".into(), "process".into()],
                )
                .is_err()
        );
        assert!(
            opened
                .vector_for_unique_request(
                    "one",
                    "Instruct: retrieve\nQuery: one",
                    "dense",
                    5,
                    &["process".into(), "events".into()],
                )
                .is_err()
        );
        assert!(opened.vector_for("q1", "changed", "anything").is_err());
        assert!(opened.vector_for("unknown", "one", "anything").is_err());
        assert!(
            opened
                .vector_for_unique_request(
                    "unknown",
                    "Instruct: retrieve\nQuery: unknown",
                    "dense",
                    5,
                    &[],
                )
                .is_err()
        );
    }

    #[test]
    fn provider_lookup_refuses_an_ambiguous_raw_request() {
        let temp = tempdir().unwrap();
        let plan = temp.path().join("source.jsonl");
        fs::write(
            &plan,
            concat!(
                "{\"query_id\":\"q1\",\"query\":\"same\",\"mode\":\"dense\",\"top_n\":5,\"relations\":[]}\n",
                "{\"query_id\":\"q2\",\"query\":\"same\",\"mode\":\"dense\",\"top_n\":5,\"relations\":[]}\n",
            ),
        )
        .unwrap();
        let out = temp.path().join("set");
        let vector = [1.0_f32, 0.0];
        write_query_vector_set(
            &out,
            &plan,
            binding(),
            2,
            "l2",
            &[
                QueryVectorSetInput {
                    query_id: "q1",
                    raw_query: "same",
                    composed_query: "same",
                    vector: &vector,
                },
                QueryVectorSetInput {
                    query_id: "q2",
                    raw_query: "same",
                    composed_query: "same",
                    vector: &vector,
                },
            ],
        )
        .unwrap();
        let binding = binding();
        let opened =
            SealedQueryVectorSet::open(&out, &binding.embedding_profile, "qwen", 2, "l2", None)
                .unwrap();
        assert!(
            opened
                .vector_for_unique_request("same", "same", "dense", 5, &[])
                .is_err()
        );
    }

    #[test]
    fn tampered_plan_vector_and_profile_are_rejected() {
        let temp = tempdir().unwrap();
        let plan = temp.path().join("source.jsonl");
        write_plan(&plan);
        let binding = binding();
        let vector = [1.0_f32, 0.0];

        for case in ["plan", "vector"] {
            let out = temp.path().join(case);
            write_query_vector_set(
                &out,
                &plan,
                binding.clone(),
                2,
                "l2",
                &[QueryVectorSetInput {
                    query_id: "q1",
                    raw_query: "one",
                    composed_query: "one",
                    vector: &vector,
                }],
            )
            .unwrap();
            if case == "plan" {
                fs::write(out.join(QUERY_VECTOR_SET_PLAN), b"{}\n").unwrap();
            } else {
                fs::write(out.join(QUERY_VECTOR_SET_VECTORS), 0_f32.to_le_bytes()).unwrap();
            }
            assert!(
                SealedQueryVectorSet::open(
                    &out,
                    &binding.embedding_profile,
                    "qwen",
                    2,
                    "l2",
                    None,
                )
                .is_err()
            );
        }

        let out = temp.path().join("profile");
        write_query_vector_set(
            &out,
            &plan,
            binding.clone(),
            2,
            "l2",
            &[QueryVectorSetInput {
                query_id: "q1",
                raw_query: "one",
                composed_query: "one",
                vector: &vector,
            }],
        )
        .unwrap();
        assert!(
            SealedQueryVectorSet::open(&out, &component("other", '8'), "qwen", 2, "l2", None,)
                .is_err()
        );
    }

    #[test]
    fn writer_rejects_bad_order_norm_and_execution_binding() {
        let temp = tempdir().unwrap();
        let plan = temp.path().join("source.jsonl");
        write_plan(&plan);
        let vector = [0.5_f32, 0.0];
        assert!(
            write_query_vector_set(
                &temp.path().join("bad-norm"),
                &plan,
                binding(),
                2,
                "l2",
                &[QueryVectorSetInput {
                    query_id: "q1",
                    raw_query: "one",
                    composed_query: "one",
                    vector: &vector,
                }],
            )
            .is_err()
        );
        let mut wrong = binding();
        wrong.execution.returned_model = "changed".into();
        assert!(wrong.validate().is_err());
    }

    #[test]
    fn schema_is_closed_json_and_names_the_contract() {
        let schema: Value =
            serde_json::from_str(include_str!("../schema/query-vector-set.v1.schema.json"))
                .unwrap();
        assert_eq!(
            schema["$id"],
            "https://livefire.dev/rag/query-vector-set.v1.schema.json"
        );
        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(
            schema["properties"]["schema_version"]["const"],
            QUERY_VECTOR_SET_SCHEMA
        );
    }
}

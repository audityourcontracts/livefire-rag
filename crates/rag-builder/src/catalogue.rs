//! Build, admit, and search a set of independently assembled dataset indexes.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use rag_embedding::{
    EmbeddingProfile, IdentifiedEmbedder, LmStudioEmbedder, adapt_model_vector, try_compose_query,
};
use rag_index::{
    FastIndex, PipelineProvenance, ProfileBoundQueryVector, SearchFilters, SearchHit, SearchMode,
};
use rag_pipeline::{
    AtomicDirectory, CatalogueArtifactRef, CatalogueDatasetEntry, CatalogueMode, ComponentRef,
    DATASET_CATALOGUE_SCHEMA, DatasetCatalogue, DatasetIdentity, Digest, RelationOverlapAllowance,
    SafeRelativePath, canonical_digest, component_digest, digest_bytes, read_json,
    resolve_existing_artifact, validate_dataset_pipeline_binding, write_canonical_json,
};
use rayon::ThreadPool;
use rayon::prelude::*;
use serde::Serialize;
use sha2::{Digest as ShaDigest, Sha256};

use super::{BatchQueryRequest, Error, Mode, Result, portable, read_batch_query_plan};

const MAX_CATALOGUE_WORKERS: usize = 64;
const RANK_MERGE_K: usize = 60;
const BATCH_REQUEST_FILE: &str = "requests.jsonl";
const BATCH_RESULT_FILE: &str = "results.jsonl";
const BATCH_MANIFEST_FILE: &str = "manifest.json";

pub(crate) struct CatalogueBuildOptions {
    pub dataset_paths: Vec<PathBuf>,
    pub overlap_allowances: Vec<String>,
    pub test_only: bool,
    pub out: PathBuf,
}

pub(crate) struct CatalogueSearchOptions<'a> {
    pub catalogue: &'a Path,
    pub query: &'a str,
    pub mode: Mode,
    pub top_n: usize,
    pub endpoint: &'a str,
    pub relations: Vec<String>,
    pub workers: usize,
    pub allow_test_only: bool,
}

pub(crate) struct CatalogueBatchSearchOptions<'a> {
    pub catalogue: &'a Path,
    pub requests: &'a Path,
    pub out: &'a Path,
    pub endpoint: &'a str,
    pub workers: usize,
    pub allow_test_only: bool,
}

struct DatasetPaths {
    prepared: PathBuf,
    plan: PathBuf,
    results: PathBuf,
    index: PathBuf,
}

struct LoadedDataset {
    prepared: rag_pipeline::PreparedCorpusManifest,
    plan: rag_pipeline::EmbeddingPlanV2,
    result_set: rag_pipeline::EmbeddingResultSetManifest,
    index: FastIndex,
    paths: DatasetPaths,
}

struct AdmittedCatalogue {
    catalogue: DatasetCatalogue,
    datasets: Vec<AdmittedDataset>,
}

struct AdmittedDataset {
    entry: CatalogueDatasetEntry,
    index: FastIndex,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct CatalogueSearchHit {
    rank: usize,
    reciprocal_rank_score: f64,
    dataset: DatasetIdentity,
    dataset_sha256: Digest,
    index_sha256: Digest,
    index_rank: usize,
    hit: SearchHit,
}

#[derive(Serialize)]
struct CatalogueSearchOutput<'a> {
    schema_version: &'static str,
    catalogue_sha256: &'a Digest,
    query: &'a str,
    rank_merge: &'a str,
    hits: &'a [CatalogueSearchHit],
}

#[derive(Serialize)]
struct CatalogueBatchSearchOutput<'a> {
    schema_version: &'static str,
    query_id: &'a str,
    catalogue_sha256: &'a Digest,
    query: &'a str,
    mode: Mode,
    top_n: usize,
    relations: &'a [String],
    rank_merge: &'a str,
    hits: &'a [CatalogueSearchHit],
}

#[derive(Debug, Clone, Serialize)]
struct BatchFileReceipt {
    path: &'static str,
    sha256: Digest,
    bytes: u64,
    rows: usize,
}

#[derive(Debug, Clone, Serialize)]
struct BatchModelReceipt {
    status: &'static str,
    configured_model: String,
    returned_model: Option<String>,
    calls: usize,
}

#[derive(Debug, Clone, Serialize)]
struct QueryVectorReceipt {
    composed_query_sha256: Digest,
    vector_sha256: Digest,
    dimensions: usize,
}

#[derive(Debug, Clone, Serialize)]
struct RankMergeReceipt {
    policy: String,
    k: usize,
}

#[derive(Debug, Clone, Serialize)]
struct RequestShapeReceipt {
    mode: Mode,
    top_n: usize,
    relations: Vec<String>,
    rows: usize,
}

#[derive(Debug, Clone, Serialize)]
struct CatalogueBatchSearchManifest {
    schema_version: &'static str,
    component_sha256: Digest,
    status: &'static str,
    catalogue_sha256: Digest,
    embedding_profile: ComponentRef,
    requests: BatchFileReceipt,
    results: BatchFileReceipt,
    request_count: usize,
    result_count: usize,
    modes: Vec<Mode>,
    top_n_values: Vec<usize>,
    relation_filters: Vec<Vec<String>>,
    request_shapes: Vec<RequestShapeReceipt>,
    model: BatchModelReceipt,
    query_vectors: Vec<QueryVectorReceipt>,
    rank_merge: RankMergeReceipt,
}

#[derive(Debug)]
struct CachedQueryVector {
    vector: ProfileBoundQueryVector,
    receipt: QueryVectorReceipt,
}

struct BatchRunExecutionOptions<'a> {
    requests_path: &'a Path,
    expected_request_sha256: &'a Digest,
    expected_request_bytes: u64,
    out: &'a Path,
    workers: usize,
    allow_test_only: bool,
}

pub(crate) fn build_catalogue(options: CatalogueBuildOptions) -> Result<()> {
    if options.dataset_paths.is_empty() || !options.dataset_paths.len().is_multiple_of(4) {
        return Err(Error::AccountingClosure(
            "each catalogue dataset needs prepared, plan, results, and index paths",
        ));
    }
    let root = catalogue_output_root(&options.out)?;
    let mut loaded = options
        .dataset_paths
        .chunks_exact(4)
        .map(|paths| {
            load_dataset(
                DatasetPaths {
                    prepared: manifest_input(&paths[0], "manifest.json")?,
                    plan: manifest_input(&paths[1], "plan.json")?,
                    results: manifest_input(&paths[2], "manifest.json")?,
                    index: manifest_input(&paths[3], "index.json")?,
                },
                options.test_only,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    loaded.sort_by(|left, right| left.prepared.dataset.id.cmp(&right.prepared.dataset.id));
    require_one_exact_profile(&loaded)?;

    let datasets = loaded
        .iter()
        .map(|dataset| catalogue_entry(&root, dataset, options.test_only))
        .collect::<Result<Vec<_>>>()?;
    let first = datasets
        .first()
        .ok_or(Error::AccountingClosure("dataset catalogue is empty"))?;
    let allowed_relation_overlaps = overlap_allowances(&datasets, &options.overlap_allowances)?;
    let mut catalogue = DatasetCatalogue {
        schema_version: DATASET_CATALOGUE_SCHEMA.into(),
        component_sha256: Digest::new("0".repeat(64))?,
        mode: if options.test_only {
            CatalogueMode::TestOnly
        } else {
            CatalogueMode::Normal
        },
        source_snapshot: first.dataset.source_snapshot.clone(),
        mapping: first.dataset.mapping.clone(),
        projection_policy: first.projection_policy.clone(),
        embedding_profile: first.embedding_profile.clone(),
        query_compatibility: "single_embedding_profile".into(),
        rank_merge: "reciprocal_rank_fusion_v1".into(),
        datasets,
        allowed_relation_overlaps,
    };
    catalogue.seal()?;
    write_canonical_json(&options.out, &catalogue)?;
    // Re-open from the published paths so build and later validation use the
    // same physical admission checks.
    admit_catalogue(&options.out)?;
    println!("{}", serde_json::to_string_pretty(&catalogue)?);
    Ok(())
}

pub(crate) fn validate_catalogue(path: &Path) -> Result<()> {
    let admitted = admit_catalogue(path)?;
    println!("{}", serde_json::to_string_pretty(&admitted.catalogue)?);
    Ok(())
}

pub(crate) async fn search_catalogue(options: CatalogueSearchOptions<'_>) -> Result<()> {
    let admitted = admit_catalogue(options.catalogue)?;
    require_search_permission(&admitted, options.allow_test_only)?;
    let embedder = LmStudioEmbedder::new(
        options.endpoint,
        &admitted
            .datasets
            .first()
            .ok_or(Error::AccountingClosure("dataset catalogue is empty"))?
            .index
            .manifest
            .embedding_profile
            .model,
    );
    let hits = search_admitted_catalogue(&admitted, &embedder, &options).await?;
    println!(
        "{}",
        serde_json::to_string_pretty(&CatalogueSearchOutput {
            schema_version: "livefire.rag.catalogue-search-result/1",
            catalogue_sha256: &admitted.catalogue.component_sha256,
            query: options.query,
            rank_merge: &admitted.catalogue.rank_merge,
            hits: &hits,
        })?
    );
    Ok(())
}

pub(crate) async fn batch_search_catalogue(options: CatalogueBatchSearchOptions<'_>) -> Result<()> {
    let plan = read_batch_query_plan(options.requests)?;
    let admitted = admit_catalogue(options.catalogue)?;
    let embedder = LmStudioEmbedder::new(
        options.endpoint,
        &admitted
            .datasets
            .first()
            .ok_or(Error::AccountingClosure("dataset catalogue is empty"))?
            .index
            .manifest
            .embedding_profile
            .model,
    );
    let manifest = publish_batch_search_run(
        &admitted,
        &embedder,
        &plan.requests,
        BatchRunExecutionOptions {
            requests_path: options.requests,
            expected_request_sha256: &plan.source_sha256,
            expected_request_bytes: plan.source_bytes,
            out: options.out,
            workers: options.workers,
            allow_test_only: options.allow_test_only,
        },
    )
    .await?;
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "status":"published",
            "component_sha256":manifest.component_sha256,
            "request_count":manifest.request_count,
            "result_count":manifest.result_count,
            "model_calls":manifest.model.calls
        }))?
    );
    Ok(())
}

async fn publish_batch_search_run<E: IdentifiedEmbedder>(
    admitted: &AdmittedCatalogue,
    embedder: &E,
    requests: &[BatchQueryRequest],
    options: BatchRunExecutionOptions<'_>,
) -> Result<CatalogueBatchSearchManifest> {
    require_search_permission(admitted, options.allow_test_only)?;
    validate_catalogue_batch_requests(admitted, requests)?;
    let source_requests_receipt =
        batch_file_receipt(options.requests_path, BATCH_REQUEST_FILE, requests.len())?;
    if &source_requests_receipt.sha256 != options.expected_request_sha256
        || source_requests_receipt.bytes != options.expected_request_bytes
    {
        return Err(Error::AccountingClosure(
            "catalogue batch requests changed after validation",
        ));
    }
    let pool = catalogue_search_pool(options.workers)?;
    let destination = if options
        .out
        .parent()
        .is_none_or(|parent| parent.as_os_str().is_empty())
    {
        Path::new(".").join(options.out)
    } else {
        options.out.to_owned()
    };
    let staging = AtomicDirectory::new(&destination)?;
    fs::copy(
        options.requests_path,
        staging.path().join(BATCH_REQUEST_FILE),
    )?;
    let result_path = staging.path().join(BATCH_RESULT_FILE);
    let mut writer = BufWriter::new(File::create(&result_path)?);
    let first = admitted
        .datasets
        .first()
        .ok_or(Error::AccountingClosure("dataset catalogue is empty"))?;
    let profile = &first.index.manifest.embedding_profile;
    let mut cache = BTreeMap::<String, CachedQueryVector>::new();
    let mut returned_model: Option<String> = None;
    let mut result_count = 0_usize;
    for request in requests {
        let options = CatalogueSearchOptions {
            catalogue: Path::new("admitted-once"),
            query: &request.query,
            mode: request.mode,
            top_n: request.top_n,
            endpoint: "unused-with-supplied-embedder",
            relations: request.relations.clone(),
            workers: options.workers,
            allow_test_only: options.allow_test_only,
        };
        validate_catalogue_search_options(&options)?;
        let search_mode: SearchMode = request.mode.into();
        let query_vector = if matches!(search_mode, SearchMode::Dense | SearchMode::Fused) {
            let composed = try_compose_query(profile, &request.query)?;
            if !cache.contains_key(&composed) {
                let response = embedder
                    .embed_identified(std::slice::from_ref(&composed))
                    .await?;
                if response.returned_model != profile.model {
                    return Err(Error::AccountingClosure(
                        "catalogue query embedding response used a different model",
                    ));
                }
                if returned_model
                    .as_ref()
                    .is_some_and(|model| model != &response.returned_model)
                {
                    return Err(Error::AccountingClosure(
                        "catalogue query embedding response used a different model",
                    ));
                }
                returned_model = Some(response.returned_model);
                let mut vectors = response.vectors;
                if vectors.len() != 1 {
                    return Err(Error::AccountingClosure(
                        "catalogue query embedding response has the wrong size",
                    ));
                }
                let vector = adapt_model_vector(profile, vectors.remove(0))?;
                let dimensions = vector.len();
                let vector_sha256 = query_vector_digest(&vector)?;
                let vector = first.index.validate_query_vector(&vector)?;
                cache.insert(
                    composed.clone(),
                    CachedQueryVector {
                        receipt: QueryVectorReceipt {
                            composed_query_sha256: digest_bytes(composed.as_bytes()),
                            vector_sha256,
                            dimensions,
                        },
                        vector,
                    },
                );
            }
            Some(
                &cache
                    .get(&composed)
                    .ok_or(Error::AccountingClosure("query vector cache is incomplete"))?
                    .vector,
            )
        } else {
            None
        };
        let hits = search_admitted_catalogue_with_vector(admitted, &options, &pool, query_vector)?;
        serde_json::to_writer(
            &mut writer,
            &CatalogueBatchSearchOutput {
                schema_version: "livefire.rag.catalogue-batch-search-result/1",
                query_id: &request.query_id,
                catalogue_sha256: &admitted.catalogue.component_sha256,
                query: &request.query,
                mode: request.mode,
                top_n: request.top_n,
                relations: &request.relations,
                rank_merge: &admitted.catalogue.rank_merge,
                hits: &hits,
            },
        )?;
        writer.write_all(b"\n")?;
        result_count = result_count
            .checked_add(1)
            .ok_or(Error::AccountingClosure("batch result count overflow"))?;
    }
    writer.flush()?;
    drop(writer);

    let requests_receipt = batch_file_receipt(
        &staging.path().join(BATCH_REQUEST_FILE),
        BATCH_REQUEST_FILE,
        requests.len(),
    )?;
    if requests_receipt.sha256 != source_requests_receipt.sha256
        || requests_receipt.bytes != source_requests_receipt.bytes
        || requests_receipt.rows != source_requests_receipt.rows
    {
        return Err(Error::AccountingClosure(
            "copied catalogue batch requests differ from the source",
        ));
    }
    let results_receipt = batch_file_receipt(&result_path, BATCH_RESULT_FILE, result_count)?;
    let modes = requests
        .iter()
        .map(|request| request.mode)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let top_n_values = requests
        .iter()
        .map(|request| request.top_n)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let relation_filters = requests
        .iter()
        .map(|request| request.relations.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let mut shape_counts = BTreeMap::<(Mode, usize, Vec<String>), usize>::new();
    for request in requests {
        *shape_counts
            .entry((request.mode, request.top_n, request.relations.clone()))
            .or_default() += 1;
    }
    let request_shapes = shape_counts
        .into_iter()
        .map(|((mode, top_n, relations), rows)| RequestShapeReceipt {
            mode,
            top_n,
            relations,
            rows,
        })
        .collect();
    let mut query_vectors = cache
        .into_values()
        .map(|cached| cached.receipt)
        .collect::<Vec<_>>();
    query_vectors
        .sort_by(|left, right| left.composed_query_sha256.cmp(&right.composed_query_sha256));
    let model_calls = query_vectors.len();
    let mut manifest = CatalogueBatchSearchManifest {
        schema_version: "livefire.rag.catalogue-batch-search-run/1",
        component_sha256: Digest::new("0".repeat(64))?,
        status: "complete",
        catalogue_sha256: admitted.catalogue.component_sha256.clone(),
        embedding_profile: admitted.catalogue.embedding_profile.clone(),
        requests: requests_receipt,
        results: results_receipt,
        request_count: requests.len(),
        result_count,
        modes,
        top_n_values,
        relation_filters,
        request_shapes,
        model: BatchModelReceipt {
            status: if model_calls == 0 {
                "not_used_all_lexical"
            } else {
                "used"
            },
            configured_model: profile.model.clone(),
            returned_model,
            calls: model_calls,
        },
        query_vectors,
        rank_merge: RankMergeReceipt {
            policy: admitted.catalogue.rank_merge.clone(),
            k: RANK_MERGE_K,
        },
    };
    validate_batch_manifest(&manifest)?;
    manifest.component_sha256 = component_digest(&manifest)?;
    validate_batch_manifest(&manifest)?;
    write_canonical_json(&staging.path().join(BATCH_MANIFEST_FILE), &manifest)?;
    staging.publish()?;
    Ok(manifest)
}

fn validate_catalogue_batch_requests(
    admitted: &AdmittedCatalogue,
    requests: &[BatchQueryRequest],
) -> Result<()> {
    if requests.is_empty() || requests.len() > super::MAX_BATCH_QUERY_REQUESTS {
        return Err(Error::InvalidBatchQuery);
    }
    let available_relations = admitted
        .catalogue
        .datasets
        .iter()
        .flat_map(|dataset| dataset.dataset.included_relations.iter().cloned())
        .collect::<BTreeSet<_>>();
    let mut surfaces = BTreeSet::new();
    let mut query_id_contract = BTreeMap::<&str, (&str, usize, &[String])>::new();
    for request in requests {
        request.validate()?;
        if !surfaces.insert((request.mode, request.query_id.as_str()))
            || request
                .relations
                .iter()
                .any(|relation| !available_relations.contains(relation))
        {
            return Err(Error::InvalidBatchQuery);
        }
        if let Some((query, top_n, relations)) = query_id_contract.get(request.query_id.as_str()) {
            if *query != request.query
                || *top_n != request.top_n
                || *relations != request.relations.as_slice()
            {
                return Err(Error::InvalidBatchQuery);
            }
        } else {
            query_id_contract.insert(
                &request.query_id,
                (&request.query, request.top_n, &request.relations),
            );
        }
    }
    Ok(())
}

fn batch_file_receipt(path: &Path, name: &'static str, rows: usize) -> Result<BatchFileReceipt> {
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
    Ok(BatchFileReceipt {
        path: name,
        sha256: Digest::new(format!("{:x}", hasher.finalize()))?,
        bytes: fs::metadata(path)?.len(),
        rows,
    })
}

fn query_vector_digest(vector: &[f32]) -> Result<Digest> {
    let mut hasher = Sha256::new();
    for value in vector {
        hasher.update(value.to_le_bytes());
    }
    Ok(Digest::new(format!("{:x}", hasher.finalize()))?)
}

fn validate_batch_manifest(manifest: &CatalogueBatchSearchManifest) -> Result<()> {
    let zero = Digest::new("0".repeat(64))?;
    let sorted_unique = |values: &[String]| {
        values
            .windows(2)
            .all(|pair| pair[0].as_str() < pair[1].as_str())
    };
    if manifest.status != "complete"
        || manifest.request_count == 0
        || manifest.request_count != manifest.requests.rows
        || manifest.result_count != manifest.results.rows
        || manifest.request_count != manifest.result_count
        || manifest.requests.path != BATCH_REQUEST_FILE
        || manifest.results.path != BATCH_RESULT_FILE
        || manifest.requests.bytes == 0
        || manifest.results.bytes == 0
        || manifest.modes.is_empty()
        || !manifest.modes.windows(2).all(|pair| pair[0] < pair[1])
        || manifest.top_n_values.is_empty()
        || !manifest
            .top_n_values
            .windows(2)
            .all(|pair| pair[0] < pair[1])
        || !manifest
            .relation_filters
            .windows(2)
            .all(|pair| pair[0] < pair[1])
        || manifest
            .relation_filters
            .iter()
            .any(|relations| !sorted_unique(relations))
        || manifest.request_shapes.is_empty()
        || manifest
            .request_shapes
            .iter()
            .map(|shape| shape.rows)
            .sum::<usize>()
            != manifest.request_count
        || !manifest.request_shapes.windows(2).all(|pair| {
            (&pair[0].mode, pair[0].top_n, &pair[0].relations)
                < (&pair[1].mode, pair[1].top_n, &pair[1].relations)
        })
        || manifest
            .request_shapes
            .iter()
            .any(|shape| shape.rows == 0 || !sorted_unique(&shape.relations))
        || manifest.model.calls != manifest.query_vectors.len()
        || !manifest
            .query_vectors
            .windows(2)
            .all(|pair| pair[0].composed_query_sha256 < pair[1].composed_query_sha256)
        || manifest.rank_merge.policy != "reciprocal_rank_fusion_v1"
        || manifest.rank_merge.k != RANK_MERGE_K
    {
        return Err(Error::AccountingClosure(
            "invalid catalogue batch run manifest",
        ));
    }
    let has_semantic_mode = manifest
        .modes
        .iter()
        .any(|mode| matches!(mode, Mode::Dense | Mode::Fused));
    if has_semantic_mode
        != (manifest.model.status == "used"
            && manifest.model.calls > 0
            && manifest.model.returned_model.as_deref()
                == Some(manifest.model.configured_model.as_str()))
        || (!has_semantic_mode
            && (manifest.model.status != "not_used_all_lexical"
                || manifest.model.calls != 0
                || manifest.model.returned_model.is_some()
                || !manifest.query_vectors.is_empty()))
        || (manifest.component_sha256 != zero
            && manifest.component_sha256 != component_digest(manifest)?)
    {
        return Err(Error::AccountingClosure(
            "invalid catalogue batch run manifest",
        ));
    }
    Ok(())
}

fn require_search_permission(admitted: &AdmittedCatalogue, allow_test_only: bool) -> Result<()> {
    if matches!(admitted.catalogue.mode, CatalogueMode::TestOnly) && !allow_test_only {
        return Err(Error::AccountingClosure(
            "test-only catalogue search needs --allow-test-only",
        ));
    }
    Ok(())
}

async fn search_admitted_catalogue<E: IdentifiedEmbedder>(
    admitted: &AdmittedCatalogue,
    embedder: &E,
    options: &CatalogueSearchOptions<'_>,
) -> Result<Vec<CatalogueSearchHit>> {
    validate_catalogue_search_options(options)?;
    let pool = catalogue_search_pool(options.workers)?;
    search_admitted_catalogue_in_pool(admitted, embedder, options, &pool).await
}

fn validate_catalogue_search_options(options: &CatalogueSearchOptions<'_>) -> Result<()> {
    if options.query.is_empty()
        || options.top_n == 0
        || options.top_n > 100
        || options.workers == 0
        || options.workers > MAX_CATALOGUE_WORKERS
        || options.relations.iter().any(String::is_empty)
    {
        return Err(Error::AccountingClosure("invalid catalogue search options"));
    }
    Ok(())
}

fn catalogue_search_pool(workers: usize) -> Result<ThreadPool> {
    if workers == 0 || workers > MAX_CATALOGUE_WORKERS {
        return Err(Error::AccountingClosure("invalid catalogue search options"));
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .thread_name(|ordinal| format!("rag-catalogue-search-{ordinal}"))
        .build()
        .map_err(|_| Error::AccountingClosure("catalogue search worker pool could not start"))
}

async fn search_admitted_catalogue_in_pool<E: IdentifiedEmbedder>(
    admitted: &AdmittedCatalogue,
    embedder: &E,
    options: &CatalogueSearchOptions<'_>,
    pool: &ThreadPool,
) -> Result<Vec<CatalogueSearchHit>> {
    let first = admitted
        .datasets
        .first()
        .ok_or(Error::AccountingClosure("dataset catalogue is empty"))?;
    let search_mode: SearchMode = options.mode.into();
    let query_vector = if matches!(search_mode, SearchMode::Dense | SearchMode::Fused) {
        let composed = try_compose_query(&first.index.manifest.embedding_profile, options.query)?;
        let response = embedder.embed_identified(&[composed]).await?;
        if response.returned_model != first.index.manifest.embedding_profile.model {
            return Err(Error::AccountingClosure(
                "catalogue query embedding response used a different model",
            ));
        }
        let mut vectors = response.vectors;
        if vectors.len() != 1 {
            return Err(Error::AccountingClosure(
                "catalogue query embedding response has the wrong size",
            ));
        }
        let vector =
            adapt_model_vector(&first.index.manifest.embedding_profile, vectors.remove(0))?;
        Some(first.index.validate_query_vector(&vector)?)
    } else {
        None
    };
    search_admitted_catalogue_with_vector(admitted, options, pool, query_vector.as_ref())
}

fn search_admitted_catalogue_with_vector(
    admitted: &AdmittedCatalogue,
    options: &CatalogueSearchOptions<'_>,
    pool: &ThreadPool,
    query_vector: Option<&ProfileBoundQueryVector>,
) -> Result<Vec<CatalogueSearchHit>> {
    let search_mode: SearchMode = options.mode.into();
    if matches!(search_mode, SearchMode::Dense | SearchMode::Fused) != query_vector.is_some() {
        return Err(Error::AccountingClosure(
            "catalogue query vector does not match the search mode",
        ));
    }
    let filters = SearchFilters {
        relations: options.relations.iter().cloned().collect(),
        ..Default::default()
    };
    let searched = pool.install(|| {
        admitted
            .datasets
            .par_iter()
            .map(|dataset| {
                let hits = match (search_mode, query_vector) {
                    (SearchMode::Dense, Some(query)) => {
                        dataset
                            .index
                            .search_dense_with_vector(query, &filters, options.top_n)
                    }
                    (SearchMode::Fused, Some(query)) => dataset.index.search_fused_with_vector(
                        options.query,
                        query,
                        &filters,
                        options.top_n,
                    ),
                    (SearchMode::Lexical, None) => dataset.index.search(
                        SearchMode::Lexical,
                        options.query,
                        None,
                        &filters,
                        options.top_n,
                    ),
                    _ => unreachable!("query vector is fixed by search mode"),
                }?;
                Ok((dataset, hits))
            })
            .collect::<Vec<Result<_>>>()
    });
    let mut ranked = Vec::new();
    for searched_dataset in searched {
        let (dataset, hits) = searched_dataset?;
        for hit in hits {
            let index_rank = hit.rank;
            ranked.push(CatalogueSearchHit {
                rank: 0,
                reciprocal_rank_score: reciprocal_rank_score(index_rank)?,
                dataset: dataset.entry.dataset.clone(),
                dataset_sha256: dataset.entry.dataset_sha256.clone(),
                index_sha256: dataset.entry.final_index.component.sha256.clone(),
                index_rank,
                hit,
            });
        }
    }
    rank_catalogue_hits(&mut ranked, options.top_n);
    Ok(ranked)
}

fn rank_catalogue_hits(hits: &mut Vec<CatalogueSearchHit>, top_n: usize) {
    hits.sort_by(|left, right| {
        right
            .reciprocal_rank_score
            .total_cmp(&left.reciprocal_rank_score)
            .then_with(|| left.dataset_sha256.cmp(&right.dataset_sha256))
            .then_with(|| left.hit.document_id.cmp(&right.hit.document_id))
    });
    hits.truncate(top_n);
    for (ordinal, hit) in hits.iter_mut().enumerate() {
        hit.rank = ordinal + 1;
    }
}

fn reciprocal_rank_score(rank: usize) -> Result<f64> {
    if rank == 0 {
        return Err(Error::AccountingClosure(
            "catalogue index returned a zero rank",
        ));
    }
    Ok(1.0 / (RANK_MERGE_K + rank) as f64)
}

fn admit_catalogue(path: &Path) -> Result<AdmittedCatalogue> {
    let path = fs::canonicalize(path)?;
    let root = path
        .parent()
        .ok_or(Error::AccountingClosure("catalogue root is absent"))?;
    let catalogue: DatasetCatalogue = read_json(&path)?;
    catalogue.validate()?;
    let mut datasets = Vec::with_capacity(catalogue.datasets.len());
    let mut expected_profile: Option<EmbeddingProfile> = None;
    for entry in &catalogue.datasets {
        let loaded = load_dataset(
            DatasetPaths {
                prepared: resolve_existing_artifact(root, &entry.prepared_corpus.path)?,
                plan: resolve_existing_artifact(root, &entry.embedding_plan.path)?,
                results: resolve_existing_artifact(root, &entry.embedding_result_set.path)?,
                index: resolve_existing_artifact(root, &entry.final_index.path)?,
            },
            matches!(catalogue.mode, CatalogueMode::TestOnly),
        )?;
        validate_loaded_entry(entry, &loaded)?;
        if expected_profile
            .as_ref()
            .is_some_and(|profile| profile != &loaded.index.manifest.embedding_profile)
        {
            return Err(Error::AccountingClosure(
                "catalogue indexes have different embedding profiles",
            ));
        }
        expected_profile.get_or_insert_with(|| loaded.index.manifest.embedding_profile.clone());
        datasets.push(AdmittedDataset {
            entry: entry.clone(),
            index: loaded.index,
        });
    }
    Ok(AdmittedCatalogue {
        catalogue,
        datasets,
    })
}

fn load_dataset(paths: DatasetPaths, allow_test_only: bool) -> Result<LoadedDataset> {
    require_manifest_name(&paths.prepared, "manifest.json")?;
    require_manifest_name(&paths.plan, "plan.json")?;
    require_manifest_name(&paths.results, "manifest.json")?;
    require_manifest_name(&paths.index, "index.json")?;
    let prepared_root = artifact_parent(&paths.prepared)?;
    let plan_root = artifact_parent(&paths.plan)?;
    let results_root = artifact_parent(&paths.results)?;
    let index_root = artifact_parent(&paths.index)?;
    let prepared = portable::load_prepared(prepared_root)?;
    let plan = portable::load_embedding_plan_v2(plan_root)?;
    let index = if allow_test_only {
        FastIndex::open_allow_test_only(index_root)?
    } else {
        FastIndex::open(index_root)?
    };
    let result_set = portable::load_completed_embedding_result_set(
        results_root,
        prepared_root,
        plan_root,
        &plan,
        &prepared,
        &index.manifest.embedding_profile,
    )?;
    Ok(LoadedDataset {
        prepared,
        plan,
        result_set,
        index,
        paths,
    })
}

fn validate_loaded_entry(entry: &CatalogueDatasetEntry, loaded: &LoadedDataset) -> Result<()> {
    validate_dataset_pipeline_binding(entry, &loaded.prepared, &loaded.plan, &loaded.result_set)?;
    if entry.prepared_corpus.component.sha256 != loaded.prepared.component_sha256
        || entry.embedding_plan.component.sha256 != loaded.plan.component_sha256
        || entry.embedding_result_set.component.sha256 != loaded.result_set.component_sha256
    {
        return Err(Error::AccountingClosure(
            "catalogue component reference differs from pipeline artifacts",
        ));
    }
    let expected_provenance = PipelineProvenance {
        dataset_sha256: entry.dataset_sha256.to_string(),
        prepared_corpus_sha256: loaded.prepared.component_sha256.to_string(),
        embedding_plan_sha256: loaded.plan.component_sha256.to_string(),
        embedding_result_set_sha256: loaded.result_set.component_sha256.to_string(),
    };
    validate_final_index_binding(
        entry,
        &loaded.plan.embedding_profile,
        &expected_provenance,
        loaded.prepared.document_order_sha256.as_str(),
        &loaded.index,
    )
}

fn validate_final_index_binding(
    entry: &CatalogueDatasetEntry,
    planned_profile: &rag_pipeline::EmbeddingProfileRef,
    expected_provenance: &PipelineProvenance,
    expected_document_order_sha256: &str,
    index: &FastIndex,
) -> Result<()> {
    let profile = &index.manifest.embedding_profile;
    if entry.final_index.component.sha256.as_str() != index.manifest.component_sha256.as_str()
        || entry.searchable_document_count != index.manifest.documents.rows
        || entry.searchable_reference_count != index.manifest.occurrences.rows
        || index.manifest.source.snapshot_sha256 != entry.dataset.source_snapshot.sha256.as_str()
        || index.manifest.source.mapping_sha256 != entry.dataset.mapping.sha256.as_str()
        || profile.id != entry.embedding_profile.id
        || profile.version != entry.embedding_profile.version
        || profile.sha256 != entry.embedding_profile.sha256.as_str()
        || profile.dimensions != planned_profile.dimensions
        || profile.normalization != planned_profile.normalization
        || index.manifest.pipeline_provenance.as_ref() != Some(expected_provenance)
        || index.manifest.vectors.document_order_sha256 != expected_document_order_sha256
        || index.manifest.test_only != entry.test_only
    {
        return Err(Error::AccountingClosure(
            "catalogue final index binding differs from pipeline artifacts",
        ));
    }
    Ok(())
}

fn catalogue_entry(
    root: &Path,
    loaded: &LoadedDataset,
    test_only: bool,
) -> Result<CatalogueDatasetEntry> {
    let entry = CatalogueDatasetEntry {
        dataset_sha256: canonical_digest(&loaded.prepared.dataset)?,
        dataset: loaded.prepared.dataset.clone(),
        projection_policy: loaded.prepared.projection_policy.clone(),
        prepared_corpus: artifact_ref(
            root,
            &loaded.paths.prepared,
            "livefire.rag.prepared-corpus",
            &loaded.prepared.schema_version,
            loaded.prepared.component_sha256.clone(),
        )?,
        embedding_plan: artifact_ref(
            root,
            &loaded.paths.plan,
            "livefire.rag.embedding-plan",
            &loaded.plan.schema_version,
            loaded.plan.component_sha256.clone(),
        )?,
        embedding_result_set: artifact_ref(
            root,
            &loaded.paths.results,
            "livefire.rag.embedding-result-set",
            &loaded.result_set.schema_version,
            loaded.result_set.component_sha256.clone(),
        )?,
        embedding_profile: loaded.plan.embedding_profile.component.clone(),
        final_index: artifact_ref(
            root,
            &loaded.paths.index,
            "livefire.rag.fast-index",
            &loaded.index.manifest.schema_version,
            Digest::new(loaded.index.manifest.component_sha256.clone())?,
        )?,
        searchable_document_count: loaded.index.manifest.documents.rows,
        searchable_reference_count: loaded.index.manifest.occurrences.rows,
        test_only,
    };
    validate_loaded_entry(&entry, loaded)?;
    Ok(entry)
}

fn require_one_exact_profile(datasets: &[LoadedDataset]) -> Result<()> {
    let Some(first) = datasets.first() else {
        return Err(Error::AccountingClosure("dataset catalogue is empty"));
    };
    if datasets.iter().skip(1).any(|dataset| {
        dataset.index.manifest.embedding_profile != first.index.manifest.embedding_profile
    }) {
        return Err(Error::AccountingClosure(
            "catalogue indexes have different embedding profiles",
        ));
    }
    Ok(())
}

fn overlap_allowances(
    datasets: &[CatalogueDatasetEntry],
    requested: &[String],
) -> Result<Vec<RelationOverlapAllowance>> {
    let mut reasons = BTreeMap::new();
    for value in requested {
        let (relation, reason) = value
            .split_once('=')
            .filter(|(relation, reason)| !relation.is_empty() && !reason.is_empty())
            .ok_or(Error::AccountingClosure(
                "relation overlap allowance must be RELATION=REASON",
            ))?;
        if reasons
            .insert(relation.to_owned(), reason.to_owned())
            .is_some()
        {
            return Err(Error::AccountingClosure(
                "relation overlap allowance is duplicated",
            ));
        }
    }
    let mut owners = BTreeMap::<String, BTreeSet<String>>::new();
    for dataset in datasets {
        for relation in &dataset.dataset.included_relations {
            owners
                .entry(relation.clone())
                .or_default()
                .insert(dataset.dataset.id.clone());
        }
    }
    reasons
        .into_iter()
        .map(|(relation, reason)| {
            let dataset_ids = owners
                .get(&relation)
                .filter(|owners| owners.len() > 1)
                .ok_or(Error::AccountingClosure(
                    "relation overlap allowance does not name an overlap",
                ))?
                .iter()
                .cloned()
                .collect();
            Ok(RelationOverlapAllowance {
                relation,
                dataset_ids,
                reason,
            })
        })
        .collect()
}

fn artifact_ref(
    root: &Path,
    path: &Path,
    id: &str,
    schema_version: &str,
    sha256: Digest,
) -> Result<CatalogueArtifactRef> {
    let path = fs::canonicalize(path)?;
    let relative = path
        .strip_prefix(root)
        .map_err(|_| Error::AccountingClosure("catalogue artifact is outside its root"))?;
    let relative = relative
        .components()
        .map(|component| component.as_os_str().to_str())
        .collect::<Option<Vec<_>>>()
        .ok_or(Error::AccountingClosure(
            "catalogue artifact path is not UTF-8",
        ))?
        .join("/");
    Ok(CatalogueArtifactRef {
        path: SafeRelativePath::new(relative)?,
        component: ComponentRef {
            id: id.into(),
            version: schema_version
                .rsplit('/')
                .next()
                .unwrap_or(schema_version)
                .into(),
            sha256,
        },
    })
}

fn catalogue_output_root(out: &Path) -> Result<PathBuf> {
    let parent = out
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    Ok(fs::canonicalize(parent)?)
}

fn manifest_input(path: &Path, filename: &str) -> Result<PathBuf> {
    let path = if path.is_dir() {
        path.join(filename)
    } else {
        path.to_owned()
    };
    Ok(fs::canonicalize(path)?)
}

fn artifact_parent(path: &Path) -> Result<&Path> {
    path.parent().ok_or(Error::AccountingClosure(
        "catalogue artifact parent is absent",
    ))
}

fn require_manifest_name(path: &Path, expected: &str) -> Result<()> {
    if path.file_name().and_then(|name| name.to_str()) != Some(expected) {
        return Err(Error::AccountingClosure(
            "catalogue artifact has the wrong manifest name",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use rag_embedding::{Embedder, IdentifiedEmbeddingBatch};
    use rag_index::{
        BuildScope, EvidenceOccurrence, FastDocument, FastOccurrence, SearchHit, SourceBinding,
        write_fast_index,
    };

    fn digest(character: char) -> Digest {
        Digest::new(character.to_string().repeat(64)).unwrap()
    }

    fn dataset(id: &str, _digest_character: char) -> DatasetIdentity {
        let component = |name: &str, character: char| ComponentRef {
            id: name.into(),
            version: "1".into(),
            sha256: digest(character),
        };
        DatasetIdentity {
            id: id.into(),
            version: "1".into(),
            source_snapshot: component("snapshot", 'a'),
            mapping: component("mapping", 'b'),
            included_relations: vec![format!("relation-{id}")],
            excluded_relations: vec![],
            structured_only_relations: vec![],
        }
    }

    fn ranked_hit(
        dataset_id: &str,
        dataset_digest: char,
        document: &str,
        rank: usize,
        score: f64,
    ) -> CatalogueSearchHit {
        CatalogueSearchHit {
            rank: 0,
            reciprocal_rank_score: reciprocal_rank_score(rank).unwrap(),
            dataset: dataset(dataset_id, dataset_digest),
            dataset_sha256: digest(dataset_digest),
            index_sha256: digest('f'),
            index_rank: rank,
            hit: SearchHit {
                rank,
                document_id: document.into(),
                semantic_text: format!("text-{document}"),
                score,
                dense_score: Some(score),
                lexical_score: None,
                eligible_occurrence_count: 1,
                occurrences_exhausted: true,
                occurrences: vec![EvidenceOccurrence {
                    event_time_ms: None,
                    relation: format!("relation-{dataset_id}"),
                    snapshot_sha256: "a".repeat(64),
                    mapping_sha256: "b".repeat(64),
                    event_id: format!("event-{document}"),
                    support_ref: format!("support-{document}"),
                }],
            },
        }
    }

    #[test]
    fn rank_merge_uses_per_index_rank_and_stable_dataset_identity() {
        let mut hits = vec![
            ranked_hit("dataset-b", '2', "b-second", 2, 10_000.0),
            ranked_hit("dataset-a", '1', "a-first", 1, 0.01),
            ranked_hit("dataset-b", '2', "b-first", 1, 0.001),
        ];
        rank_catalogue_hits(&mut hits, 3);
        assert_eq!(
            hits.iter()
                .map(|hit| (
                    hit.rank,
                    hit.dataset.id.as_str(),
                    hit.hit.document_id.as_str()
                ))
                .collect::<Vec<_>>(),
            vec![
                (1, "dataset-a", "a-first"),
                (2, "dataset-b", "b-first"),
                (3, "dataset-b", "b-second"),
            ]
        );
        assert_eq!(hits[2].index_rank, 2);
        assert_eq!(hits[2].hit.score, 10_000.0);
    }

    #[test]
    fn overlap_flags_are_explicit_exact_and_sorted() {
        let entry = |id: &str, relation: &str| {
            let dataset = DatasetIdentity {
                included_relations: vec![relation.into()],
                ..dataset(id, if id == "a" { '1' } else { '2' })
            };
            CatalogueDatasetEntry {
                dataset_sha256: canonical_digest(&dataset).unwrap(),
                dataset,
                projection_policy: ComponentRef {
                    id: "projection".into(),
                    version: "1".into(),
                    sha256: digest('3'),
                },
                prepared_corpus: test_artifact(id, "prepared", '4'),
                embedding_plan: test_artifact(id, "plan", '5'),
                embedding_result_set: test_artifact(id, "results", '6'),
                embedding_profile: ComponentRef {
                    id: "profile".into(),
                    version: "1".into(),
                    sha256: digest('7'),
                },
                final_index: test_artifact(id, "index", '8'),
                searchable_document_count: 1,
                searchable_reference_count: 1,
                test_only: false,
            }
        };
        let entries = vec![entry("a", "shared"), entry("b", "shared")];
        let allowances = overlap_allowances(&entries, &["shared=known duplicate".into()]).unwrap();
        assert_eq!(allowances[0].dataset_ids, ["a", "b"]);
        assert!(overlap_allowances(&entries, &[]).unwrap().is_empty());
        assert!(overlap_allowances(&entries, &["missing=reason".into()]).is_err());
        assert!(overlap_allowances(&entries, &["shared=".into()]).is_err());
    }

    fn test_artifact(id: &str, kind: &str, character: char) -> CatalogueArtifactRef {
        CatalogueArtifactRef {
            path: SafeRelativePath::new(format!("{id}/{kind}.json")).unwrap(),
            component: ComponentRef {
                id: kind.into(),
                version: "1".into(),
                sha256: digest(character),
            },
        }
    }

    struct CountingEmbedder {
        calls: AtomicUsize,
        returned_model: &'static str,
    }

    impl Embedder for CountingEmbedder {
        async fn embed(&self, texts: &[String]) -> rag_embedding::Result<Vec<Vec<f32>>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
        }
    }

    impl IdentifiedEmbedder for CountingEmbedder {
        async fn embed_identified(
            &self,
            texts: &[String],
        ) -> rag_embedding::Result<IdentifiedEmbeddingBatch> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(IdentifiedEmbeddingBatch {
                vectors: texts.iter().map(|_| vec![1.0, 0.0]).collect(),
                returned_model: self.returned_model.into(),
            })
        }
    }

    fn searchable_index(root: &Path, id: &str) -> FastIndex {
        let out = root.join(id);
        let documents = vec![
            FastDocument {
                document_id: format!("{id}-first"),
                document_sha256: "1".repeat(64),
                document_kind: "activity".into(),
                semantic_text: "encoded process launch".into(),
                facets_json: "[]".into(),
                relations_json: format!("[\"relation-{id}\"]"),
                occurrence_count: 1,
                vector_ordinal: 0,
            },
            FastDocument {
                document_id: format!("{id}-second"),
                document_sha256: "2".repeat(64),
                document_kind: "activity".into(),
                semantic_text: "ordinary browser launch".into(),
                facets_json: "[]".into(),
                relations_json: format!("[\"relation-{id}\"]"),
                occurrence_count: 1,
                vector_ordinal: 1,
            },
        ];
        let occurrences = documents
            .iter()
            .map(|document| FastOccurrence {
                occurrence_id: format!("occ-{}", document.document_id),
                document_id: document.document_id.clone(),
                event_time_ms: None,
                relation: format!("relation-{id}"),
                exact_attributes_json: "{}".into(),
                snapshot_sha256: "a".repeat(64),
                mapping_sha256: "b".repeat(64),
                event_id: format!("event-{}", document.document_id),
                support_ref: format!("support-{}", document.document_id),
            })
            .collect::<Vec<_>>();
        write_fast_index(
            &out,
            SourceBinding {
                snapshot_sha256: "a".repeat(64),
                mapping_sha256: "b".repeat(64),
            },
            BuildScope::Sample,
            &documents,
            &occurrences,
            &[vec![1.0, 0.0], vec![0.0, 1.0]],
            EmbeddingProfile {
                id: "profile".into(),
                version: "1".into(),
                sha256: "7".repeat(64),
                model: "local-model".into(),
                dimensions: 2,
                normalization: "l2".into(),
                vector_derivation: None,
                query_instruction: None,
                query_composition: None,
            },
        )
        .unwrap();
        FastIndex::open(&out).unwrap()
    }

    fn admitted_search_fixture() -> (tempfile::TempDir, AdmittedCatalogue) {
        let root = tempfile::tempdir().unwrap();
        let projection = ComponentRef {
            id: "projection".into(),
            version: "1".into(),
            sha256: digest('3'),
        };
        let profile = ComponentRef {
            id: "profile".into(),
            version: "1".into(),
            sha256: digest('7'),
        };
        let mut datasets = Vec::new();
        let mut entries = Vec::new();
        for (id, character) in [("dataset-a", '1'), ("dataset-b", '2')] {
            let mut index = searchable_index(root.path(), id);
            let dataset = dataset(id, character);
            let pipeline_provenance = PipelineProvenance {
                dataset_sha256: canonical_digest(&dataset).unwrap().to_string(),
                prepared_corpus_sha256: "4".repeat(64),
                embedding_plan_sha256: "5".repeat(64),
                embedding_result_set_sha256: "6".repeat(64),
            };
            index.manifest.pipeline_provenance = Some(pipeline_provenance);
            let entry = CatalogueDatasetEntry {
                dataset_sha256: canonical_digest(&dataset).unwrap(),
                dataset,
                projection_policy: projection.clone(),
                prepared_corpus: test_artifact(id, "prepared", '4'),
                embedding_plan: test_artifact(id, "plan", '5'),
                embedding_result_set: test_artifact(id, "results", '6'),
                embedding_profile: profile.clone(),
                final_index: CatalogueArtifactRef {
                    path: SafeRelativePath::new(format!("{id}/index.json")).unwrap(),
                    component: ComponentRef {
                        id: "index".into(),
                        version: "2".into(),
                        sha256: Digest::new(index.manifest.component_sha256.clone()).unwrap(),
                    },
                },
                searchable_document_count: 2,
                searchable_reference_count: 2,
                test_only: false,
            };
            entries.push(entry.clone());
            datasets.push(AdmittedDataset { entry, index });
        }
        let mut catalogue = DatasetCatalogue {
            schema_version: DATASET_CATALOGUE_SCHEMA.into(),
            component_sha256: digest('0'),
            mode: CatalogueMode::Normal,
            source_snapshot: entries[0].dataset.source_snapshot.clone(),
            mapping: entries[0].dataset.mapping.clone(),
            projection_policy: projection,
            embedding_profile: profile,
            query_compatibility: "single_embedding_profile".into(),
            rank_merge: "reciprocal_rank_fusion_v1".into(),
            datasets: entries,
            allowed_relation_overlaps: vec![],
        };
        catalogue.seal().unwrap();
        (
            root,
            AdmittedCatalogue {
                catalogue,
                datasets,
            },
        )
    }

    #[tokio::test]
    async fn catalogue_search_embeds_once_and_is_worker_order_independent() {
        let (_root, admitted) = admitted_search_fixture();
        let embedder = CountingEmbedder {
            calls: AtomicUsize::new(0),
            returned_model: "local-model",
        };
        let options = |workers| CatalogueSearchOptions {
            catalogue: Path::new("unused-by-admitted-search"),
            query: "encoded process",
            mode: Mode::Dense,
            top_n: 4,
            endpoint: "http://127.0.0.1:1234",
            relations: vec![],
            workers,
            allow_test_only: false,
        };
        let serial = search_admitted_catalogue(&admitted, &embedder, &options(1))
            .await
            .unwrap();
        assert_eq!(embedder.calls.load(Ordering::Relaxed), 1);
        let parallel = search_admitted_catalogue(&admitted, &embedder, &options(2))
            .await
            .unwrap();
        assert_eq!(embedder.calls.load(Ordering::Relaxed), 2);
        assert_eq!(serial, parallel);
        assert_eq!(
            serial
                .iter()
                .map(|hit| hit.dataset.id.as_str())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["dataset-a", "dataset-b"])
        );
    }

    #[tokio::test]
    async fn lexical_catalogue_search_does_not_embed() {
        let (_root, admitted) = admitted_search_fixture();
        let embedder = CountingEmbedder {
            calls: AtomicUsize::new(0),
            returned_model: "local-model",
        };
        let hits = search_admitted_catalogue(
            &admitted,
            &embedder,
            &CatalogueSearchOptions {
                catalogue: Path::new("unused-by-admitted-search"),
                query: "encoded process",
                mode: Mode::Lexical,
                top_n: 4,
                endpoint: "http://127.0.0.1:1234",
                relations: vec![],
                workers: 2,
                allow_test_only: false,
            },
        )
        .await
        .unwrap();
        assert_eq!(embedder.calls.load(Ordering::Relaxed), 0);
        assert_eq!(hits.len(), 2);
    }

    fn batch_request(query_id: &str, mode: Mode) -> BatchQueryRequest {
        BatchQueryRequest {
            query_id: query_id.into(),
            query: "encoded process".into(),
            mode,
            top_n: 4,
            relations: vec![],
        }
    }

    fn write_batch_requests(root: &Path, name: &str, requests: &[BatchQueryRequest]) -> PathBuf {
        let path = root.join(name);
        let mut bytes = Vec::new();
        for request in requests {
            serde_json::to_writer(&mut bytes, request).unwrap();
            bytes.push(b'\n');
        }
        fs::write(&path, bytes).unwrap();
        path
    }

    async fn publish_test_batch_search_run<E: IdentifiedEmbedder>(
        admitted: &AdmittedCatalogue,
        embedder: &E,
        requests: &[BatchQueryRequest],
        requests_path: &Path,
        out: &Path,
        workers: usize,
        allow_test_only: bool,
    ) -> Result<CatalogueBatchSearchManifest> {
        let receipt = batch_file_receipt(requests_path, BATCH_REQUEST_FILE, requests.len())?;
        publish_batch_search_run(
            admitted,
            embedder,
            requests,
            BatchRunExecutionOptions {
                requests_path,
                expected_request_sha256: &receipt.sha256,
                expected_request_bytes: receipt.bytes,
                out,
                workers,
                allow_test_only,
            },
        )
        .await
    }

    struct FailOnSecondEmbedder {
        calls: AtomicUsize,
    }

    impl Embedder for FailOnSecondEmbedder {
        async fn embed(&self, texts: &[String]) -> rag_embedding::Result<Vec<Vec<f32>>> {
            if self.calls.fetch_add(1, Ordering::Relaxed) == 1 {
                return Err(rag_embedding::EmbeddingError::Invalid(
                    "intentional second-call failure",
                ));
            }
            Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
        }
    }

    impl IdentifiedEmbedder for FailOnSecondEmbedder {
        async fn embed_identified(
            &self,
            texts: &[String],
        ) -> rag_embedding::Result<IdentifiedEmbeddingBatch> {
            Ok(IdentifiedEmbeddingBatch {
                vectors: Embedder::embed(self, texts).await?,
                returned_model: "local-model".into(),
            })
        }
    }

    #[tokio::test]
    async fn catalogue_batch_search_publishes_atomic_ordered_tree_and_reuses_query_vector() {
        let (root, admitted) = admitted_search_fixture();
        let embedder = CountingEmbedder {
            calls: AtomicUsize::new(0),
            returned_model: "local-model",
        };
        let requests = vec![
            batch_request("q-lexical", Mode::Lexical),
            batch_request("q-semantic", Mode::Dense),
            batch_request("q-semantic", Mode::Fused),
        ];
        let request_path = write_batch_requests(root.path(), "input.jsonl", &requests);
        let request_bytes = fs::read(&request_path).unwrap();
        let out = root.path().join("run");
        let manifest = publish_test_batch_search_run(
            &admitted,
            &embedder,
            &requests,
            &request_path,
            &out,
            2,
            false,
        )
        .await
        .unwrap();
        assert_eq!(embedder.calls.load(Ordering::Relaxed), 1);
        assert_eq!(manifest.model.calls, 1);
        assert_eq!(manifest.query_vectors.len(), 1);
        assert_eq!(
            manifest.component_sha256,
            component_digest(&manifest).unwrap()
        );
        assert_eq!(
            fs::read(out.join(BATCH_REQUEST_FILE)).unwrap(),
            request_bytes
        );
        assert!(out.join(BATCH_MANIFEST_FILE).is_file());
        let rows = fs::read_to_string(out.join(BATCH_RESULT_FILE))
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(rows.len(), 3);
        assert_eq!(
            rows.iter()
                .map(|row| row["query_id"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["q-lexical", "q-semantic", "q-semantic"]
        );
        assert!(rows.iter().all(|row| {
            row["schema_version"] == "livefire.rag.catalogue-batch-search-result/1"
                && row["catalogue_sha256"] == admitted.catalogue.component_sha256.as_str()
                && row["query"] == "encoded process"
                && row["top_n"] == 4
                && row["rank_merge"] == admitted.catalogue.rank_merge
                && row["hits"].as_array().is_some_and(|hits| !hits.is_empty())
        }));
        let mut wrong_digest = manifest;
        wrong_digest.component_sha256 = digest('1');
        assert!(validate_batch_manifest(&wrong_digest).is_err());
    }

    #[tokio::test]
    async fn lexical_catalogue_batch_search_makes_zero_model_calls() {
        let (root, admitted) = admitted_search_fixture();
        let embedder = CountingEmbedder {
            calls: AtomicUsize::new(0),
            returned_model: "local-model",
        };
        let requests = vec![
            batch_request("q-lexical-1", Mode::Lexical),
            batch_request("q-lexical-2", Mode::Lexical),
        ];
        let request_path = write_batch_requests(root.path(), "lexical.jsonl", &requests);
        let out = root.path().join("lexical-run");
        let manifest = publish_test_batch_search_run(
            &admitted,
            &embedder,
            &requests,
            &request_path,
            &out,
            2,
            false,
        )
        .await
        .unwrap();
        assert_eq!(embedder.calls.load(Ordering::Relaxed), 0);
        assert_eq!(manifest.model.calls, 0);
        assert_eq!(manifest.model.status, "not_used_all_lexical");
        assert!(manifest.model.returned_model.is_none());
        assert_eq!(
            fs::read_to_string(out.join(BATCH_RESULT_FILE))
                .unwrap()
                .lines()
                .count(),
            2
        );
    }

    #[tokio::test]
    async fn different_semantic_queries_make_distinct_model_calls() {
        let (root, admitted) = admitted_search_fixture();
        let embedder = CountingEmbedder {
            calls: AtomicUsize::new(0),
            returned_model: "local-model",
        };
        let requests = vec![
            BatchQueryRequest {
                query: "first semantic query".into(),
                ..batch_request("q-first", Mode::Dense)
            },
            BatchQueryRequest {
                query: "second semantic query".into(),
                ..batch_request("q-second", Mode::Fused)
            },
        ];
        let request_path = write_batch_requests(root.path(), "distinct.jsonl", &requests);
        let manifest = publish_test_batch_search_run(
            &admitted,
            &embedder,
            &requests,
            &request_path,
            &root.path().join("distinct-run"),
            2,
            false,
        )
        .await
        .unwrap();
        assert_eq!(embedder.calls.load(Ordering::Relaxed), 2);
        assert_eq!(manifest.model.calls, 2);
        assert_eq!(manifest.query_vectors.len(), 2);
    }

    #[tokio::test]
    async fn test_only_catalogue_batch_search_refuses_before_model_or_output() {
        let (root, mut admitted) = admitted_search_fixture();
        admitted.catalogue.mode = CatalogueMode::TestOnly;
        let embedder = CountingEmbedder {
            calls: AtomicUsize::new(0),
            returned_model: "local-model",
        };
        let requests = vec![batch_request("q-dense", Mode::Dense)];
        let request_path = write_batch_requests(root.path(), "test-only.jsonl", &requests);
        let out = root.path().join("refused-run");
        let error = publish_test_batch_search_run(
            &admitted,
            &embedder,
            &requests,
            &request_path,
            &out,
            2,
            false,
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("--allow-test-only"));
        assert_eq!(embedder.calls.load(Ordering::Relaxed), 0);
        assert!(!out.exists());

        publish_test_batch_search_run(
            &admitted,
            &embedder,
            &requests,
            &request_path,
            &out,
            2,
            true,
        )
        .await
        .unwrap();
        assert_eq!(embedder.calls.load(Ordering::Relaxed), 1);
        assert!(out.join(BATCH_MANIFEST_FILE).is_file());
    }

    #[tokio::test]
    async fn catalogue_batch_search_failure_and_existing_output_publish_nothing() {
        let (root, admitted) = admitted_search_fixture();
        let requests = vec![
            BatchQueryRequest {
                query: "first query".into(),
                ..batch_request("q-first", Mode::Dense)
            },
            BatchQueryRequest {
                query: "second query".into(),
                ..batch_request("q-second", Mode::Fused)
            },
        ];
        let request_path = write_batch_requests(root.path(), "failure.jsonl", &requests);
        let out = root.path().join("failed-run");
        let failing = FailOnSecondEmbedder {
            calls: AtomicUsize::new(0),
        };
        assert!(
            publish_test_batch_search_run(
                &admitted,
                &failing,
                &requests,
                &request_path,
                &out,
                2,
                false,
            )
            .await
            .is_err()
        );
        assert_eq!(failing.calls.load(Ordering::Relaxed), 2);
        assert!(!out.exists());
        assert!(fs::read_dir(root.path()).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains("partial")
        }));

        fs::create_dir(&out).unwrap();
        let counting = CountingEmbedder {
            calls: AtomicUsize::new(0),
            returned_model: "local-model",
        };
        assert!(
            publish_test_batch_search_run(
                &admitted,
                &counting,
                &requests,
                &request_path,
                &out,
                2,
                false,
            )
            .await
            .is_err()
        );
        assert_eq!(counting.calls.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn catalogue_batch_search_rejects_request_scope_before_output_or_model() {
        let (root, admitted) = admitted_search_fixture();
        let embedder = CountingEmbedder {
            calls: AtomicUsize::new(0),
            returned_model: "local-model",
        };
        for (name, request) in [
            (
                "unknown",
                BatchQueryRequest {
                    relations: vec!["relation-unknown".into()],
                    ..batch_request("q", Mode::Dense)
                },
            ),
            (
                "unsorted",
                BatchQueryRequest {
                    relations: vec!["relation-dataset-b".into(), "relation-dataset-a".into()],
                    ..batch_request("q", Mode::Dense)
                },
            ),
            (
                "duplicate",
                BatchQueryRequest {
                    relations: vec!["relation-dataset-a".into(), "relation-dataset-a".into()],
                    ..batch_request("q", Mode::Dense)
                },
            ),
        ] {
            let requests = vec![request];
            let request_path =
                write_batch_requests(root.path(), &format!("{name}.jsonl"), &requests);
            let out = root.path().join(format!("{name}-run"));
            assert!(
                publish_test_batch_search_run(
                    &admitted,
                    &embedder,
                    &requests,
                    &request_path,
                    &out,
                    2,
                    false,
                )
                .await
                .is_err()
            );
            assert!(!out.exists());
        }
        assert_eq!(embedder.calls.load(Ordering::Relaxed), 0);

        let over_limit = (0..=crate::MAX_BATCH_QUERY_REQUESTS)
            .map(|ordinal| batch_request(&format!("q-{ordinal}"), Mode::Lexical))
            .collect::<Vec<_>>();
        assert!(validate_catalogue_batch_requests(&admitted, &over_limit).is_err());
    }

    #[tokio::test]
    async fn dense_catalogue_search_rejects_a_different_returned_model() {
        let (_root, admitted) = admitted_search_fixture();
        let embedder = CountingEmbedder {
            calls: AtomicUsize::new(0),
            returned_model: "different-model",
        };
        let error = search_admitted_catalogue(
            &admitted,
            &embedder,
            &CatalogueSearchOptions {
                catalogue: Path::new("unused-by-admitted-search"),
                query: "encoded process",
                mode: Mode::Dense,
                top_n: 4,
                endpoint: "http://127.0.0.1:1234",
                relations: vec![],
                workers: 2,
                allow_test_only: false,
            },
        )
        .await
        .unwrap_err();
        assert_eq!(embedder.calls.load(Ordering::Relaxed), 1);
        assert!(error.to_string().contains("different model"));
    }

    #[test]
    fn final_index_admission_rejects_stale_identity_count_and_profile() {
        let (_root, admitted) = admitted_search_fixture();
        let dataset = &admitted.datasets[0];
        let planned_profile = rag_pipeline::EmbeddingProfileRef {
            component: dataset.entry.embedding_profile.clone(),
            model_artifact: ComponentRef {
                id: "model".into(),
                version: "1".into(),
                sha256: digest('9'),
            },
            tokenizer: ComponentRef {
                id: "tokenizer".into(),
                version: "1".into(),
                sha256: digest('8'),
            },
            maximum_input_tokens: 1_024,
            pooling: "last_token".into(),
            normalization: "l2".into(),
            dimensions: 2,
            dtype: "f32le".into(),
            document_format: "{semantic_text}".into(),
        };
        let expected_provenance = dataset.index.manifest.pipeline_provenance.as_ref().unwrap();
        let expected_order = &dataset.index.manifest.vectors.document_order_sha256;
        validate_final_index_binding(
            &dataset.entry,
            &planned_profile,
            expected_provenance,
            expected_order,
            &dataset.index,
        )
        .unwrap();

        for field in 0..4 {
            let mut wrong_provenance = expected_provenance.clone();
            match field {
                0 => wrong_provenance.dataset_sha256 = "0".repeat(64),
                1 => wrong_provenance.prepared_corpus_sha256 = "0".repeat(64),
                2 => wrong_provenance.embedding_plan_sha256 = "0".repeat(64),
                _ => wrong_provenance.embedding_result_set_sha256 = "0".repeat(64),
            }
            assert!(
                validate_final_index_binding(
                    &dataset.entry,
                    &planned_profile,
                    &wrong_provenance,
                    expected_order,
                    &dataset.index,
                )
                .is_err(),
                "upstream provenance field {field}"
            );
        }

        let mut stale_digest = dataset.entry.clone();
        stale_digest.final_index.component.sha256 = digest('0');
        assert!(
            validate_final_index_binding(
                &stale_digest,
                &planned_profile,
                expected_provenance,
                expected_order,
                &dataset.index
            )
            .is_err()
        );
        let mut stale_count = dataset.entry.clone();
        stale_count.searchable_reference_count += 1;
        assert!(
            validate_final_index_binding(
                &stale_count,
                &planned_profile,
                expected_provenance,
                expected_order,
                &dataset.index
            )
            .is_err()
        );
        let mut mixed_profile = planned_profile;
        mixed_profile.normalization = "none".into();
        assert!(
            validate_final_index_binding(
                &dataset.entry,
                &mixed_profile,
                expected_provenance,
                expected_order,
                &dataset.index
            )
            .is_err()
        );
    }

    #[test]
    fn final_index_admission_rejects_an_index_from_another_pipeline_chain() {
        let (_root, admitted) = admitted_search_fixture();
        let expected = &admitted.datasets[0];
        let swapped = &admitted.datasets[1];
        let mut entry = expected.entry.clone();
        // Model the build command receiving the wrong index path: it records
        // that index's valid self digest and matching counts, so those checks
        // alone cannot identify the swap.
        entry.final_index.component.sha256 =
            Digest::new(swapped.index.manifest.component_sha256.clone()).unwrap();
        let planned_profile = rag_pipeline::EmbeddingProfileRef {
            component: entry.embedding_profile.clone(),
            model_artifact: ComponentRef {
                id: "model".into(),
                version: "1".into(),
                sha256: digest('9'),
            },
            tokenizer: ComponentRef {
                id: "tokenizer".into(),
                version: "1".into(),
                sha256: digest('8'),
            },
            maximum_input_tokens: 1_024,
            pooling: "last_token".into(),
            normalization: "l2".into(),
            dimensions: 2,
            dtype: "f32le".into(),
            document_format: "{semantic_text}".into(),
        };
        assert!(
            validate_final_index_binding(
                &entry,
                &planned_profile,
                expected
                    .index
                    .manifest
                    .pipeline_provenance
                    .as_ref()
                    .unwrap(),
                // Use the swapped index's order here to isolate the upstream
                // provenance check from the separate order check.
                &swapped.index.manifest.vectors.document_order_sha256,
                &swapped.index,
            )
            .is_err()
        );
    }
}

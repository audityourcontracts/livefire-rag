//! Dataset-oriented prepare, embed, and assemble commands.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{BufReader, Read},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use arrow_array::RecordBatch;
use rag_embedding::{
    EmbeddingShard, EmbeddingTaskOptions, LmStudioEmbedder, RetryPolicy, decode_sha256_hex,
    execute_embedding_task, format_document_input, parse_bound_embedding_profile,
    parse_embedding_profile, validate_vector,
};
use rag_index::{
    BuildScope, FastDocument, FastIndexManifest, OrderedVectorShard, SourceBinding,
    documents_from_parquet_shards, occurrences_from_parquet_shards, vectors_from_embedding_shards,
    write_fast_index_from_streams,
};
use rag_ocsf::{LocalSnapshotReader, SnapshotReader};
use rag_pipeline::{
    AtomicDirectory, ComponentRef, DatasetIdentity, Digest, DocumentKind, EMBEDDING_PLAN_SCHEMA,
    EmbeddingInputSlice, EmbeddingPlan, EmbeddingProfileRef, EmbeddingResultSetManifest,
    EmbeddingTask, ExecutorReceipt, ObjectEntry, PREPARED_CORPUS_SCHEMA, PreparedCorpusManifest,
    PreparedDocumentObject, PreparedDocumentRow, PreparedOccurrenceObject, PreparedOccurrenceRow,
    RESULT_SET_SCHEMA, ReceiptEntry, RelationAccounting, SafeRelativePath, VECTOR_RECEIPT_SCHEMA,
    VectorObject, VectorResultReceipt, canonical_digest, canonical_json_bytes, digest_bytes,
    document_order_digest, embedding_input_order_digest, read_json, read_prepared_documents,
    read_prepared_occurrences, remove_stale_atomic_writes, resolve_existing_artifact,
    resolve_output_artifact, validate_prepared_documents, write_canonical_json,
    write_prepared_documents, write_prepared_occurrences,
};
use rag_projection::{
    ComponentRef as ProjectionComponentRef, ProjectionContext, ProjectionInput, project,
};
use serde_json::{Value, json};
use sha2::{Digest as ShaDigest, Sha256};

use super::{Error, Result, fast_document, parse_event_time_ms, strings};

const MANIFEST_FILE: &str = "manifest.json";
const DOCUMENT_SCHEMA_BYTES: &[u8] =
    include_bytes!("../../../specs/prepared-document-row.v1.schema.json");
const OCCURRENCE_SCHEMA_BYTES: &[u8] =
    include_bytes!("../../../specs/prepared-occurrence-row.v1.schema.json");
const PROJECTION_POLICY_BYTES: &[u8] =
    include_bytes!("../../../specs/evidence-projection-policy.v2.json");
const PREPARATION_SOURCE_BYTES: &[u8] = include_bytes!("portable.rs");
const OCCURRENCE_SHARD_ROWS: usize = 8_192;
// Preparation still groups semantic documents in memory. This explicit limit
// keeps that known first-iteration constraint from becoming an unbounded OOM.
const MAX_IN_MEMORY_DOCUMENTS: usize = 600_000;

pub(crate) struct PrepareOptions {
    pub snapshot: PathBuf,
    pub dataset_id: String,
    pub dataset_version: String,
    pub relations: Vec<String>,
    pub out: PathBuf,
    pub document_shard_rows: usize,
}

pub(crate) struct PlanOptions {
    pub prepared: PathBuf,
    pub embedding_profile: PathBuf,
    pub out: PathBuf,
    pub task_documents: usize,
}

pub(crate) struct EmbedOptions {
    pub prepared: PathBuf,
    pub plan: PathBuf,
    pub embedding_profile: PathBuf,
    pub embedding_endpoint: String,
    pub out: PathBuf,
    pub batch_size: usize,
    pub requests_in_flight: usize,
}

pub(crate) struct AssembleOptions {
    pub prepared: PathBuf,
    pub plan: PathBuf,
    pub embeddings: PathBuf,
    pub embedding_profile: PathBuf,
    pub out: PathBuf,
}

#[derive(Debug)]
struct DocumentAccumulator {
    document: FastDocument,
    primary_relation: String,
    relations: BTreeSet<String>,
}

pub(crate) fn prepare(options: PrepareOptions) -> Result<()> {
    if options.dataset_id.is_empty()
        || options.dataset_version.is_empty()
        || options.document_shard_rows == 0
        || options.document_shard_rows > 65_536
    {
        return Err(Error::AccountingClosure(
            "invalid dataset preparation options",
        ));
    }
    let reader = LocalSnapshotReader::open(&options.snapshot)?;
    let identity = reader.identity();
    let mut included = options.relations;
    included.sort();
    included.dedup();
    if included.is_empty() || included.iter().any(|relation| relation.is_empty()) {
        return Err(Error::AccountingClosure("dataset relation scope is empty"));
    }
    let available = reader
        .typed_relations()
        .map(|relation| (relation.name.as_str(), relation))
        .collect::<BTreeMap<_, _>>();
    if included
        .iter()
        .any(|relation| !available.contains_key(relation.as_str()))
        || included
            .iter()
            .any(|relation| relation == "ocsf_ext_livefire_system_metric")
    {
        return Err(Error::AccountingClosure(
            "dataset relation scope is invalid",
        ));
    }

    let context = ProjectionContext {
        snapshot: ProjectionComponentRef {
            id: identity.snapshot_id.clone(),
            version: identity.snapshot_version.clone(),
            sha256: identity.snapshot_sha256.to_string(),
            uri: None,
        },
        mapping_pack: ProjectionComponentRef {
            id: identity.mapping_id.clone(),
            version: identity.mapping_version.clone(),
            sha256: identity.mapping_sha256.to_string(),
            uri: None,
        },
    };
    let staging = AtomicDirectory::new(&options.out)?;
    let root = staging.path();
    let mut documents = BTreeMap::<String, DocumentAccumulator>::new();
    let mut searchable_by_relation = BTreeMap::<String, u64>::new();
    let mut occurrence_objects = Vec::new();
    let mut occurrence_count = 0_u64;

    for relation_name in &included {
        let relation = available[relation_name.as_str()];
        let mut source_row_ordinal = 0_u64;
        let mut occurrence_buffer = Vec::with_capacity(OCCURRENCE_SHARD_ROWS);
        let mut relation_part = 0_u64;
        for batch in reader.scan(relation)? {
            let batch = batch?;
            project_prepared_batch(
                &batch,
                relation_name,
                &context,
                &mut source_row_ordinal,
                &mut documents,
                &mut occurrence_buffer,
            )?;
            flush_occurrence_shards(
                root,
                relation_name,
                &mut occurrence_buffer,
                &mut relation_part,
                &mut occurrence_objects,
                false,
            )?;
        }
        let relation_occurrence_count = relation_part
            .checked_mul(OCCURRENCE_SHARD_ROWS as u64)
            .and_then(|count| count.checked_add(occurrence_buffer.len() as u64))
            .ok_or(Error::CountOverflow)?;
        flush_occurrence_shards(
            root,
            relation_name,
            &mut occurrence_buffer,
            &mut relation_part,
            &mut occurrence_objects,
            true,
        )?;
        occurrence_count = occurrence_count
            .checked_add(relation_occurrence_count)
            .ok_or(Error::CountOverflow)?;
        searchable_by_relation.insert(relation_name.clone(), relation_occurrence_count);
    }
    if documents.is_empty() {
        return Err(Error::AccountingClosure(
            "dataset produced no searchable documents",
        ));
    }

    let mut prepared_documents = Vec::with_capacity(documents.len());
    for (ordinal, mut accumulated) in documents.into_values().enumerate() {
        accumulated.document.vector_ordinal = ordinal as u64;
        accumulated.document.relations_json = canonical_string(&accumulated.relations)?;
        prepared_documents.push(PreparedDocumentRow {
            document_ordinal: ordinal as u64,
            document_id: accumulated.document.document_id,
            document_sha256: Digest::new(accumulated.document.document_sha256)?,
            semantic_text_sha256: digest_bytes(accumulated.document.semantic_text.as_bytes()),
            semantic_text: accumulated.document.semantic_text,
            document_kind: match accumulated.document.document_kind.as_str() {
                "activity" => DocumentKind::Activity,
                "state" => DocumentKind::State,
                "detection" => DocumentKind::Detection,
                _ => return Err(Error::AccountingClosure("unknown projected document kind")),
            },
            primary_relation: accumulated.primary_relation,
            facets_json: accumulated.document.facets_json,
            relations_json: accumulated.document.relations_json,
            occurrence_count: accumulated.document.occurrence_count,
        });
    }
    let all_typed = reader
        .typed_relations()
        .map(|relation| relation.name.clone())
        .collect::<Vec<_>>();
    let structured_only = all_typed
        .iter()
        .filter(|relation| relation.as_str() == "ocsf_ext_livefire_system_metric")
        .cloned()
        .collect::<Vec<_>>();
    let excluded = all_typed
        .iter()
        .filter(|relation| !included.contains(relation) && !structured_only.contains(relation))
        .cloned()
        .collect::<Vec<_>>();
    let dataset = DatasetIdentity {
        id: options.dataset_id,
        version: options.dataset_version,
        source_snapshot: component(
            &identity.snapshot_id,
            &identity.snapshot_version,
            identity.snapshot_sha256.as_str(),
        )?,
        mapping: component(
            &identity.mapping_id,
            &identity.mapping_version,
            identity.mapping_sha256.as_str(),
        )?,
        included_relations: included.clone(),
        excluded_relations: excluded.clone(),
        structured_only_relations: structured_only.clone(),
    };

    let mut document_objects = Vec::new();
    for (ordinal, rows) in prepared_documents
        .chunks(options.document_shard_rows)
        .enumerate()
    {
        let relative = format!("documents/part-{ordinal:06}.parquet");
        let path = root.join(&relative);
        write_prepared_documents(&path, rows)?;
        document_objects.push(PreparedDocumentObject {
            object: object_entry(
                &relative,
                &path,
                rows.len() as u64,
                canonical_digest(&rows)?,
            )?,
            ordinal: ordinal as u32,
            first_document_id: rows[0].document_id.clone(),
            last_document_id: rows[rows.len() - 1].document_id.clone(),
            embedding_input_order_sha256: embedding_input_order_digest(rows),
        });
    }
    // Object ordinals are global and contiguous across relation directories.
    for (ordinal, object) in occurrence_objects.iter_mut().enumerate() {
        object.ordinal = ordinal as u32;
    }

    let mut relation_accounting = BTreeMap::new();
    for relation in reader.typed_relations() {
        let searchable = searchable_by_relation
            .get(&relation.name)
            .copied()
            .unwrap_or(0);
        relation_accounting.insert(
            relation.name.clone(),
            RelationAccounting {
                source_rows: relation.rows,
                searchable_occurrences: searchable,
                selected_occurrences: searchable,
                excluded_rows: if included.contains(&relation.name) {
                    relation
                        .rows
                        .checked_sub(searchable)
                        .ok_or(Error::AccountingClosure(
                            "searchable rows exceed source rows",
                        ))?
                } else {
                    relation.rows
                },
            },
        );
    }
    let mut manifest = PreparedCorpusManifest {
        schema_version: PREPARED_CORPUS_SCHEMA.into(),
        component_sha256: zero_digest()?,
        dataset,
        projection_policy: component(
            "livefire.rag.generic-evidence-projection-policy",
            "2",
            &sha256_bytes(PROJECTION_POLICY_BYTES),
        )?,
        document_schema: component(
            "livefire.rag.prepared-document-row",
            "1",
            &sha256_bytes(DOCUMENT_SCHEMA_BYTES),
        )?,
        occurrence_schema: component(
            "livefire.rag.prepared-occurrence-row",
            "1",
            &sha256_bytes(OCCURRENCE_SCHEMA_BYTES),
        )?,
        preparation_implementation: component(
            "livefire.rag.portable-preparation",
            env!("CARGO_PKG_VERSION"),
            &sha256_bytes(PREPARATION_SOURCE_BYTES),
        )?,
        document_count: prepared_documents.len() as u64,
        occurrence_count,
        document_order_sha256: document_order_digest(
            prepared_documents
                .iter()
                .map(|row| row.document_id.as_str()),
        ),
        embedding_input_order_sha256: embedding_input_order_digest(&prepared_documents),
        documents: document_objects,
        occurrences: occurrence_objects,
        relation_accounting,
    };
    manifest.seal()?;
    validate_prepared_documents(&manifest, &prepared_documents)?;
    write_canonical_json(&root.join(MANIFEST_FILE), &manifest)?;
    write_canonical_json(
        &root.join("accounting.json"),
        &json!({
            "schema_version": "livefire.rag.prepared-accounting/1",
            "dataset": manifest.dataset,
            "documents": manifest.document_count,
            "occurrences": manifest.occurrence_count,
            "preparation_document_grouping": "in_memory_btree_v1",
            "preparation_document_limit": MAX_IN_MEMORY_DOCUMENTS,
            "occurrence_shard_rows": OCCURRENCE_SHARD_ROWS,
            "relations": manifest.relation_accounting,
        }),
    )?;
    staging.publish()?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}

pub(crate) fn plan_embeddings(options: PlanOptions) -> Result<()> {
    if options.task_documents == 0 || options.task_documents > 65_536 {
        return Err(Error::AccountingClosure("invalid embedding task size"));
    }
    let prepared = load_prepared(&options.prepared)?;
    let profile_bytes = fs::read(&options.embedding_profile)?;
    let compact = parse_embedding_profile(&profile_bytes)?;
    let profile = profile_ref(&profile_bytes, &compact)?;
    let mut tasks = Vec::new();
    let mut ordinal = 0_u64;
    for object in &prepared.documents {
        let rows = read_prepared_documents(&object.object.path.join_to(&options.prepared))?;
        for (offset, task_rows) in rows.chunks(options.task_documents).enumerate() {
            let start = ordinal;
            let count = task_rows.len() as u64;
            let end = start + count;
            let order = embedding_input_order_digest(task_rows);
            let task_material = json!({
                "prepared": prepared.component_sha256,
                "profile": profile.component.sha256,
                "start": start,
                "end": end,
                "order": order,
            });
            let task_id = canonical_digest(&task_material)?.to_string();
            let sequence = tasks.len();
            tasks.push(EmbeddingTask {
                task_id,
                ordinal_start: start,
                ordinal_end: end,
                input_slices: vec![EmbeddingInputSlice {
                    path: object.object.path.clone(),
                    object_sha256: object.object.sha256.clone(),
                    row_offset: (offset * options.task_documents) as u64,
                    rows: count,
                    embedding_input_order_sha256: order.clone(),
                }],
                embedding_input_order_sha256: order,
                result_path: SafeRelativePath::new(format!("parts/part-{sequence:06}.f32"))?,
                receipt_path: SafeRelativePath::new(format!("receipts/part-{sequence:06}.json"))?,
            });
            ordinal = end;
        }
    }
    let mut plan = EmbeddingPlan {
        schema_version: EMBEDDING_PLAN_SCHEMA.into(),
        component_sha256: zero_digest()?,
        prepared_corpus_sha256: prepared.component_sha256.clone(),
        dataset: prepared.dataset.clone(),
        embedding_profile: profile,
        document_count: prepared.document_count,
        document_order_sha256: prepared.document_order_sha256.clone(),
        tasks,
    };
    plan.seal()?;
    validate_plan_streaming(&options.prepared, &prepared, &plan)?;
    let staging = AtomicDirectory::new(&options.out)?;
    write_canonical_json(&staging.path().join("plan.json"), &plan)?;
    staging.publish()?;
    println!("{}", serde_json::to_string_pretty(&plan)?);
    Ok(())
}

pub(crate) async fn embed(options: EmbedOptions) -> Result<()> {
    let prepared = load_prepared(&options.prepared)?;
    let plan: EmbeddingPlan = read_json(&manifest_or_file(&options.plan, "plan.json"))?;
    validate_plan_streaming(&options.prepared, &prepared, &plan)?;
    let profile_bytes = fs::read(&options.embedding_profile)?;
    let profile = parse_bound_embedding_profile(
        &profile_bytes,
        plan.embedding_profile.component.sha256.as_str(),
    )?;
    validate_plan_profile_fields(&plan.embedding_profile, &profile_bytes, &profile)?;
    let runtime = component_from_value(
        serde_json::from_slice::<Value>(&profile_bytes)?
            .get("runtime")
            .ok_or(Error::AccountingClosure(
                "embedding runtime component is absent",
            ))?,
    )?;
    fs::create_dir_all(options.out.join("parts"))?;
    fs::create_dir_all(options.out.join("receipts"))?;
    let embedder = Arc::new(LmStudioEmbedder::with_timeout(
        &options.embedding_endpoint,
        &profile.model,
        Duration::from_secs(300),
    )?);
    remove_stale_embedding_partials(&options.out)?;
    let mut conformance_validated = false;
    let mut receipts = Vec::with_capacity(plan.tasks.len());
    let mut entries = Vec::with_capacity(plan.tasks.len());
    let mut task_documents = TaskDocumentLoader::new(&options.prepared, &prepared);
    for task in &plan.tasks {
        let rows = task_documents.load(task)?;
        let texts = rows
            .iter()
            .map(|row| {
                let input = format_document_input(
                    &plan.embedding_profile.document_format,
                    &row.semantic_text,
                )?;
                if input.len()
                    > usize::try_from(plan.embedding_profile.maximum_input_tokens)
                        .map_err(|_| rag_embedding::EmbeddingError::Invalid("token bound"))?
                {
                    return Err(rag_embedding::EmbeddingError::Invalid(
                        "document exceeds conservative token bound",
                    ));
                }
                Ok(input)
            })
            .collect::<rag_embedding::Result<Vec<_>>>()?;
        let input_token_upper_bound = texts.iter().try_fold(0_u64, |total, input| {
            total
                .checked_add(input.len() as u64)
                .ok_or(Error::CountOverflow)
        })?;
        let vector_path = resolve_output_artifact(&options.out, &task.result_path)?;
        let began = Instant::now();
        let receipt_path = resolve_output_artifact(&options.out, &task.receipt_path)?;
        if let Some(receipt) = validate_completed_embedding_task(
            &receipt_path,
            &vector_path,
            task,
            &plan,
            &profile,
            &runtime,
        )? {
            entries.push(ReceiptEntry {
                task_id: task.task_id.clone(),
                path: task.receipt_path.clone(),
                sha256: receipt.component_sha256.clone(),
            });
            receipts.push(receipt);
            continue;
        }
        if !conformance_validated {
            validate_lmstudio_conformance(&embedder, &profile_bytes, &profile).await?;
            conformance_validated = true;
        }
        let stats = execute_embedding_task(
            Arc::clone(&embedder),
            &profile,
            &texts,
            &vector_path,
            decode_sha256_hex(task.embedding_input_order_sha256.as_str())?,
            EmbeddingTaskOptions {
                batch_size: options.batch_size,
                max_in_flight: options.requests_in_flight,
                retry: RetryPolicy::default(),
            },
        )
        .await?;
        let shard = EmbeddingShard::open(&vector_path)?;
        shard.validate_normalization(&profile.normalization)?;
        let mut receipt = VectorResultReceipt {
            schema_version: VECTOR_RECEIPT_SCHEMA.into(),
            component_sha256: zero_digest()?,
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
                bytes: fs::metadata(&vector_path)?.len(),
                sha256: file_digest(&vector_path)?,
                dimensions: profile.dimensions,
                dtype: "f32le".into(),
                embedding_input_order_sha256: task.embedding_input_order_sha256.clone(),
            },
            executor: ExecutorReceipt {
                implementation: component(
                    "livefire.rag.embedding-executor.lmstudio",
                    env!("CARGO_PKG_VERSION"),
                    &sha256_bytes(include_bytes!("../../../crates/rag-embedding/src/task.rs")),
                )?,
                runtime: runtime.clone(),
                returned_model: stats.returned_model.clone(),
                requests: stats.requests as u64,
                retries: stats.retries as u64,
                // UTF-8 bytes are a conservative upper bound on BPE tokens.
                input_bytes_upper_bound: input_token_upper_bound,
                elapsed_ms: began.elapsed().as_millis().try_into().unwrap_or(u64::MAX),
                conformance_passed: true,
            },
            finite_values_validated: true,
            normalization_validated: true,
        };
        receipt.seal()?;
        receipt.validate_against(&plan)?;
        write_canonical_json(&receipt_path, &receipt)?;
        entries.push(ReceiptEntry {
            task_id: task.task_id.clone(),
            path: task.receipt_path.clone(),
            sha256: receipt.component_sha256.clone(),
        });
        receipts.push(receipt);
    }
    let mut result_set = EmbeddingResultSetManifest {
        schema_version: RESULT_SET_SCHEMA.into(),
        component_sha256: zero_digest()?,
        plan_sha256: plan.component_sha256.clone(),
        prepared_corpus_sha256: plan.prepared_corpus_sha256.clone(),
        embedding_profile_sha256: plan.embedding_profile.component.sha256.clone(),
        document_count: plan.document_count,
        document_order_sha256: plan.document_order_sha256.clone(),
        receipts: entries,
    };
    result_set.seal()?;
    result_set.validate(&plan, &receipts)?;
    let result_manifest_path = options.out.join(MANIFEST_FILE);
    if result_manifest_path.exists() {
        let existing: EmbeddingResultSetManifest = read_json(&result_manifest_path)?;
        if existing.validate(&plan, &receipts).is_err() || existing != result_set {
            fs::remove_file(&result_manifest_path)?;
            write_canonical_json(&result_manifest_path, &result_set)?;
        }
    } else {
        write_canonical_json(&result_manifest_path, &result_set)?;
    }
    validate_embedding_artifact_coverage(&options.out, &plan)?;
    println!("{}", serde_json::to_string_pretty(&result_set)?);
    Ok(())
}

pub(crate) fn assemble(options: AssembleOptions) -> Result<()> {
    let prepared = load_prepared(&options.prepared)?;
    let plan: EmbeddingPlan = read_json(&manifest_or_file(&options.plan, "plan.json"))?;
    let result_set: EmbeddingResultSetManifest =
        read_json(&options.embeddings.join(MANIFEST_FILE))?;
    validate_embedding_artifact_coverage(&options.embeddings, &plan)?;
    validate_plan_streaming(&options.prepared, &prepared, &plan)?;
    let receipts = plan
        .tasks
        .iter()
        .map(|task| {
            read_json(&resolve_existing_artifact(
                &options.embeddings,
                &task.receipt_path,
            )?)
        })
        .collect::<rag_pipeline::Result<Vec<VectorResultReceipt>>>()?;
    result_set.validate(&plan, &receipts)?;
    let profile_bytes = fs::read(&options.embedding_profile)?;
    let profile = parse_bound_embedding_profile(
        &profile_bytes,
        plan.embedding_profile.component.sha256.as_str(),
    )?;
    validate_plan_profile_fields(&plan.embedding_profile, &profile_bytes, &profile)?;
    for receipt in &receipts {
        let path = resolve_existing_artifact(&options.embeddings, &receipt.vector.path)?;
        if file_digest(&path)? != receipt.vector.sha256 {
            return Err(Error::AccountingClosure(
                "embedding vector object digest differs",
            ));
        }
    }
    let vector_shards = plan
        .tasks
        .iter()
        .map(|task| {
            Ok(OrderedVectorShard {
                path: resolve_existing_artifact(&options.embeddings, &task.result_path)?,
                first_vector_ordinal: task.ordinal_start,
                vector_count: task.row_count(),
                dimensions: profile.dimensions,
                order_sha256: task.embedding_input_order_sha256.to_string(),
            })
        })
        .collect::<rag_pipeline::Result<Vec<_>>>()?;
    let vectors = vectors_from_embedding_shards(vector_shards)?;
    let document_paths = prepared
        .documents
        .iter()
        .map(|object| resolve_existing_artifact(&options.prepared, &object.object.path))
        .collect::<rag_pipeline::Result<Vec<_>>>()?;
    let occurrence_paths = prepared
        .occurrences
        .iter()
        .map(|object| resolve_existing_artifact(&options.prepared, &object.object.path))
        .collect::<rag_pipeline::Result<Vec<_>>>()?;
    let document_rows = documents_from_parquet_shards(document_paths);
    let occurrence_rows = occurrences_from_parquet_shards(occurrence_paths);
    let staging = AtomicDirectory::new(&options.out)?;
    let staged_index = staging.path().join("assembled-index");
    let manifest = write_fast_index_from_streams(
        &staged_index,
        SourceBinding {
            snapshot_sha256: prepared.dataset.source_snapshot.sha256.to_string(),
            mapping_sha256: prepared.dataset.mapping.sha256.to_string(),
        },
        // Fast-index v2 has no explicit dataset-scope field. Keep the output
        // partial so a dataset miss is never represented as corpus-wide.
        BuildScope::Sample,
        document_rows,
        occurrence_rows,
        vectors,
        profile,
    )?;
    write_portable_build_report(&staged_index, &prepared, &manifest)?;
    staging.publish_child("assembled-index")?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}

fn write_portable_build_report(
    out: &Path,
    prepared: &PreparedCorpusManifest,
    manifest: &FastIndexManifest,
) -> Result<()> {
    let accounting = portable_dataset_accounting(prepared, manifest.occurrences.rows)?;
    let report = json!({
        "schema_version": "livefire.rag.fast-build-report/1",
        "source": manifest.source,
        "build_scope": manifest.build_scope,
        "complete": manifest.complete,
        "document_count": manifest.documents.rows,
        "occurrence_count": manifest.occurrences.rows,
        "vector_count": manifest.vectors.count,
        "embedding_profile_sha256": manifest.embedding_profile.sha256,
        // Embedding happened in the previous portable stage. These counters
        // describe work done by assembly, so both are zero.
        "cache_hits": 0,
        "embedded": 0,
        "accounting": accounting,
    });
    write_canonical_json(&out.join("build-report.json"), &report)?;
    Ok(())
}

fn portable_dataset_accounting(
    prepared: &PreparedCorpusManifest,
    indexed_occurrences: u64,
) -> Result<Value> {
    let mut source_rows_by_relation = BTreeMap::new();
    let mut structured_only_by_relation = BTreeMap::new();
    let mut excluded_by_scope_by_relation = BTreeMap::new();
    let mut source_rows = 0_u64;
    let mut structured_only = 0_u64;
    let mut excluded_by_scope = 0_u64;
    for (relation, accounting) in &prepared.relation_accounting {
        source_rows = source_rows
            .checked_add(accounting.source_rows)
            .ok_or(Error::CountOverflow)?;
        source_rows_by_relation.insert(relation.clone(), accounting.source_rows);
        if prepared
            .dataset
            .structured_only_relations
            .contains(relation)
        {
            structured_only = structured_only
                .checked_add(accounting.source_rows)
                .ok_or(Error::CountOverflow)?;
            structured_only_by_relation.insert(relation.clone(), accounting.source_rows);
        } else if prepared.dataset.excluded_relations.contains(relation) {
            excluded_by_scope = excluded_by_scope
                .checked_add(accounting.source_rows)
                .ok_or(Error::CountOverflow)?;
            excluded_by_scope_by_relation.insert(relation.clone(), accounting.source_rows);
        } else if prepared.dataset.included_relations.contains(relation)
            && accounting.excluded_rows > 0
        {
            // Included relations can still contain malformed, unknown, or
            // deliberately structured-only rows. They were inspected during
            // preparation but do not have searchable document projections.
            structured_only = structured_only
                .checked_add(accounting.excluded_rows)
                .ok_or(Error::CountOverflow)?;
            structured_only_by_relation.insert(relation.clone(), accounting.excluded_rows);
        }
    }
    let closed = indexed_occurrences
        .checked_add(structured_only)
        .and_then(|value| value.checked_add(excluded_by_scope))
        .ok_or(Error::CountOverflow)?;
    if source_rows != closed {
        return Err(Error::AccountingClosure(
            "portable dataset source accounting does not close",
        ));
    }
    Ok(json!({
            "coverage_semantics": "dataset_scope_only_not_source_corpus_coverage",
            "semantic_source_coverage_complete": false,
            "dataset_id": prepared.dataset.id,
            "dataset_version": prepared.dataset.version,
            "source_records": source_rows,
            "source_records_by_relation": source_rows_by_relation,
            "indexed_occurrences": indexed_occurrences,
            "structured_only_occurrences": structured_only,
            "structured_only_by_relation": structured_only_by_relation,
            "excluded_by_scope_occurrences": excluded_by_scope,
            "excluded_by_scope_by_relation": excluded_by_scope_by_relation,
    }))
}

fn project_prepared_batch(
    batch: &RecordBatch,
    relation: &str,
    context: &ProjectionContext,
    source_row_ordinal: &mut u64,
    documents: &mut BTreeMap<String, DocumentAccumulator>,
    occurrences: &mut Vec<PreparedOccurrenceRow>,
) -> Result<()> {
    let event_ids = strings(batch, "event_id")?;
    let json_rows = strings(batch, "typed_event_json")?;
    let support = strings(batch, "support_ref")?;
    for row in 0..batch.num_rows() {
        let projected = project(ProjectionInput {
            relation_name: relation,
            event_id: event_ids.value(row),
            typed_event_json: json_rows.value(row),
            support_ref: support.value(row),
            context,
        })?;
        let ordinal = *source_row_ordinal;
        *source_row_ordinal = source_row_ordinal
            .checked_add(1)
            .ok_or(Error::CountOverflow)?;
        let Some(document) = projected.document else {
            continue;
        };
        let document_sha256 = sha256_bytes(&serde_json::to_vec(&document)?);
        let mut fast = fast_document(document.clone(), document_sha256.clone())?;
        fast.facets_json = canonical_string(&document.facets)?;
        if !documents.contains_key(&document.document_id)
            && documents.len() >= MAX_IN_MEMORY_DOCUMENTS
        {
            return Err(Error::AccountingClosure(
                "portable preparation in-memory document limit exceeded",
            ));
        }
        let entry = documents
            .entry(document.document_id.clone())
            .or_insert_with(|| DocumentAccumulator {
                document: fast.clone(),
                primary_relation: relation.to_owned(),
                relations: BTreeSet::from([relation.to_owned()]),
            });
        if entry.document.document_sha256 != document_sha256
            || entry.document.semantic_text != fast.semantic_text
            || entry.document.facets_json != fast.facets_json
        {
            return Err(Error::InconsistentDocument(document.document_id));
        }
        entry.document.occurrence_count = entry
            .document
            .occurrence_count
            .checked_add(1)
            .ok_or(Error::CountOverflow)?;
        entry.relations.insert(relation.to_owned());
        let event_time_ms = parse_event_time_ms(
            projected.occurrence.event_time.as_deref(),
            projected.occurrence.event_time_availability,
        )?;
        let mut occurrence_hasher = Sha256::new();
        occurrence_hasher.update(context.snapshot.sha256.as_bytes());
        occurrence_hasher.update([0]);
        occurrence_hasher.update(relation.as_bytes());
        occurrence_hasher.update([0]);
        occurrence_hasher.update(event_ids.value(row).as_bytes());
        occurrences.push(PreparedOccurrenceRow {
            occurrence_id: format!("occ-{:x}", occurrence_hasher.finalize()),
            document_id: document.document_id,
            event_time_ms,
            relation: relation.to_owned(),
            source_row_ordinal: ordinal,
            exact_attributes_json: canonical_string(&projected.occurrence.exact_attributes)?,
            snapshot_sha256: Digest::new(context.snapshot.sha256.clone())?,
            mapping_sha256: Digest::new(context.mapping_pack.sha256.clone())?,
            event_id: event_ids.value(row).to_owned(),
            support_ref: support.value(row).to_owned(),
        });
    }
    Ok(())
}

fn flush_occurrence_shards(
    root: &Path,
    relation: &str,
    buffer: &mut Vec<PreparedOccurrenceRow>,
    relation_part: &mut u64,
    objects: &mut Vec<PreparedOccurrenceObject>,
    flush_remainder: bool,
) -> Result<()> {
    while buffer.len() >= OCCURRENCE_SHARD_ROWS || flush_remainder && !buffer.is_empty() {
        let rows_to_write = buffer.len().min(OCCURRENCE_SHARD_ROWS);
        let remainder = buffer.split_off(rows_to_write);
        let rows = std::mem::replace(buffer, remainder);
        let relative = format!("occurrences/{relation}/part-{:06}.parquet", *relation_part);
        let path = root.join(&relative);
        write_prepared_occurrences(&path, &rows)?;
        objects.push(PreparedOccurrenceObject {
            object: object_entry(
                &relative,
                &path,
                rows.len() as u64,
                canonical_digest(&rows)?,
            )?,
            // The caller rewrites global ordinals after all relation shards
            // are known. Keep this checked relation-local value until then.
            ordinal: u32::try_from(*relation_part)
                .map_err(|_| Error::AccountingClosure("too many occurrence shards"))?,
            relation: relation.to_owned(),
        });
        *relation_part = relation_part.checked_add(1).ok_or(Error::CountOverflow)?;
    }
    Ok(())
}

fn load_prepared(root: &Path) -> Result<PreparedCorpusManifest> {
    let manifest: PreparedCorpusManifest = read_json(&root.join(MANIFEST_FILE))?;
    manifest.validate()?;
    verify_prepared_objects(root, &manifest)?;
    Ok(manifest)
}

fn verify_prepared_objects(root: &Path, manifest: &PreparedCorpusManifest) -> Result<()> {
    for object in manifest
        .documents
        .iter()
        .map(|value| &value.object)
        .chain(manifest.occurrences.iter().map(|value| &value.object))
    {
        let path = resolve_existing_artifact(root, &object.path)?;
        if fs::metadata(&path)?.len() != object.bytes || file_digest(&path)? != object.sha256 {
            return Err(Error::AccountingClosure("prepared object digest differs"));
        }
    }
    let mut previous_occurrence_order: Option<(String, u64)> = None;
    let mut occurrence_rows = 0_u64;
    for object in &manifest.occurrences {
        let rows =
            read_prepared_occurrences(&resolve_existing_artifact(root, &object.object.path)?)?;
        if rows.len() as u64 != object.object.rows
            || rows.iter().any(|row| row.validate().is_err())
            || rows.iter().any(|row| {
                row.relation != object.relation
                    || row.snapshot_sha256 != manifest.dataset.source_snapshot.sha256
                    || row.mapping_sha256 != manifest.dataset.mapping.sha256
            })
            || canonical_digest(&rows)? != object.object.logical_order_sha256
        {
            return Err(Error::AccountingClosure(
                "prepared occurrence object metadata differs",
            ));
        }
        for row in &rows {
            let order = (row.relation.clone(), row.source_row_ordinal);
            if previous_occurrence_order
                .as_ref()
                .is_some_and(|previous| previous >= &order)
            {
                return Err(Error::AccountingClosure(
                    "prepared occurrence object order differs",
                ));
            }
            previous_occurrence_order = Some(order);
        }
        occurrence_rows = occurrence_rows
            .checked_add(rows.len() as u64)
            .ok_or(Error::CountOverflow)?;
    }
    if occurrence_rows != manifest.occurrence_count {
        return Err(Error::AccountingClosure(
            "prepared occurrence object coverage differs",
        ));
    }
    Ok(())
}

struct TaskDocumentLoader<'a> {
    root: &'a Path,
    prepared: &'a PreparedCorpusManifest,
    cached: Option<(SafeRelativePath, Vec<PreparedDocumentRow>)>,
}

impl<'a> TaskDocumentLoader<'a> {
    fn new(root: &'a Path, prepared: &'a PreparedCorpusManifest) -> Self {
        Self {
            root,
            prepared,
            cached: None,
        }
    }

    fn load(&mut self, task: &EmbeddingTask) -> Result<Vec<PreparedDocumentRow>> {
        let capacity = usize::try_from(task.row_count())
            .map_err(|_| Error::AccountingClosure("task row count overflow"))?;
        let mut task_rows = Vec::with_capacity(capacity);
        for slice in &task.input_slices {
            let object = self
                .prepared
                .documents
                .iter()
                .find(|object| object.object.path == slice.path)
                .ok_or(Error::AccountingClosure(
                    "embedding task input object is absent",
                ))?;
            if object.object.sha256 != slice.object_sha256 {
                return Err(Error::AccountingClosure(
                    "embedding task input object digest differs",
                ));
            }
            if self
                .cached
                .as_ref()
                .is_none_or(|(path, _)| path != &slice.path)
            {
                let path = resolve_existing_artifact(self.root, &slice.path)?;
                let rows = read_prepared_documents(&path)?;
                if rows.len() as u64 != object.object.rows
                    || rows.iter().any(|row| row.validate().is_err())
                    || rows.first().map(|row| &row.document_id) != Some(&object.first_document_id)
                    || rows.last().map(|row| &row.document_id) != Some(&object.last_document_id)
                    || canonical_digest(&rows)? != object.object.logical_order_sha256
                    || embedding_input_order_digest(&rows) != object.embedding_input_order_sha256
                {
                    return Err(Error::AccountingClosure(
                        "prepared document object metadata differs",
                    ));
                }
                self.cached = Some((slice.path.clone(), rows));
            }
            let rows = &self.cached.as_ref().expect("cache was populated").1;
            let start = usize::try_from(slice.row_offset)
                .map_err(|_| Error::AccountingClosure("task slice offset overflow"))?;
            let count = usize::try_from(slice.rows)
                .map_err(|_| Error::AccountingClosure("task slice row count overflow"))?;
            let end = start
                .checked_add(count)
                .ok_or(Error::AccountingClosure("task slice range overflow"))?;
            let selected = rows.get(start..end).ok_or(Error::AccountingClosure(
                "task slice exceeds prepared object",
            ))?;
            if embedding_input_order_digest(selected) != slice.embedding_input_order_sha256 {
                return Err(Error::AccountingClosure(
                    "task slice input order digest differs",
                ));
            }
            task_rows.extend_from_slice(selected);
        }
        if task_rows.len() != capacity
            || task_rows.iter().enumerate().any(|(offset, row)| {
                row.document_ordinal != task.ordinal_start.saturating_add(offset as u64)
            })
            || embedding_input_order_digest(&task_rows) != task.embedding_input_order_sha256
        {
            return Err(Error::AccountingClosure(
                "embedding task rows differ from prepared order",
            ));
        }
        Ok(task_rows)
    }
}

fn validate_plan_streaming(
    root: &Path,
    prepared: &PreparedCorpusManifest,
    plan: &EmbeddingPlan,
) -> Result<()> {
    plan.validate()?;
    prepared.validate()?;
    if plan.prepared_corpus_sha256 != prepared.component_sha256
        || plan.dataset != prepared.dataset
        || plan.document_count != prepared.document_count
        || plan.document_order_sha256 != prepared.document_order_sha256
    {
        return Err(Error::AccountingClosure(
            "embedding plan differs from prepared corpus",
        ));
    }
    let mut document_order = Sha256::new();
    let mut embedding_order = Sha256::new();
    embedding_order.update(b"livefire.rag.embedding-input-order/1\0");
    let mut previous_document_id: Option<String> = None;
    let mut rows_seen = 0_u64;
    let mut loader = TaskDocumentLoader::new(root, prepared);
    for task in &plan.tasks {
        for row in loader.load(task)? {
            if row.document_ordinal != rows_seen
                || previous_document_id
                    .as_ref()
                    .is_some_and(|previous| previous >= &row.document_id)
            {
                return Err(Error::AccountingClosure(
                    "prepared document stream is not canonical",
                ));
            }
            document_order.update(row.document_id.as_bytes());
            document_order.update([0]);
            for field in [
                row.document_id.as_str(),
                row.document_sha256.as_str(),
                row.semantic_text_sha256.as_str(),
            ] {
                embedding_order.update(field.as_bytes());
                embedding_order.update([0]);
            }
            previous_document_id = Some(row.document_id);
            rows_seen = rows_seen.checked_add(1).ok_or(Error::CountOverflow)?;
        }
    }
    if rows_seen != prepared.document_count
        || format!("{:x}", document_order.finalize()) != prepared.document_order_sha256.as_str()
        || format!("{:x}", embedding_order.finalize())
            != prepared.embedding_input_order_sha256.as_str()
    {
        return Err(Error::AccountingClosure(
            "prepared document stream digest differs",
        ));
    }
    Ok(())
}

fn validate_embedding_artifact_coverage(root: &Path, plan: &EmbeddingPlan) -> Result<()> {
    let mut expected = BTreeSet::from([MANIFEST_FILE.to_owned()]);
    for task in &plan.tasks {
        expected.insert(task.result_path.as_str().to_owned());
        expected.insert(task.receipt_path.as_str().to_owned());
    }
    let canonical_root = fs::canonicalize(root)?;
    let mut pending = vec![canonical_root.clone()];
    let mut actual = BTreeSet::new();
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(&directory)? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            if file_type.is_symlink() {
                return Err(Error::AccountingClosure(
                    "embedding artifact tree contains a symlink",
                ));
            }
            let path = entry.path();
            if file_type.is_dir() {
                pending.push(path);
            } else if file_type.is_file() {
                let relative = path
                    .strip_prefix(&canonical_root)
                    .map_err(|_| Error::AccountingClosure("embedding artifact path escaped"))?
                    .to_str()
                    .ok_or(Error::AccountingClosure(
                        "embedding artifact path is not UTF-8",
                    ))?
                    .to_owned();
                actual.insert(relative);
            } else {
                return Err(Error::AccountingClosure(
                    "embedding artifact tree contains an unsupported entry",
                ));
            }
        }
    }
    if actual != expected {
        return Err(Error::AccountingClosure(
            "embedding artifact coverage differs from plan",
        ));
    }
    Ok(())
}

fn validate_completed_embedding_task(
    receipt_path: &Path,
    vector_path: &Path,
    task: &EmbeddingTask,
    plan: &EmbeddingPlan,
    profile: &rag_embedding::EmbeddingProfile,
    runtime: &ComponentRef,
) -> Result<Option<VectorResultReceipt>> {
    if !receipt_path.try_exists()? {
        return Ok(None);
    }
    let receipt = match read_json::<VectorResultReceipt>(receipt_path) {
        Ok(receipt) => receipt,
        Err(_) => {
            fs::remove_file(receipt_path)?;
            return Ok(None);
        }
    };
    let expected = rag_embedding::EmbeddingShardExpectation {
        row_count: task.row_count(),
        dimensions: profile.dimensions,
        order_sha256: decode_sha256_hex(task.embedding_input_order_sha256.as_str())?,
    };
    let valid = receipt.validate_against(plan).is_ok()
        && receipt.executor.returned_model == profile.model
        && &receipt.executor.runtime == runtime
        && EmbeddingShard::open_expected(vector_path, expected)
            .and_then(|shard| shard.validate_normalization(&profile.normalization))
            .is_ok()
        && file_digest(vector_path).is_ok_and(|digest| digest == receipt.vector.sha256);
    if valid {
        Ok(Some(receipt))
    } else {
        fs::remove_file(receipt_path)?;
        Ok(None)
    }
}

fn remove_stale_embedding_partials(root: &Path) -> Result<()> {
    remove_stale_atomic_writes(root)?;
    for directory in [root.join("parts"), root.join("receipts")] {
        remove_stale_atomic_writes(&directory)?;
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_str().ok_or(Error::AccountingClosure(
                "embedding partial path is not UTF-8",
            ))?;
            if name.starts_with('.') && name.ends_with(".partial") {
                if !entry.file_type()?.is_file() {
                    return Err(Error::AccountingClosure(
                        "embedding partial is not a regular file",
                    ));
                }
                fs::remove_file(entry.path())?;
            }
        }
    }
    Ok(())
}

fn profile_ref(
    bytes: &[u8],
    compact: &rag_embedding::EmbeddingProfile,
) -> Result<EmbeddingProfileRef> {
    let value: Value = serde_json::from_slice(bytes)?;
    let model = value
        .get("model_artifact_set")
        .ok_or(Error::AccountingClosure(
            "embedding model component is absent",
        ))?;
    let tokenizer = value.get("tokenizer").ok_or(Error::AccountingClosure(
        "embedding tokenizer component is absent",
    ))?;
    Ok(EmbeddingProfileRef {
        component: component(&compact.id, &compact.version, &compact.sha256)?,
        model_artifact: component_from_value(model)?,
        tokenizer: component_from_value(tokenizer)?,
        maximum_input_tokens: value
            .get("maximum_tokens")
            .and_then(Value::as_u64)
            .and_then(|number| u32::try_from(number).ok())
            .ok_or(Error::AccountingClosure(
                "embedding maximum tokens are absent",
            ))?,
        pooling: value
            .get("pooling")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_owned(),
        normalization: compact.normalization.clone(),
        dimensions: compact.dimensions,
        dtype: "f32le".into(),
        document_format: value
            .get("document_prefix")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned()
            + "{semantic_text}",
    })
}

fn validate_plan_profile_fields(
    planned: &EmbeddingProfileRef,
    bytes: &[u8],
    compact: &rag_embedding::EmbeddingProfile,
) -> Result<()> {
    if profile_ref(bytes, compact)? != *planned {
        return Err(Error::AccountingClosure(
            "embedding plan profile fields differ from profile bytes",
        ));
    }
    Ok(())
}

async fn validate_lmstudio_conformance(
    embedder: &LmStudioEmbedder,
    profile_bytes: &[u8],
    profile: &rag_embedding::EmbeddingProfile,
) -> Result<()> {
    const FIXTURES: [&[u8]; 2] = [
        include_bytes!("../../../fixtures/embedding-conformance.v1.json"),
        include_bytes!("../../../fixtures/generic-evidence-embedding-conformance.v1.json"),
    ];
    let profile_value: Value = serde_json::from_slice(profile_bytes)?;
    let conformance = profile_value
        .get("conformance")
        .and_then(Value::as_object)
        .ok_or(Error::AccountingClosure(
            "embedding conformance contract is absent",
        ))?;
    let fixture_sha256 = conformance
        .get("fixture_sha256")
        .and_then(Value::as_str)
        .ok_or(Error::AccountingClosure(
            "embedding conformance fixture digest is absent",
        ))?;
    let expected_output_sha256 = conformance
        .get("normalized_output_sha256")
        .and_then(Value::as_str)
        .ok_or(Error::AccountingClosure(
            "embedding conformance output digest is absent",
        ))?;
    let fixture_bytes = FIXTURES
        .into_iter()
        .find(|bytes| sha256_bytes(bytes) == fixture_sha256)
        .ok_or(Error::AccountingClosure(
            "embedding conformance fixture is unavailable",
        ))?;
    let fixture: Value = serde_json::from_slice(fixture_bytes)?;
    let request =
        fixture
            .get("request")
            .and_then(Value::as_object)
            .ok_or(Error::AccountingClosure(
                "embedding conformance request is absent",
            ))?;
    if request.get("model").and_then(Value::as_str) != Some(profile.model.as_str()) {
        return Err(Error::AccountingClosure(
            "embedding conformance model differs from profile",
        ));
    }
    let inputs = request
        .get("input")
        .and_then(Value::as_array)
        .ok_or(Error::AccountingClosure(
            "embedding conformance inputs are absent",
        ))?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or(Error::AccountingClosure(
                    "embedding conformance input is invalid",
                ))
        })
        .collect::<Result<Vec<_>>>()?;
    let dimensions = usize::try_from(profile.dimensions)
        .map_err(|_| Error::AccountingClosure("embedding dimensions overflow"))?;
    for _ in 0..2 {
        let (batch, normalized_sha256) = embedder.conformance_probe(&inputs).await?;
        if batch.returned_model != profile.model || batch.vectors.len() != inputs.len() {
            return Err(Error::AccountingClosure(
                "embedding conformance response identity differs",
            ));
        }
        for vector in &batch.vectors {
            validate_vector(vector, dimensions, &profile.normalization)?;
        }
        if normalized_sha256 != expected_output_sha256 {
            return Err(Error::AccountingClosure(
                "embedding conformance output digest differs",
            ));
        }
    }
    Ok(())
}

fn component_from_value(value: &Value) -> Result<ComponentRef> {
    component(
        value
            .get("id")
            .and_then(Value::as_str)
            .ok_or(Error::AccountingClosure("component ID is absent"))?,
        value
            .get("version")
            .and_then(Value::as_str)
            .ok_or(Error::AccountingClosure("component version is absent"))?,
        value
            .get("sha256")
            .and_then(Value::as_str)
            .ok_or(Error::AccountingClosure("component digest is absent"))?,
    )
}

fn component(id: &str, version: &str, sha256: &str) -> Result<ComponentRef> {
    let value = ComponentRef {
        id: id.to_owned(),
        version: version.to_owned(),
        sha256: Digest::new(sha256)?,
    };
    value.validate()?;
    Ok(value)
}

fn object_entry(
    relative: &str,
    path: &Path,
    rows: u64,
    logical_order_sha256: Digest,
) -> Result<ObjectEntry> {
    Ok(ObjectEntry {
        path: SafeRelativePath::new(relative)?,
        rows,
        bytes: fs::metadata(path)?.len(),
        sha256: file_digest(path)?,
        logical_order_sha256,
    })
}

fn file_digest(path: &Path) -> Result<Digest> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Digest::new(format!("{:x}", hasher.finalize())).map_err(Error::from)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn zero_digest() -> Result<Digest> {
    Ok(Digest::new("0".repeat(64))?)
}

fn canonical_string(value: &impl serde::Serialize) -> Result<String> {
    String::from_utf8(canonical_json_bytes(value)?)
        .map_err(|_| Error::AccountingClosure("canonical JSON is not UTF-8"))
}

fn manifest_or_file(root: &Path, name: &str) -> PathBuf {
    if root.is_dir() {
        root.join(name)
    } else {
        root.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rag_pipeline::read_prepared_occurrences;

    fn occurrence(sequence: usize) -> PreparedOccurrenceRow {
        PreparedOccurrenceRow {
            occurrence_id: format!("occ-{sequence}"),
            document_id: "doc-a".into(),
            event_time_ms: Some(sequence as u64),
            relation: "events".into(),
            source_row_ordinal: sequence as u64,
            exact_attributes_json: "{}".into(),
            snapshot_sha256: Digest::new("a".repeat(64)).unwrap(),
            mapping_sha256: Digest::new("b".repeat(64)).unwrap(),
            event_id: format!("event-{sequence}"),
            support_ref: format!("support-{sequence}"),
        }
    }

    #[test]
    fn occurrence_preparation_flushes_fixed_bounded_parts_and_remainder() {
        let directory = tempfile::tempdir().unwrap();
        let mut buffer = (0..OCCURRENCE_SHARD_ROWS + 3)
            .map(occurrence)
            .collect::<Vec<_>>();
        let mut part = 0;
        let mut objects = Vec::new();
        flush_occurrence_shards(
            directory.path(),
            "events",
            &mut buffer,
            &mut part,
            &mut objects,
            false,
        )
        .unwrap();
        assert_eq!(buffer.len(), 3);
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].object.rows, OCCURRENCE_SHARD_ROWS as u64);
        flush_occurrence_shards(
            directory.path(),
            "events",
            &mut buffer,
            &mut part,
            &mut objects,
            true,
        )
        .unwrap();
        assert!(buffer.is_empty());
        assert_eq!(objects.len(), 2);
        assert_eq!(objects[1].object.rows, 3);
        assert_eq!(
            read_prepared_occurrences(&objects[1].object.path.join_to(directory.path()))
                .unwrap()
                .len(),
            3
        );
    }

    #[test]
    fn dataset_accounting_includes_nonsearchable_rows_from_included_relations() {
        let component = |id: &str, byte: char| ComponentRef {
            id: id.into(),
            version: "1".into(),
            sha256: Digest::new(byte.to_string().repeat(64)).unwrap(),
        };
        let prepared = PreparedCorpusManifest {
            schema_version: PREPARED_CORPUS_SCHEMA.into(),
            component_sha256: Digest::new("0".repeat(64)).unwrap(),
            dataset: DatasetIdentity {
                id: "mixed-events".into(),
                version: "1".into(),
                source_snapshot: component("snapshot", 'a'),
                mapping: component("mapping", 'b'),
                included_relations: vec!["events".into()],
                excluded_relations: vec!["network".into()],
                structured_only_relations: vec!["metrics".into()],
            },
            projection_policy: component("projection", 'c'),
            document_schema: component("documents", 'd'),
            occurrence_schema: component("occurrences", 'e'),
            preparation_implementation: component("implementation", 'f'),
            document_count: 1,
            occurrence_count: 1,
            document_order_sha256: Digest::new("1".repeat(64)).unwrap(),
            embedding_input_order_sha256: Digest::new("2".repeat(64)).unwrap(),
            documents: vec![],
            occurrences: vec![],
            relation_accounting: BTreeMap::from([
                (
                    "events".into(),
                    RelationAccounting {
                        source_rows: 2,
                        searchable_occurrences: 1,
                        selected_occurrences: 1,
                        excluded_rows: 1,
                    },
                ),
                (
                    "metrics".into(),
                    RelationAccounting {
                        source_rows: 3,
                        searchable_occurrences: 0,
                        selected_occurrences: 0,
                        excluded_rows: 3,
                    },
                ),
                (
                    "network".into(),
                    RelationAccounting {
                        source_rows: 4,
                        searchable_occurrences: 0,
                        selected_occurrences: 0,
                        excluded_rows: 4,
                    },
                ),
            ]),
        };
        let accounting = portable_dataset_accounting(&prepared, 1).unwrap();
        assert_eq!(accounting["source_records"], 9);
        assert_eq!(accounting["indexed_occurrences"], 1);
        assert_eq!(accounting["structured_only_occurrences"], 4);
        assert_eq!(accounting["structured_only_by_relation"]["events"], 1);
        assert_eq!(accounting["structured_only_by_relation"]["metrics"], 3);
        assert_eq!(accounting["excluded_by_scope_occurrences"], 4);
    }
}

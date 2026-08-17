//! Read-only streaming adapter for released `livefire-ocsf` snapshots.
//!
//! The adapter consumes only the public on-disk receipt and Parquet format. It
//! deliberately has no source or path dependency on `livefire-ocsf`.

#![forbid(unsafe_code)]

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
    path::{Component, Path, PathBuf},
    sync::Arc,
};

use arrow_array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int64Array, ListArray, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Schema};
use parquet::{
    arrow::{
        ProjectionMask,
        arrow_reader::{ArrowReaderMetadata, ParquetRecordBatchReaderBuilder},
    },
    file::metadata::ParquetMetaData,
};
use rag_contracts::Sha256;
use serde::Deserialize;
use sha2::{Digest, Sha256 as Sha256Hasher};
use thiserror::Error;

const RECEIPT_FILE: &str = "build-receipt.json";
const REQUIRED_EVENT_COLUMNS_V1: [&str; 3] = ["event_id", "typed_event_json", "support_ref"];
const REQUIRED_EVENT_COLUMNS_V2: [&str; 3] = ["event_id", "empty_object_paths", "support_ref"];
const LOGICAL_EVENT_COLUMNS: [&str; 3] = ["event_id", "typed_event_json", "support_ref"];
const REQUIRED_CORE_RELATIONS: [&str; 7] = [
    "events",
    "event_facets",
    "entities",
    "observables",
    "participants",
    "event_observables",
    "relationships",
];
const SCHEMA_V3_GRAPH_RELATIONS: [&str; 8] = [
    "events",
    "event_facets",
    "entities",
    "observables",
    "participants",
    "event_observables",
    "relationships",
    "subject_aliases",
];
const SCHEMA_V3_TYPED_RELATIONS: [&str; 19] = [
    "ocsf_account_change",
    "ocsf_api_activity",
    "ocsf_application_lifecycle",
    "ocsf_authentication",
    "ocsf_cloud_resources_inventory_info",
    "ocsf_datastore_activity",
    "ocsf_detection_finding",
    "ocsf_dns_activity",
    "ocsf_email_activity",
    "ocsf_entity_management",
    "ocsf_event_log_activity",
    "ocsf_ext_livefire_configuration_snapshot",
    "ocsf_ext_livefire_system_metric",
    "ocsf_file_activity",
    "ocsf_http_activity",
    "ocsf_inventory_info",
    "ocsf_network_activity",
    "ocsf_process_activity",
    "ocsf_user_inventory",
];

/// The identities and discovered relation objects bound by one snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OcsfSnapshot {
    pub schema_version: u8,
    pub snapshot_id: String,
    pub snapshot_version: String,
    pub snapshot_sha256: Sha256,
    pub dataset_sha256: Sha256,
    pub mapping_id: String,
    pub mapping_version: String,
    pub mapping_sha256: Sha256,
    pub ocsf_schema_sha256: Sha256,
    pub extension_pack_sha256: Sha256,
    pub relation_contract_id: String,
    pub relation_contract_version: String,
    pub relation_contract_sha256: Sha256,
    /// Exact M45 query-capability sidecar, when the snapshot manifest requires it.
    pub snapshot_capabilities_sha256: Option<Sha256>,
    pub normalized_events: u64,
    pub relations: Vec<OcsfRelation>,
}

/// A content-bound Parquet relation. `path` always remains relative to the
/// snapshot root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OcsfRelation {
    pub name: String,
    pub kind: RelationKind,
    pub path: PathBuf,
    pub rows: u64,
    pub object_sha256: Sha256,
    pub logical_sha256: Sha256,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelationKind {
    /// One of the class-specific `ocsf_*` semantic event relations.
    TypedSemantic,
    /// A runner-facing semantic graph relation such as `events` or `entities`.
    SemanticGraph,
    /// An authority-only relation. It is retained in identity metadata but is
    /// never returned by [`SnapshotReader::typed_relations`].
    Authority,
}

/// An owning, fallible stream of Arrow record batches.
pub type RecordBatchStream =
    Box<dyn Iterator<Item = Result<RecordBatch, OcsfError>> + Send + 'static>;

/// One physical Parquet row group in admitted file order.
///
/// `first_row` is the zero-based relation row ordinal of this group's first
/// row. The next group always starts at `first_row + rows`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OcsfRowGroup {
    pub ordinal: usize,
    pub first_row: u64,
    pub rows: u64,
    pub compressed_bytes: u64,
}

#[derive(Debug, Clone)]
struct CachedParquetObject {
    path: PathBuf,
    metadata: ArrowReaderMetadata,
    row_groups: Vec<OcsfRowGroup>,
    typed_encoding: Option<TypedEncoding>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TypedEncoding {
    JsonV1,
    ColumnsV2,
}

/// A content-admitted Parquet object whose digest and footer were validated
/// once before any row-group readers are created.
///
/// Each scan opens an independent file descriptor and reuses the cached footer.
/// Snapshot objects must therefore remain immutable for the lifetime of this
/// handle, as required by the snapshot boundary.
#[derive(Debug, Clone)]
pub struct AdmittedParquetObject {
    relation: OcsfRelation,
    object: CachedParquetObject,
    batch_size: usize,
    #[cfg(test)]
    digest_validations: usize,
}

impl AdmittedParquetObject {
    pub fn relation(&self) -> &OcsfRelation {
        &self.relation
    }

    pub fn schema(&self) -> &Schema {
        self.object.metadata.schema().as_ref()
    }

    pub fn row_groups(&self) -> &[OcsfRowGroup] {
        &self.object.row_groups
    }

    #[cfg(test)]
    fn digest_validation_count(&self) -> usize {
        self.digest_validations
    }

    /// Scan all row groups while reading only the named root columns.
    pub fn scan_projected(
        &self,
        required_columns: &[&str],
    ) -> Result<RecordBatchStream, OcsfError> {
        self.open_reader(None, Some(required_columns))
    }

    /// Scan one row group while reading only the named root columns.
    pub fn scan_row_group(
        &self,
        row_group_ordinal: usize,
        required_columns: &[&str],
    ) -> Result<RecordBatchStream, OcsfError> {
        if row_group_ordinal >= self.object.row_groups.len() {
            return Err(OcsfError::UnknownRowGroup {
                relation: self.relation.name.clone(),
                ordinal: row_group_ordinal,
            });
        }
        self.open_reader(Some(vec![row_group_ordinal]), Some(required_columns))
    }

    fn scan_all_columns(&self) -> Result<RecordBatchStream, OcsfError> {
        match self.object.typed_encoding {
            Some(_) => self.open_reader(None, Some(&LOGICAL_EVENT_COLUMNS)),
            None => self.open_reader(None, None),
        }
    }

    fn open_reader(
        &self,
        row_groups: Option<Vec<usize>>,
        required_columns: Option<&[&str]>,
    ) -> Result<RecordBatchStream, OcsfError> {
        let typed_columns = self.object.typed_encoding == Some(TypedEncoding::ColumnsV2);
        if typed_columns {
            let required_columns = required_columns.ok_or(OcsfError::InvalidProjection(
                "typed relations require logical columns",
            ))?;
            for column in required_columns {
                if !LOGICAL_EVENT_COLUMNS.contains(column) {
                    return Err(OcsfError::ProjectionColumn {
                        relation: self.relation.name.clone(),
                        column: (*column).to_owned(),
                    });
                }
            }
        }

        let mut builder = ParquetRecordBatchReaderBuilder::new_with_metadata(
            File::open(&self.object.path)?,
            self.object.metadata.clone(),
        )
        .with_batch_size(self.batch_size);
        if let Some(row_groups) = row_groups {
            builder = builder.with_row_groups(row_groups);
        }
        if let Some(required_columns) = required_columns {
            if required_columns.is_empty() {
                return Err(OcsfError::InvalidProjection(
                    "at least one required column must be named",
                ));
            }
            let schema = self.object.metadata.schema();
            let mut roots = BTreeSet::new();
            let physical_columns: Vec<&str> =
                if typed_columns && required_columns.contains(&"typed_event_json") {
                    schema
                        .fields()
                        .iter()
                        .map(|field| field.name().as_str())
                        .collect()
                } else {
                    required_columns.to_vec()
                };
            for column in physical_columns {
                let ordinal = schema
                    .index_of(column)
                    .map_err(|_| OcsfError::ProjectionColumn {
                        relation: self.relation.name.clone(),
                        column: (*column).to_owned(),
                    })?;
                roots.insert(ordinal);
            }
            builder = builder.with_projection(ProjectionMask::roots(
                self.object.metadata.parquet_schema(),
                roots,
            ));
        }
        let reader = builder.build()?;
        if typed_columns {
            let requested = required_columns
                .expect("typed-column projections were checked")
                .iter()
                .map(|column| (*column).to_owned())
                .collect::<Vec<_>>();
            Ok(Box::new(reader.map(move |batch| {
                logical_typed_batch(&batch?, &requested)
            })))
        } else {
            Ok(Box::new(reader.map(|batch| batch.map_err(OcsfError::from))))
        }
    }
}

/// Read-only boundary used by builders. Implementations validate inexpensive
/// metadata when opened and stream rows without staging JSONL.
pub trait SnapshotReader {
    fn identity(&self) -> &OcsfSnapshot;

    fn typed_relations(&self) -> impl Iterator<Item = &OcsfRelation> {
        self.identity()
            .relations
            .iter()
            .filter(|relation| relation.kind == RelationKind::TypedSemantic)
    }

    fn scan(&self, relation: &OcsfRelation) -> Result<RecordBatchStream, OcsfError>;
}

/// Adapter for released `livefire-ocsf` local snapshots through manifest version 3.
#[derive(Debug)]
pub struct LocalSnapshotReader {
    identity: OcsfSnapshot,
    objects: BTreeMap<String, CachedParquetObject>,
    batch_size: usize,
}

impl LocalSnapshotReader {
    /// Open a snapshot and perform fast admission: receipt/digest shape, safe
    /// paths, object uniqueness, Parquet metadata row counts, and typed-column
    /// contracts. Multi-gigabyte object content is not rehashed.
    pub fn open(root: impl AsRef<Path>) -> Result<Self, OcsfError> {
        Self::open_with_batch_size(root, 8192)
    }

    /// Open with an explicit Arrow batch size.
    pub fn open_with_batch_size(
        root: impl AsRef<Path>,
        batch_size: usize,
    ) -> Result<Self, OcsfError> {
        if batch_size == 0 {
            return Err(OcsfError::InvalidReceipt("batch size must be positive"));
        }
        let root = root.as_ref().canonicalize()?;
        if !root.is_dir() {
            return Err(OcsfError::InvalidReceipt(
                "snapshot root must be a directory",
            ));
        }
        let receipt: BuildReceiptView =
            serde_json::from_slice(&std::fs::read(root.join(RECEIPT_FILE))?)?;
        let receipt_schema_version = receipt.schema_version;
        let BuildReceiptView {
            snapshot_manifest: manifest,
            runnable_snapshot,
            closure,
            completeness_receipt,
            output_logical_sha256,
            snapshot_capabilities,
            ..
        } = receipt;
        if !matches!(receipt_schema_version, 1 | 2) {
            return Err(OcsfError::UnsupportedSchema(receipt_schema_version));
        }
        let manifest_schema_version = manifest.schema_version;
        if !matches!(
            (receipt_schema_version, manifest_schema_version),
            (1, 1) | (2, 2 | 3)
        ) {
            return Err(OcsfError::UnsupportedSchema(manifest.schema_version));
        }

        let snapshot_sha256 = Sha256::new(manifest.logical_sha256)?;
        let dataset_sha256 = Sha256::new(manifest.dataset_sha256)?;
        let mapping_sha256 = Sha256::new(manifest.mapping_pack_sha256)?;
        let ocsf_schema_sha256 = Sha256::new(manifest.ocsf_schema_sha256)?;
        let extension_pack_sha256 = Sha256::new(manifest.extension_pack_sha256)?;
        let relation_contract_sha256 = Sha256::new(manifest.relation_contract_sha256)?;
        let output_logical_sha256 = Sha256::new(output_logical_sha256)?;
        let snapshot_capabilities_sha256 = validate_snapshot_capabilities(
            &root,
            manifest_schema_version,
            &snapshot_sha256,
            runnable_snapshot.source_rows,
            runnable_snapshot.normalized_events,
            snapshot_capabilities,
        )?;
        validate_component(&runnable_snapshot.component)?;
        validate_component(&runnable_snapshot.mapping_pack)?;
        validate_component(&runnable_snapshot.relation_contract)?;
        let runnable_snapshot_sha256 = Sha256::new(runnable_snapshot.component.sha256.clone())?;
        let runnable_dataset_sha256 = Sha256::new(runnable_snapshot.dataset_sha256)?;
        let runnable_mapping_sha256 = Sha256::new(runnable_snapshot.mapping_pack.sha256.clone())?;
        let runnable_relation_sha256 = Sha256::new(runnable_snapshot.relation_contract.sha256)?;
        let completeness_snapshot_sha256 =
            Sha256::new(completeness_receipt.normalized_snapshot_sha256)?;
        let completeness_dataset_sha256 = Sha256::new(completeness_receipt.dataset_sha256)?;
        let completeness_mapping_sha256 = Sha256::new(completeness_receipt.mapping_pack_sha256)?;
        let completeness_relation_sha256 =
            Sha256::new(completeness_receipt.relation_contract_sha256)?;
        if snapshot_sha256 != output_logical_sha256
            || snapshot_sha256 != runnable_snapshot_sha256
            || snapshot_sha256 != completeness_snapshot_sha256
            || dataset_sha256 != runnable_dataset_sha256
            || dataset_sha256 != completeness_dataset_sha256
            || mapping_sha256 != runnable_mapping_sha256
            || mapping_sha256 != completeness_mapping_sha256
            || relation_contract_sha256 != runnable_relation_sha256
            || relation_contract_sha256 != completeness_relation_sha256
            || runnable_snapshot.source_rows != closure.input_rows
            || closure.input_rows != closure.mapped_source_records
            || closure.input_rows != completeness_receipt.metrics.source_rows
            || closure.mapped_source_records != completeness_receipt.metrics.mapped_source_records
            || closure.rejected_malformed_records != 0
            || closure.unsupported_records != 0
            || closure.unresolved_provenance_fields != 0
            || closure.provenance_digest_mismatches != 0
            || completeness_receipt.metrics.rejected_malformed_records != 0
        {
            return Err(OcsfError::InvalidReceipt(
                "snapshot receipt identity closure",
            ));
        }

        let mut seen_names = BTreeSet::new();
        let mut seen_paths = BTreeSet::new();
        let mut normalized_events = None;
        let mut relations = Vec::with_capacity(manifest.objects.len());
        let mut objects = BTreeMap::new();
        let mut typed_rows = 0_u64;
        for object in manifest.objects {
            if !valid_relation_name(&object.relation) || !seen_names.insert(object.relation.clone())
            {
                return Err(OcsfError::InvalidReceipt(
                    "relation names must be lowercase identifiers and unique",
                ));
            }
            let relative_path = safe_relative_path(&object.path)?;
            if !seen_paths.insert(relative_path.clone()) {
                return Err(OcsfError::InvalidReceipt("relation paths must be unique"));
            }
            let kind =
                classify_relation(manifest_schema_version, &object.relation, &relative_path)?;
            let object_path = resolve_beneath(&root, &relative_path)?;
            let file = File::open(&object_path)?;
            let metadata = ArrowReaderMetadata::load(&file, Default::default())?;
            validate_row_count(&object.relation, object.rows, metadata.metadata())?;
            let typed_encoding = if kind == RelationKind::TypedSemantic {
                let encoding = if manifest_schema_version == 1 {
                    TypedEncoding::JsonV1
                } else {
                    TypedEncoding::ColumnsV2
                };
                validate_typed_schema(&object.relation, metadata.schema().as_ref(), encoding)?;
                typed_rows = typed_rows
                    .checked_add(object.rows)
                    .ok_or(OcsfError::InvalidReceipt("typed row count overflow"))?;
                Some(encoding)
            } else {
                None
            };
            let row_groups =
                validate_row_groups(&object.relation, object.rows, metadata.metadata())?;
            if object.relation == "events" {
                normalized_events = Some(object.rows);
            }
            objects.insert(
                object.relation.clone(),
                CachedParquetObject {
                    path: object_path,
                    metadata,
                    row_groups,
                    typed_encoding,
                },
            );
            relations.push(OcsfRelation {
                name: object.relation,
                kind,
                path: relative_path,
                rows: object.rows,
                object_sha256: Sha256::new(object.sha256)?,
                logical_sha256: Sha256::new(object.logical_sha256)?,
            });
        }

        for required in REQUIRED_CORE_RELATIONS {
            if !seen_names.contains(required) {
                return Err(OcsfError::MissingRelation(required));
            }
        }
        if manifest_schema_version == 3 && !seen_names.contains("subject_aliases") {
            return Err(OcsfError::MissingRelation("subject_aliases"));
        }
        if manifest_schema_version == 3 && !seen_names.contains("field_provenance") {
            return Err(OcsfError::MissingRelation("field_provenance"));
        }
        if manifest_schema_version == 3 {
            for required in SCHEMA_V3_TYPED_RELATIONS {
                if !seen_names.contains(required) {
                    return Err(OcsfError::MissingRelation(required));
                }
            }
        }
        let normalized_events = normalized_events.expect("required events relation was checked");
        if typed_rows != normalized_events
            || runnable_snapshot.normalized_events != normalized_events
            || closure.event_rows != normalized_events
            || closure.mapped_events != normalized_events
            || completeness_receipt.metrics.normalized_events != normalized_events
        {
            return Err(OcsfError::InvalidReceipt("normalized event count closure"));
        }
        Ok(Self {
            identity: OcsfSnapshot {
                schema_version: manifest.schema_version,
                snapshot_id: runnable_snapshot.component.id,
                snapshot_version: runnable_snapshot.component.version,
                snapshot_sha256,
                dataset_sha256,
                mapping_id: runnable_snapshot.mapping_pack.id,
                mapping_version: runnable_snapshot.mapping_pack.version,
                mapping_sha256,
                ocsf_schema_sha256,
                extension_pack_sha256,
                relation_contract_id: runnable_snapshot.relation_contract.id,
                relation_contract_version: runnable_snapshot.relation_contract.version,
                relation_contract_sha256,
                snapshot_capabilities_sha256,
                normalized_events,
                relations,
            },
            objects,
            batch_size,
        })
    }

    /// Content-admit one receipt-bound Parquet object for independent projected
    /// row-group reads. Its full object digest is checked exactly once here;
    /// the already validated footer is reused by every reader from the handle.
    pub fn admit_object(
        &self,
        relation: &OcsfRelation,
    ) -> Result<AdmittedParquetObject, OcsfError> {
        let admitted = self.bound_relation(relation)?;
        let object = self
            .objects
            .get(&admitted.name)
            .expect("every admitted relation has cached object metadata");
        verify_object_digest(&object.path, &admitted.object_sha256)?;
        Ok(AdmittedParquetObject {
            relation: admitted.clone(),
            object: object.clone(),
            batch_size: self.batch_size,
            #[cfg(test)]
            digest_validations: 1,
        })
    }

    fn bound_relation<'a>(
        &'a self,
        relation: &OcsfRelation,
    ) -> Result<&'a OcsfRelation, OcsfError> {
        let admitted = self
            .identity
            .relations
            .iter()
            .find(|candidate| candidate.name == relation.name)
            .ok_or_else(|| OcsfError::UnknownRelation(relation.name.clone()))?;
        if admitted != relation {
            return Err(OcsfError::RelationBinding(relation.name.clone()));
        }
        Ok(admitted)
    }
}

impl SnapshotReader for LocalSnapshotReader {
    fn identity(&self) -> &OcsfSnapshot {
        &self.identity
    }

    fn scan(&self, relation: &OcsfRelation) -> Result<RecordBatchStream, OcsfError> {
        self.admit_object(relation)?.scan_all_columns()
    }
}

fn verify_object_digest(path: &Path, expected: &Sha256) -> Result<(), OcsfError> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256Hasher::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    if format!("{:x}", hasher.finalize()) != expected.as_str() {
        return Err(OcsfError::ObjectDigest(path.display().to_string()));
    }
    Ok(())
}

fn validate_snapshot_capabilities(
    root: &Path,
    manifest_schema_version: u8,
    snapshot_sha256: &Sha256,
    source_rows: u64,
    normalized_events: u64,
    reference: Option<SnapshotCapabilitiesObjectView>,
) -> Result<Option<Sha256>, OcsfError> {
    match (manifest_schema_version, reference) {
        (1 | 2, None) => Ok(None),
        (1 | 2, Some(_)) => Err(OcsfError::InvalidReceipt(
            "legacy snapshot unexpectedly declares capabilities",
        )),
        (3, Some(reference)) => {
            if reference.path != "capabilities/snapshot-capabilities.v1.json" {
                return Err(OcsfError::InvalidReceipt(
                    "snapshot capability path is not canonical",
                ));
            }
            let digest = Sha256::new(reference.sha256)?;
            let relative = safe_relative_path(&reference.path)?;
            let path = resolve_beneath(root, &relative)?;
            verify_object_digest(&path, &digest)?;
            let capability: SnapshotCapabilitiesView =
                serde_json::from_slice(&std::fs::read(path)?)?;
            if capability.schema_version != 1
                || Sha256::new(capability.snapshot_logical_sha256)? != *snapshot_sha256
                || capability.mapping_completeness.state != "complete"
                || capability.mapping_completeness.source_rows != source_rows
                || capability.mapping_completeness.normalized_events != normalized_events
                || !capability.mapping_completeness.has_zero_gaps()
            {
                return Err(OcsfError::InvalidReceipt(
                    "snapshot capability identity closure",
                ));
            }
            Ok(Some(digest))
        }
        (3, None) => Err(OcsfError::InvalidReceipt(
            "manifest version 3 requires snapshot capabilities",
        )),
        (version, _) => Err(OcsfError::UnsupportedSchema(version)),
    }
}

fn safe_relative_path(value: &str) -> Result<PathBuf, OcsfError> {
    if value.is_empty() {
        return Err(OcsfError::UnsafePath(value.to_owned()));
    }
    let path = Path::new(value);
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(OcsfError::UnsafePath(value.to_owned()));
    }
    Ok(path.to_path_buf())
}

fn valid_relation_name(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
}

fn resolve_beneath(root: &Path, relative: &Path) -> Result<PathBuf, OcsfError> {
    let path = root.join(relative);
    let canonical = path.canonicalize()?;
    if !canonical.starts_with(root) || !canonical.is_file() {
        return Err(OcsfError::UnsafePath(relative.display().to_string()));
    }
    Ok(canonical)
}

fn classify_relation(
    schema_version: u8,
    name: &str,
    path: &Path,
) -> Result<RelationKind, OcsfError> {
    let directory = match (schema_version, name) {
        (1, "field_provenance") => "authority",
        (1, _) => "semantic",
        (2 | 3, "field_provenance") => "provenance",
        (2, name) if name.starts_with("ocsf_") => "normalized",
        (3, name) if SCHEMA_V3_TYPED_RELATIONS.contains(&name) => "normalized",
        (2, _) => "graph",
        (3, name) if SCHEMA_V3_GRAPH_RELATIONS.contains(&name) => "graph",
        (3, _) => {
            return Err(OcsfError::InvalidReceipt(
                "manifest version 3 declares an unknown relation",
            ));
        }
        _ => return Err(OcsfError::UnsupportedSchema(schema_version)),
    };
    let expected = PathBuf::from(directory).join(format!("{name}.parquet"));
    if path != expected {
        return Err(OcsfError::UnexpectedRelationPath {
            relation: name.to_owned(),
            expected,
            actual: path.to_path_buf(),
        });
    }
    if name == "field_provenance" {
        Ok(RelationKind::Authority)
    } else if name.starts_with("ocsf_") {
        Ok(RelationKind::TypedSemantic)
    } else {
        Ok(RelationKind::SemanticGraph)
    }
}

fn validate_row_count(
    relation: &str,
    expected: u64,
    metadata: &ParquetMetaData,
) -> Result<(), OcsfError> {
    let actual = u64::try_from(metadata.file_metadata().num_rows()).map_err(|_| {
        OcsfError::InvalidParquet {
            relation: relation.to_owned(),
            reason: "negative row count",
        }
    })?;
    if actual != expected {
        return Err(OcsfError::RowCount {
            relation: relation.to_owned(),
            expected,
            actual,
        });
    }
    Ok(())
}

fn validate_row_groups(
    relation: &str,
    expected_rows: u64,
    metadata: &ParquetMetaData,
) -> Result<Vec<OcsfRowGroup>, OcsfError> {
    let mut first_row = 0_u64;
    let mut groups = Vec::with_capacity(metadata.num_row_groups());
    for (ordinal, group) in metadata.row_groups().iter().enumerate() {
        let rows = u64::try_from(group.num_rows()).map_err(|_| OcsfError::InvalidParquet {
            relation: relation.to_owned(),
            reason: "negative row-group row count",
        })?;
        if rows == 0 {
            return Err(OcsfError::InvalidParquet {
                relation: relation.to_owned(),
                reason: "empty row group",
            });
        }
        let compressed_bytes =
            u64::try_from(group.compressed_size()).map_err(|_| OcsfError::InvalidParquet {
                relation: relation.to_owned(),
                reason: "negative row-group compressed size",
            })?;
        groups.push(OcsfRowGroup {
            ordinal,
            first_row,
            rows,
            compressed_bytes,
        });
        first_row = first_row
            .checked_add(rows)
            .ok_or(OcsfError::InvalidParquet {
                relation: relation.to_owned(),
                reason: "row-group row count overflow",
            })?;
    }
    if first_row != expected_rows {
        return Err(OcsfError::InvalidParquet {
            relation: relation.to_owned(),
            reason: "row-group coverage differs from file row count",
        });
    }
    Ok(groups)
}

fn validate_typed_schema(
    relation: &str,
    schema: &arrow_schema::Schema,
    encoding: TypedEncoding,
) -> Result<(), OcsfError> {
    let required = match encoding {
        TypedEncoding::JsonV1 => &REQUIRED_EVENT_COLUMNS_V1,
        TypedEncoding::ColumnsV2 => &REQUIRED_EVENT_COLUMNS_V2,
    };
    for name in required {
        let field = schema
            .field_with_name(name)
            .map_err(|_| OcsfError::RequiredColumn {
                relation: relation.to_owned(),
                column: name,
            })?;
        let nullable = encoding == TypedEncoding::ColumnsV2 && *name == "empty_object_paths";
        if field.data_type() != &DataType::Utf8 || field.is_nullable() != nullable {
            return Err(OcsfError::RequiredColumn {
                relation: relation.to_owned(),
                column: name,
            });
        }
    }
    if encoding == TypedEncoding::ColumnsV2
        && (schema.field_with_name("typed_event_json").is_ok()
            || !schema
                .fields()
                .iter()
                .any(|field| field.name().starts_with("event.")))
    {
        return Err(OcsfError::InvalidParquet {
            relation: relation.to_owned(),
            reason: "schema-version-2 typed relation layout",
        });
    }
    Ok(())
}

/// Turn one schema-version-2 flattened typed batch back into the three-column
/// logical row shape used by the existing RAG projection code. This keeps the
/// snapshot format change inside the adapter instead of teaching every caller
/// about physical Arrow columns.
fn logical_typed_batch(
    batch: &RecordBatch,
    requested: &[String],
) -> Result<RecordBatch, OcsfError> {
    let mut event_json = None;
    if requested.iter().any(|name| name == "typed_event_json") {
        let empty_objects = batch
            .column_by_name("empty_object_paths")
            .and_then(|array| array.as_any().downcast_ref::<StringArray>())
            .ok_or(OcsfError::InvalidParquet {
                relation: "typed relation".to_owned(),
                reason: "missing empty-object paths",
            })?;
        let mut rows = Vec::with_capacity(batch.num_rows());
        for row in 0..batch.num_rows() {
            let mut root = serde_json::Map::new();
            for (field, array) in batch.schema().fields().iter().zip(batch.columns()) {
                let Some(path) = field.name().strip_prefix("event.") else {
                    continue;
                };
                if let Some(value) =
                    typed_column_value(field.name(), field.data_type(), array, row)?
                {
                    insert_json_path(&mut root, path, value)?;
                }
            }
            if !empty_objects.is_null(row) {
                let paths: Vec<String> = serde_json::from_str(empty_objects.value(row))?;
                for path in paths {
                    insert_json_path(
                        &mut root,
                        &path,
                        serde_json::Value::Object(Default::default()),
                    )?;
                }
            }
            rows.push(serde_json::to_string(&serde_json::Value::Object(root))?);
        }
        event_json = Some(Arc::new(StringArray::from(rows)) as ArrayRef);
    }

    let mut fields = Vec::with_capacity(requested.len());
    let mut arrays = Vec::with_capacity(requested.len());
    for name in LOGICAL_EVENT_COLUMNS {
        if !requested.iter().any(|requested| requested == name) {
            continue;
        }
        fields.push(arrow_schema::Field::new(name, DataType::Utf8, false));
        let array = if name == "typed_event_json" {
            event_json
                .as_ref()
                .expect("typed JSON was requested")
                .clone()
        } else {
            batch
                .column_by_name(name)
                .ok_or_else(|| OcsfError::ProjectionColumn {
                    relation: "typed relation".to_owned(),
                    column: name.to_owned(),
                })?
                .clone()
        };
        arrays.push(array);
    }
    Ok(RecordBatch::try_new(
        std::sync::Arc::new(Schema::new(fields)),
        arrays,
    )?)
}

fn typed_column_value(
    path: &str,
    data_type: &DataType,
    array: &ArrayRef,
    row: usize,
) -> Result<Option<serde_json::Value>, OcsfError> {
    if array.is_null(row) {
        return Ok(None);
    }
    let invalid = || OcsfError::InvalidParquet {
        relation: "typed relation".to_owned(),
        reason: "unsupported schema-version-2 typed column",
    };
    let value = match data_type {
        DataType::Utf8 => {
            let array = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(invalid)?;
            let text = array.value(row);
            if residual_json_column(path) {
                serde_json::from_str(text).map_err(|source| OcsfError::TypedJson {
                    path: path.to_owned(),
                    row,
                    bytes: text.len(),
                    source,
                })?
            } else {
                serde_json::Value::String(text.to_owned())
            }
        }
        DataType::Int64 => serde_json::Value::from(
            array
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(invalid)?
                .value(row),
        ),
        DataType::Float64 => serde_json::Value::Number(
            serde_json::Number::from_f64(
                array
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(invalid)?
                    .value(row),
            )
            .ok_or_else(invalid)?,
        ),
        DataType::Boolean => serde_json::Value::Bool(
            array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(invalid)?
                .value(row),
        ),
        DataType::List(item) => {
            let list = array
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(invalid)?;
            let values = list.value(row);
            let mut items = Vec::with_capacity(values.len());
            for index in 0..values.len() {
                if values.is_null(index) {
                    return Err(invalid());
                }
                items.push(
                    typed_column_value(path, item.data_type(), &values, index)?
                        .ok_or_else(invalid)?,
                );
            }
            serde_json::Value::Array(items)
        }
        _ => return Err(invalid()),
    };
    Ok(Some(value))
}

/// Schema v2 stores genuinely open objects and object arrays as canonical JSON
/// strings. These are the residual fields used by the normalization envelope;
/// `unmapped` is especially important because it retains source command data.
fn residual_json_column(path: &str) -> bool {
    if matches!(
        path,
        "event.authorization_grants"
            | "event.header"
            | "event.semantics"
            | "event.state_transitions"
            | "event.workload_endpoints"
    ) {
        return true;
    }
    if !path.starts_with("event.ocsf.")
        && !path.starts_with("event.databucket.")
        && !path.starts_with("event.resource_details.")
        && !path.starts_with("event.process.")
        && !path.starts_with("event.src_endpoint.")
        && !path.starts_with("event.dst_endpoint.")
    {
        return false;
    }
    matches!(
        path.rsplit('.').next(),
        Some(
            "agent_list"
                | "ancestry"
                | "anomaly_analyses"
                | "answers"
                | "attacks"
                | "auth_factors"
                | "authorizations"
                | "cis_controls"
                | "containers"
                | "data"
                | "data_classifications"
                | "discovery_details"
                | "edges"
                | "endpoint_connections"
                | "enrichments"
                | "environment_variables"
                | "evidences"
                | "extension_list"
                | "extensions"
                | "files"
                | "fingerprints"
                | "gpu_info_list"
                | "groups"
                | "hashes"
                | "http_cookies"
                | "http_headers"
                | "ja4_fingerprint_list"
                | "kb_article_list"
                | "kill_chain"
                | "loggers"
                | "malware"
                | "manager"
                | "metrics"
                | "network_interfaces"
                | "nodes"
                | "observables"
                | "osint"
                | "packet_list"
                | "parameters"
                | "parent_process"
                | "policies"
                | "programmatic_credentials"
                | "proxy_endpoint"
                | "related_analytics"
                | "related_events"
                | "resources"
                | "sans"
                | "scim_group_schema"
                | "scim_user_schema"
                | "signatures"
                | "software_components"
                | "tags"
                | "tickets"
                | "tls_extension_list"
                | "traits"
                | "transformation_info_list"
                | "unmapped"
                | "urls"
                | "vulnerabilities"
                | "xattributes"
        )
    )
}

fn insert_json_path(
    root: &mut serde_json::Map<String, serde_json::Value>,
    path: &str,
    value: serde_json::Value,
) -> Result<(), OcsfError> {
    let mut current = root;
    let mut parts = path.split('.').peekable();
    while let Some(part) = parts.next() {
        if parts.peek().is_none() {
            if current.insert(part.to_owned(), value.clone()).is_some() {
                return Err(OcsfError::InvalidParquet {
                    relation: "typed relation".to_owned(),
                    reason: "duplicate decoded typed path",
                });
            }
        } else {
            current = match current
                .entry(part.to_owned())
                .or_insert_with(|| serde_json::Value::Object(Default::default()))
            {
                serde_json::Value::Object(map) => map,
                _ => {
                    return Err(OcsfError::InvalidParquet {
                        relation: "typed relation".to_owned(),
                        reason: "conflicting decoded typed path",
                    });
                }
            };
        }
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
struct BuildReceiptView {
    schema_version: u8,
    snapshot_manifest: SnapshotManifestView,
    runnable_snapshot: RunnableSnapshotView,
    closure: ClosureView,
    completeness_receipt: CompletenessReceiptView,
    output_logical_sha256: String,
    #[serde(default)]
    snapshot_capabilities: Option<SnapshotCapabilitiesObjectView>,
}

#[derive(Debug, Deserialize)]
struct SnapshotCapabilitiesObjectView {
    path: String,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct SnapshotCapabilitiesView {
    schema_version: u8,
    snapshot_logical_sha256: String,
    mapping_completeness: MappingCompletenessView,
}

#[derive(Debug, Deserialize)]
struct MappingCompletenessView {
    state: String,
    source_rows: u64,
    normalized_events: u64,
    unassigned_source_signatures: u64,
    unsupported_source_signatures: u64,
    intentionally_ignored_source_signatures: u64,
    valid_rows_without_normalized_output: u64,
    raw_fallback_rows: u64,
    normalized_events_without_source_provenance: u64,
    native_fields_without_disposition: u64,
    fields_without_typed_projection_or_metadata_justification: u64,
    mapped_fields_without_round_trip_provenance: u64,
    observed_semantic_variants_without_test: u64,
    field_disposition_audit_failures: u64,
}

impl MappingCompletenessView {
    fn has_zero_gaps(&self) -> bool {
        self.unassigned_source_signatures == 0
            && self.unsupported_source_signatures == 0
            && self.intentionally_ignored_source_signatures == 0
            && self.valid_rows_without_normalized_output == 0
            && self.raw_fallback_rows == 0
            && self.normalized_events_without_source_provenance == 0
            && self.native_fields_without_disposition == 0
            && self.fields_without_typed_projection_or_metadata_justification == 0
            && self.mapped_fields_without_round_trip_provenance == 0
            && self.observed_semantic_variants_without_test == 0
            && self.field_disposition_audit_failures == 0
    }
}

#[derive(Debug, Deserialize)]
struct ComponentView {
    id: String,
    version: String,
    sha256: String,
}

fn validate_component(component: &ComponentView) -> Result<(), OcsfError> {
    if component.id.is_empty() || component.version.is_empty() {
        return Err(OcsfError::InvalidReceipt(
            "component identity is incomplete",
        ));
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
struct RunnableSnapshotView {
    component: ComponentView,
    dataset_sha256: String,
    mapping_pack: ComponentView,
    relation_contract: ComponentView,
    normalized_events: u64,
    source_rows: u64,
}

#[derive(Debug, Deserialize)]
struct ClosureView {
    input_rows: u64,
    mapped_source_records: u64,
    mapped_events: u64,
    event_rows: u64,
    rejected_malformed_records: u64,
    unsupported_records: u64,
    unresolved_provenance_fields: u64,
    provenance_digest_mismatches: u64,
}

#[derive(Debug, Deserialize)]
struct CompletenessReceiptView {
    dataset_sha256: String,
    mapping_pack_sha256: String,
    normalized_snapshot_sha256: String,
    relation_contract_sha256: String,
    metrics: CompletenessMetricsView,
}

#[derive(Debug, Deserialize)]
struct CompletenessMetricsView {
    source_rows: u64,
    mapped_source_records: u64,
    rejected_malformed_records: u64,
    normalized_events: u64,
}

#[derive(Debug, Deserialize)]
struct SnapshotManifestView {
    schema_version: u8,
    dataset_sha256: String,
    ocsf_schema_sha256: String,
    extension_pack_sha256: String,
    mapping_pack_sha256: String,
    relation_contract_sha256: String,
    objects: Vec<RelationObjectView>,
    logical_sha256: String,
}

#[derive(Debug, Deserialize)]
struct RelationObjectView {
    relation: String,
    path: String,
    rows: u64,
    sha256: String,
    logical_sha256: String,
}

#[derive(Debug, Error)]
pub enum OcsfError {
    #[error("snapshot I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid build receipt JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error(
        "invalid residual JSON in typed column {path:?} at batch row {row} ({bytes} bytes): {source}"
    )]
    TypedJson {
        path: String,
        row: usize,
        bytes: usize,
        source: serde_json::Error,
    },
    #[error("invalid snapshot digest: {0}")]
    Digest(#[from] rag_contracts::ContractError),
    #[error("unsupported snapshot schema version {0}")]
    UnsupportedSchema(u8),
    #[error("invalid build receipt: {0}")]
    InvalidReceipt(&'static str),
    #[error("unsafe snapshot-relative path {0:?}")]
    UnsafePath(String),
    #[error("relation {relation:?} path must be {expected:?}, found {actual:?}")]
    UnexpectedRelationPath {
        relation: String,
        expected: PathBuf,
        actual: PathBuf,
    },
    #[error("required relation {0:?} is missing")]
    MissingRelation(&'static str),
    #[error("Parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    #[error("Arrow stream error: {0}")]
    Arrow(#[from] arrow_schema::ArrowError),
    #[error("relation {relation:?} declares {expected} rows but Parquet contains {actual}")]
    RowCount {
        relation: String,
        expected: u64,
        actual: u64,
    },
    #[error("relation {relation:?} has invalid Parquet metadata: {reason}")]
    InvalidParquet {
        relation: String,
        reason: &'static str,
    },
    #[error("typed relation {relation:?} requires non-null UTF-8 column {column:?}")]
    RequiredColumn {
        relation: String,
        column: &'static str,
    },
    #[error("relation {0:?} was not admitted with this reader")]
    UnknownRelation(String),
    #[error("relation {0:?} does not match its admitted identity")]
    RelationBinding(String),
    #[error("snapshot object digest differs from the admitted receipt: {0}")]
    ObjectDigest(String),
    #[error("relation {relation:?} has no row group {ordinal}")]
    UnknownRowGroup { relation: String, ordinal: usize },
    #[error("relation {relation:?} does not contain required projection column {column:?}")]
    ProjectionColumn { relation: String, column: String },
    #[error("invalid Parquet projection: {0}")]
    InvalidProjection(&'static str),
}

#[cfg(test)]
mod tests;

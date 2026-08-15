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
};

use arrow_array::RecordBatch;
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
const REQUIRED_EVENT_COLUMNS: [&str; 3] = ["event_id", "typed_event_json", "support_ref"];
const REQUIRED_CORE_RELATIONS: [&str; 7] = [
    "events",
    "event_facets",
    "entities",
    "observables",
    "participants",
    "event_observables",
    "relationships",
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
    pub relation_contract_sha256: Sha256,
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
        self.open_reader(None, None)
    }

    fn open_reader(
        &self,
        row_groups: Option<Vec<usize>>,
        required_columns: Option<&[&str]>,
    ) -> Result<RecordBatchStream, OcsfError> {
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
            for column in required_columns {
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
        Ok(Box::new(reader.map(|batch| batch.map_err(OcsfError::from))))
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

/// Adapter for the current schema-version-1 `livefire-ocsf` local snapshot.
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
            ..
        } = receipt;
        if receipt_schema_version != 1 {
            return Err(OcsfError::UnsupportedSchema(receipt_schema_version));
        }
        if manifest.schema_version != 1 {
            return Err(OcsfError::UnsupportedSchema(manifest.schema_version));
        }

        let snapshot_sha256 = Sha256::new(manifest.logical_sha256)?;
        let dataset_sha256 = Sha256::new(manifest.dataset_sha256)?;
        let mapping_sha256 = Sha256::new(manifest.mapping_pack_sha256)?;
        let ocsf_schema_sha256 = Sha256::new(manifest.ocsf_schema_sha256)?;
        let extension_pack_sha256 = Sha256::new(manifest.extension_pack_sha256)?;
        let relation_contract_sha256 = Sha256::new(manifest.relation_contract_sha256)?;
        let output_logical_sha256 = Sha256::new(output_logical_sha256)?;
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
            let kind = classify_relation(&object.relation, &relative_path)?;
            let object_path = resolve_beneath(&root, &relative_path)?;
            let file = File::open(&object_path)?;
            let metadata = ArrowReaderMetadata::load(&file, Default::default())?;
            validate_row_count(&object.relation, object.rows, metadata.metadata())?;
            if kind == RelationKind::TypedSemantic {
                validate_typed_schema(&object.relation, metadata.schema().as_ref())?;
                typed_rows = typed_rows
                    .checked_add(object.rows)
                    .ok_or(OcsfError::InvalidReceipt("typed row count overflow"))?;
            }
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
                relation_contract_sha256,
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

fn classify_relation(name: &str, path: &Path) -> Result<RelationKind, OcsfError> {
    let expected = if name == "field_provenance" {
        PathBuf::from("authority").join(format!("{name}.parquet"))
    } else {
        PathBuf::from("semantic").join(format!("{name}.parquet"))
    };
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

fn validate_typed_schema(relation: &str, schema: &arrow_schema::Schema) -> Result<(), OcsfError> {
    for name in REQUIRED_EVENT_COLUMNS {
        let field = schema
            .field_with_name(name)
            .map_err(|_| OcsfError::RequiredColumn {
                relation: relation.to_owned(),
                column: name,
            })?;
        if field.data_type() != &DataType::Utf8 || field.is_nullable() {
            return Err(OcsfError::RequiredColumn {
                relation: relation.to_owned(),
                column: name,
            });
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

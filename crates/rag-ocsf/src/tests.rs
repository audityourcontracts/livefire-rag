use std::{fs::File, sync::Arc};

use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};
use serde_json::{Value, json};
use sha2::{Digest, Sha256 as Sha256Hasher};
use tempfile::TempDir;

use super::*;

struct Fixture {
    root: TempDir,
}

struct V2Fixture {
    root: TempDir,
}

impl V2Fixture {
    fn write() -> Self {
        let root = tempfile::tempdir().expect("fixture root");
        for directory in ["graph", "normalized", "provenance"] {
            std::fs::create_dir(root.path().join(directory)).expect("snapshot directory");
        }

        let events_schema = Arc::new(Schema::new(vec![
            Field::new("event_id", DataType::Utf8, false),
            Field::new("event_time_ms", DataType::UInt64, false),
            Field::new("support_ref", DataType::Utf8, false),
        ]));
        let events = RecordBatch::try_new(
            events_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["evt_1", "evt_2"])) as ArrayRef,
                Arc::new(UInt64Array::from(vec![1_u64, 2])) as ArrayRef,
                Arc::new(StringArray::from(vec!["sup_1", "sup_2"])) as ArrayRef,
            ],
        )
        .expect("events batch");
        write_batch(
            root.path().join("graph/events.parquet"),
            events_schema,
            events,
        );

        let typed_schema = v2_typed_schema();
        let typed = RecordBatch::try_new(
            typed_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["evt_1", "evt_2"])) as ArrayRef,
                Arc::new(StringArray::from(vec![Some("alice"), Some("bob")])) as ArrayRef,
                Arc::new(StringArray::from(vec![Some("account created"), None])) as ArrayRef,
                Arc::new(Int64Array::from(vec![Some(1_i64), Some(2_i64)])) as ArrayRef,
                Arc::new(StringArray::from(vec![
                    Some(r#"{"requestParameters":{"userName":"alice"}}"#),
                    Some("{}"),
                ])) as ArrayRef,
                Arc::new(StringArray::from(vec![
                    Some("CreateUser"),
                    Some("DisableUser"),
                ])) as ArrayRef,
                Arc::new(StringArray::from(vec![
                    Some("account_change"),
                    Some("account_change"),
                ])) as ArrayRef,
                Arc::new(StringArray::from(vec![Some("sup_1"), Some("sup_2")])) as ArrayRef,
                Arc::new(StringArray::from(vec![Some(r#"["ocsf.actor"]"#), None])) as ArrayRef,
                Arc::new(StringArray::from(vec!["sup_1", "sup_2"])) as ArrayRef,
            ],
        )
        .expect("typed batch");
        write_batch(
            root.path().join("normalized/ocsf_account_change.parquet"),
            typed_schema.clone(),
            typed,
        );

        for relation in REQUIRED_CORE_RELATIONS
            .into_iter()
            .filter(|relation| *relation != "events")
        {
            let schema = Arc::new(Schema::new(vec![Field::new(
                "support_ref",
                DataType::Utf8,
                false,
            )]));
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(StringArray::from(Vec::<String>::new())) as ArrayRef],
            )
            .expect("empty core batch");
            write_batch(
                root.path().join(format!("graph/{relation}.parquet")),
                schema,
                batch,
            );
        }

        let objects = REQUIRED_CORE_RELATIONS
            .into_iter()
            .map(|relation| object_v2(relation, if relation == "events" { 2 } else { 0 }))
            .chain(std::iter::once(object_v2("ocsf_account_change", 2)))
            .collect::<Vec<_>>();
        let mut receipt = json!({
            "schema_version": 2,
            "snapshot_manifest": {
                "schema_version": 2,
                "dataset_sha256": hex('d'),
                "ocsf_schema_sha256": hex('3'),
                "extension_pack_sha256": hex('4'),
                "mapping_pack_sha256": hex('b'),
                "relation_contract_sha256": hex('5'),
                "objects": objects,
                "logical_sha256": hex('a')
            },
            "output_logical_sha256": hex('a'),
            "runnable_snapshot": {
                "component": {"id":"fixture.snapshot","version":"2","sha256":hex('a')},
                "dataset_sha256":hex('d'),
                "mapping_pack":{"id":"fixture.mapping","version":"2","sha256":hex('b')},
                "relation_contract":{"id":"fixture.relations","version":"3","sha256":hex('5')},
                "normalized_events":2,
                "source_rows":2
            },
            "closure": {
                "input_rows":2,"mapped_source_records":2,"mapped_events":2,"event_rows":2,
                "rejected_malformed_records":0,"unsupported_records":0,
                "unresolved_provenance_fields":0,"provenance_digest_mismatches":0
            },
            "completeness_receipt": {
                "dataset_sha256":hex('d'),
                "mapping_pack_sha256":hex('b'),
                "normalized_snapshot_sha256":hex('a'),
                "relation_contract_sha256":hex('5'),
                "metrics":{
                    "source_rows":2,"mapped_source_records":2,
                    "rejected_malformed_records":0,"normalized_events":2
                }
            }
        });
        for object in receipt["snapshot_manifest"]["objects"]
            .as_array_mut()
            .expect("manifest objects")
        {
            let relative = object["path"].as_str().expect("object path");
            object["sha256"] = Value::String(file_digest(&root.path().join(relative)));
        }
        std::fs::write(
            root.path().join(RECEIPT_FILE),
            serde_json::to_vec_pretty(&receipt).expect("receipt JSON"),
        )
        .expect("write receipt");
        Self { root }
    }

    fn write_v3() -> Self {
        let fixture = Self::write();
        let typed_schema = v2_typed_schema();
        let mut added_typed_objects = Vec::new();
        for relation in SCHEMA_V3_TYPED_RELATIONS
            .into_iter()
            .filter(|relation| *relation != "ocsf_account_change")
        {
            let path = fixture
                .root
                .path()
                .join(format!("normalized/{relation}.parquet"));
            write_batch(
                path.clone(),
                typed_schema.clone(),
                RecordBatch::new_empty(typed_schema.clone()),
            );
            let mut object = object_v2(relation, 0);
            object["sha256"] = json!(file_digest(&path));
            added_typed_objects.push(object);
        }
        let schema = Arc::new(Schema::new(vec![Field::new(
            "support_ref",
            DataType::Utf8,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from(Vec::<String>::new())) as ArrayRef],
        )
        .expect("empty subject-alias batch");
        let relation_path = fixture.root.path().join("graph/subject_aliases.parquet");
        write_batch(relation_path.clone(), schema, batch);

        let provenance_schema = Arc::new(Schema::new(vec![Field::new(
            "support_ref",
            DataType::Utf8,
            false,
        )]));
        let provenance_batch = RecordBatch::try_new(
            provenance_schema.clone(),
            vec![Arc::new(StringArray::from(Vec::<String>::new())) as ArrayRef],
        )
        .expect("empty provenance batch");
        let provenance_path = fixture
            .root
            .path()
            .join("provenance/field_provenance.parquet");
        write_batch(provenance_path.clone(), provenance_schema, provenance_batch);

        let capability = json!({
            "schema_version": 1,
            "snapshot_logical_sha256": hex('a'),
            "event_time_bounds": {"start_ms": 1, "end_ms": 2},
            "classes": [],
            "searchable_subjects": [],
            "mapping_completeness": {
                "state": "complete",
                "source_rows": 2,
                "normalized_events": 2,
                "unassigned_source_signatures": 0,
                "unsupported_source_signatures": 0,
                "intentionally_ignored_source_signatures": 0,
                "valid_rows_without_normalized_output": 0,
                "raw_fallback_rows": 0,
                "normalized_events_without_source_provenance": 0,
                "native_fields_without_disposition": 0,
                "fields_without_typed_projection_or_metadata_justification": 0,
                "mapped_fields_without_round_trip_provenance": 0,
                "observed_semantic_variants_without_test": 0,
                "field_disposition_audit_failures": 0
            }
        });
        let capability_path = fixture
            .root
            .path()
            .join("capabilities/snapshot-capabilities.v1.json");
        std::fs::create_dir(capability_path.parent().expect("capability parent"))
            .expect("capability directory");
        std::fs::write(
            &capability_path,
            serde_json::to_vec_pretty(&capability).expect("capability JSON"),
        )
        .expect("write capability");

        let receipt_path = fixture.root.path().join(RECEIPT_FILE);
        let mut receipt: Value =
            serde_json::from_slice(&std::fs::read(&receipt_path).expect("read schema v2 receipt"))
                .expect("receipt JSON");
        receipt["snapshot_manifest"]["schema_version"] = json!(3);
        receipt["runnable_snapshot"]["component"]["version"] = json!("45");
        let mut subject_aliases = object_v2("subject_aliases", 0);
        subject_aliases["sha256"] = json!(file_digest(&relation_path));
        receipt["snapshot_manifest"]["objects"]
            .as_array_mut()
            .expect("manifest objects")
            .push(subject_aliases);
        let mut field_provenance = object_v2("field_provenance", 0);
        field_provenance["sha256"] = json!(file_digest(&provenance_path));
        receipt["snapshot_manifest"]["objects"]
            .as_array_mut()
            .expect("manifest objects")
            .push(field_provenance);
        receipt["snapshot_manifest"]["objects"]
            .as_array_mut()
            .expect("manifest objects")
            .extend(added_typed_objects);
        receipt["snapshot_capabilities"] = json!({
            "path": "capabilities/snapshot-capabilities.v1.json",
            "sha256": file_digest(&capability_path)
        });
        std::fs::write(
            receipt_path,
            serde_json::to_vec_pretty(&receipt).expect("schema v3 receipt JSON"),
        )
        .expect("write schema v3 receipt");
        fixture
    }
}

impl Fixture {
    fn write() -> Self {
        let root = tempfile::tempdir().expect("fixture root");
        std::fs::create_dir(root.path().join("semantic")).expect("semantic directory");

        let events_schema = Arc::new(Schema::new(vec![
            Field::new("event_id", DataType::Utf8, false),
            Field::new("event_time_ms", DataType::UInt64, false),
            Field::new("support_ref", DataType::Utf8, false),
        ]));
        let events = RecordBatch::try_new(
            events_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["evt_1", "evt_2"])) as ArrayRef,
                Arc::new(UInt64Array::from(vec![1_u64, 2])) as ArrayRef,
                Arc::new(StringArray::from(vec!["sup_1", "sup_2"])) as ArrayRef,
            ],
        )
        .expect("events batch");
        write_batch(
            root.path().join("semantic/events.parquet"),
            events_schema,
            events,
        );

        let typed_schema = Arc::new(Schema::new(vec![
            Field::new("event_id", DataType::Utf8, false),
            Field::new("typed_event_json", DataType::Utf8, false),
            Field::new("support_ref", DataType::Utf8, false),
        ]));
        let typed = RecordBatch::try_new(
            typed_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["evt_1", "evt_2"])) as ArrayRef,
                Arc::new(StringArray::from(vec![
                    "{\"activity\":\"one\"}",
                    "{\"activity\":\"two\"}",
                ])) as ArrayRef,
                Arc::new(StringArray::from(vec!["sup_1", "sup_2"])) as ArrayRef,
            ],
        )
        .expect("typed batch");
        write_batch(
            root.path().join("semantic/ocsf_process_activity.parquet"),
            typed_schema,
            typed,
        );

        for relation in REQUIRED_CORE_RELATIONS
            .into_iter()
            .filter(|relation| *relation != "events")
        {
            let schema = Arc::new(Schema::new(vec![Field::new(
                "support_ref",
                DataType::Utf8,
                false,
            )]));
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(StringArray::from(Vec::<String>::new())) as ArrayRef],
            )
            .expect("empty core batch");
            write_batch(
                root.path().join(format!("semantic/{relation}.parquet")),
                schema,
                batch,
            );
        }

        let mut receipt = json!({
            "schema_version": 1,
            "snapshot_manifest": {
                "schema_version": 1,
                "dataset_sha256": hex('d'),
                "source_inventory_sha256": hex('1'),
                "field_inventory_sha256": hex('2'),
                "ocsf_schema_sha256": hex('3'),
                "extension_pack_sha256": hex('4'),
                "mapping_pack_sha256": hex('b'),
                "relation_contract_sha256": hex('5'),
                "normalizer_sha256": hex('6'),
                "objects": REQUIRED_CORE_RELATIONS
                    .into_iter()
                    .map(|relation| object(relation, if relation == "events" { 2 } else { 0 }))
                    .chain(std::iter::once(object("ocsf_process_activity", 2)))
                    .collect::<Vec<_>>(),
                "logical_sha256": hex('a')
            },
            "output_logical_sha256": hex('a'),
            "runnable_snapshot": {
                "component": {"id":"fixture.snapshot","version":"1","sha256":hex('a')},
                "dataset_sha256":hex('d'),
                "mapping_pack":{"id":"fixture.mapping","version":"1","sha256":hex('b')},
                "relation_contract":{"id":"fixture.relations","version":"1","sha256":hex('5')},
                "normalized_events":2,
                "source_rows":2
            },
            "closure": {
                "input_rows":2,"mapped_source_records":2,"mapped_events":2,"event_rows":2,
                "rejected_malformed_records":0,"unsupported_records":0,
                "unresolved_provenance_fields":0,"provenance_digest_mismatches":0
            },
            "completeness_receipt": {
                "dataset_sha256":hex('d'),
                "mapping_pack_sha256":hex('b'),
                "normalized_snapshot_sha256":hex('a'),
                "relation_contract_sha256":hex('5'),
                "metrics":{
                    "source_rows":2,"mapped_source_records":2,
                    "rejected_malformed_records":0,"normalized_events":2
                }
            }
        });
        for object in receipt["snapshot_manifest"]["objects"]
            .as_array_mut()
            .expect("manifest objects")
        {
            let relative = object["path"].as_str().expect("object path");
            object["sha256"] = Value::String(file_digest(&root.path().join(relative)));
        }
        std::fs::write(
            root.path().join(RECEIPT_FILE),
            serde_json::to_vec_pretty(&receipt).expect("receipt JSON"),
        )
        .expect("write receipt");
        Self { root }
    }

    fn mutate_receipt(&self, mutate: impl FnOnce(&mut Value)) {
        let path = self.root.path().join(RECEIPT_FILE);
        let mut receipt: Value =
            serde_json::from_slice(&std::fs::read(&path).expect("read receipt"))
                .expect("parse receipt");
        mutate(&mut receipt);
        std::fs::write(path, serde_json::to_vec_pretty(&receipt).expect("JSON"))
            .expect("write receipt");
    }
}

fn object(relation: &str, rows: u64) -> Value {
    json!({
        "relation": relation,
        "path": format!("semantic/{relation}.parquet"),
        "rows": rows,
        "sha256": hex('7'),
        "logical_sha256": hex('8')
    })
}

fn object_v2(relation: &str, rows: u64) -> Value {
    let directory = match relation {
        "field_provenance" => "provenance",
        relation if relation.starts_with("ocsf_") => "normalized",
        _ => "graph",
    };
    json!({
        "relation": relation,
        "path": format!("{directory}/{relation}.parquet"),
        "rows": rows,
        "sha256": hex('7'),
        "logical_sha256": hex('8')
    })
}

fn v2_typed_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("event_id", DataType::Utf8, false),
        Field::new("event.account", DataType::Utf8, true),
        Field::new("event.ocsf.message", DataType::Utf8, true),
        Field::new("event.ocsf.time", DataType::Int64, true),
        Field::new("event.ocsf.unmapped", DataType::Utf8, true),
        Field::new("event.operation", DataType::Utf8, true),
        Field::new("event.semantic_class", DataType::Utf8, true),
        Field::new("event.support_ref", DataType::Utf8, true),
        Field::new("empty_object_paths", DataType::Utf8, true),
        Field::new("support_ref", DataType::Utf8, false),
    ]))
}

fn hex(character: char) -> String {
    std::iter::repeat_n(character, 64).collect()
}

fn file_digest(path: &Path) -> String {
    format!(
        "{:x}",
        Sha256Hasher::digest(std::fs::read(path).expect("fixture object"))
    )
}

fn write_batch(path: PathBuf, schema: Arc<Schema>, batch: RecordBatch) {
    let mut writer = ArrowWriter::try_new(
        File::create(path).expect("Parquet file"),
        schema,
        Some(
            WriterProperties::builder()
                .set_max_row_group_row_count(Some(1))
                .build(),
        ),
    )
    .expect("Parquet writer");
    writer.write(&batch).expect("write batch");
    writer.close().expect("close writer");
}

#[test]
fn opens_current_receipt_discovers_typed_relations_and_streams_batches() {
    let fixture = Fixture::write();
    let reader =
        LocalSnapshotReader::open_with_batch_size(fixture.root.path(), 1).expect("snapshot opens");

    assert_eq!(reader.identity().normalized_events, 2);
    assert_eq!(reader.identity().snapshot_sha256.as_str(), hex('a'));
    assert_eq!(reader.identity().mapping_sha256.as_str(), hex('b'));
    let relations: Vec<_> = reader.typed_relations().collect();
    assert_eq!(relations.len(), 1);
    assert_eq!(relations[0].name, "ocsf_process_activity");
    assert_eq!(
        relations[0].path,
        Path::new("semantic/ocsf_process_activity.parquet")
    );

    let batches = reader
        .scan(relations[0])
        .expect("scan opens")
        .collect::<Result<Vec<_>, _>>()
        .expect("stream succeeds");
    assert_eq!(batches.len(), 2);
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
    let first = batches[0]
        .column_by_name("event_id")
        .expect("event_id")
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("UTF-8");
    assert_eq!(first.value(0), "evt_1");
    assert!(!first.is_null(0));
}

#[test]
fn opens_schema_v2_and_reconstructs_the_logical_typed_rows() {
    let fixture = V2Fixture::write();
    let reader = LocalSnapshotReader::open_with_batch_size(fixture.root.path(), 1)
        .expect("schema v2 snapshot opens");
    assert_eq!(reader.identity().schema_version, 2);
    let relation = reader.typed_relations().next().expect("typed relation");
    assert_eq!(relation.name, "ocsf_account_change");
    assert_eq!(
        relation.path,
        Path::new("normalized/ocsf_account_change.parquet")
    );

    let object = reader.admit_object(relation).expect("object admits");
    let batch = object
        .scan_row_group(0, &["event_id", "typed_event_json", "support_ref"])
        .expect("logical scan opens")
        .next()
        .expect("first batch")
        .expect("logical reconstruction");
    assert_eq!(batch.num_columns(), 3);
    let typed = batch
        .column_by_name("typed_event_json")
        .expect("typed JSON")
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("UTF-8");
    let value: Value = serde_json::from_str(typed.value(0)).expect("valid typed JSON");
    assert_eq!(value["account"], "alice");
    assert_eq!(value["operation"], "CreateUser");
    assert_eq!(value["ocsf"]["time"], 1);
    assert_eq!(
        value["ocsf"]["unmapped"]["requestParameters"]["userName"],
        "alice"
    );
    assert_eq!(value["ocsf"]["actor"], json!({}));

    let identifiers = object
        .scan_projected(&["support_ref", "event_id"])
        .expect("identifier-only projection")
        .collect::<Result<Vec<_>, _>>()
        .expect("identifier rows");
    assert!(
        identifiers
            .iter()
            .all(|batch| batch.column_by_name("typed_event_json").is_none())
    );
    assert_eq!(
        identifiers.iter().map(RecordBatch::num_rows).sum::<usize>(),
        2
    );
}

#[test]
fn opens_schema_v3_and_binds_the_capability_sidecar() {
    let fixture = V2Fixture::write_v3();
    let reader = LocalSnapshotReader::open(fixture.root.path()).expect("schema v3 snapshot opens");
    assert_eq!(reader.identity().schema_version, 3);
    assert_eq!(reader.identity().snapshot_version, "45");
    assert_eq!(
        reader
            .identity()
            .snapshot_capabilities_sha256
            .as_ref()
            .expect("capability digest")
            .as_str(),
        file_digest(
            &fixture
                .root
                .path()
                .join("capabilities/snapshot-capabilities.v1.json")
        )
    );
    assert!(reader.identity().relations.iter().any(|relation| {
        relation.name == "subject_aliases" && relation.kind == RelationKind::SemanticGraph
    }));
    assert_eq!(reader.typed_relations().count(), 19);
}

#[test]
fn schema_v3_requires_an_exact_capability_sidecar_and_subject_aliases() {
    let fixture = V2Fixture::write_v3();
    std::fs::write(
        fixture
            .root
            .path()
            .join("capabilities/snapshot-capabilities.v1.json"),
        b"{}",
    )
    .expect("tamper capability");
    assert!(matches!(
        LocalSnapshotReader::open(fixture.root.path()),
        Err(OcsfError::ObjectDigest(_))
    ));

    let fixture = V2Fixture::write_v3();
    let receipt_path = fixture.root.path().join(RECEIPT_FILE);
    let mut receipt: Value =
        serde_json::from_slice(&std::fs::read(&receipt_path).expect("read schema v3 receipt"))
            .expect("receipt JSON");
    receipt["snapshot_manifest"]["objects"]
        .as_array_mut()
        .expect("objects")
        .retain(|object| object["relation"] != "subject_aliases");
    std::fs::write(
        receipt_path,
        serde_json::to_vec_pretty(&receipt).expect("receipt JSON"),
    )
    .expect("write receipt");
    assert!(matches!(
        LocalSnapshotReader::open(fixture.root.path()),
        Err(OcsfError::MissingRelation("subject_aliases"))
    ));

    let fixture = V2Fixture::write_v3();
    let receipt_path = fixture.root.path().join(RECEIPT_FILE);
    let mut receipt: Value =
        serde_json::from_slice(&std::fs::read(&receipt_path).expect("read schema v3 receipt"))
            .expect("receipt JSON");
    receipt["snapshot_manifest"]["objects"]
        .as_array_mut()
        .expect("objects")
        .retain(|object| object["relation"] != "field_provenance");
    std::fs::write(
        receipt_path,
        serde_json::to_vec_pretty(&receipt).expect("receipt JSON"),
    )
    .expect("write receipt");
    assert!(matches!(
        LocalSnapshotReader::open(fixture.root.path()),
        Err(OcsfError::MissingRelation("field_provenance"))
    ));
}

#[test]
fn schema_v3_rejects_incomplete_capabilities_and_unknown_graph_relations() {
    let fixture = V2Fixture::write_v3();
    let capability_path = fixture
        .root
        .path()
        .join("capabilities/snapshot-capabilities.v1.json");
    let mut capability: Value =
        serde_json::from_slice(&std::fs::read(&capability_path).expect("read capability"))
            .expect("capability JSON");
    capability["mapping_completeness"]["raw_fallback_rows"] = json!(1);
    std::fs::write(
        &capability_path,
        serde_json::to_vec_pretty(&capability).expect("capability JSON"),
    )
    .expect("write capability");
    let receipt_path = fixture.root.path().join(RECEIPT_FILE);
    let mut receipt: Value =
        serde_json::from_slice(&std::fs::read(&receipt_path).expect("read receipt"))
            .expect("receipt JSON");
    receipt["snapshot_capabilities"]["sha256"] = json!(file_digest(&capability_path));
    std::fs::write(
        &receipt_path,
        serde_json::to_vec_pretty(&receipt).expect("receipt JSON"),
    )
    .expect("write receipt");
    assert!(matches!(
        LocalSnapshotReader::open(fixture.root.path()),
        Err(OcsfError::InvalidReceipt(
            "snapshot capability identity closure"
        ))
    ));

    let fixture = V2Fixture::write_v3();
    let receipt_path = fixture.root.path().join(RECEIPT_FILE);
    let mut receipt: Value =
        serde_json::from_slice(&std::fs::read(&receipt_path).expect("read receipt"))
            .expect("receipt JSON");
    receipt["snapshot_manifest"]["objects"]
        .as_array_mut()
        .expect("objects")
        .push(json!({
            "relation": "unexpected_graph",
            "path": "graph/unexpected_graph.parquet",
            "rows": 0,
            "sha256": hex('7'),
            "logical_sha256": hex('8')
        }));
    std::fs::write(
        receipt_path,
        serde_json::to_vec_pretty(&receipt).expect("receipt JSON"),
    )
    .expect("write receipt");
    assert!(matches!(
        LocalSnapshotReader::open(fixture.root.path()),
        Err(OcsfError::InvalidReceipt(
            "manifest version 3 declares an unknown relation"
        ))
    ));
}

#[test]
fn object_admission_and_legacy_scan_reject_post_open_digest_drift() {
    let fixture = Fixture::write();
    let reader = LocalSnapshotReader::open(fixture.root.path()).expect("snapshot opens");
    let relation = reader.typed_relations().next().expect("typed relation");
    use std::io::Write as _;
    File::options()
        .append(true)
        .open(fixture.root.path().join(&relation.path))
        .expect("typed object")
        .write_all(b"post-admission mutation")
        .expect("mutate typed object");
    assert!(matches!(
        reader.admit_object(relation),
        Err(OcsfError::ObjectDigest(_))
    ));
    assert!(matches!(
        reader.scan(relation),
        Err(OcsfError::ObjectDigest(_))
    ));
}

#[test]
fn admitted_object_exposes_exact_row_groups_and_projects_required_columns() {
    let fixture = Fixture::write();
    let reader =
        LocalSnapshotReader::open_with_batch_size(fixture.root.path(), 1).expect("snapshot opens");
    let relation = reader.typed_relations().next().expect("typed relation");
    let object = reader.admit_object(relation).expect("object admits");

    assert_eq!(object.relation(), relation);
    assert_eq!(object.digest_validation_count(), 1);
    assert_eq!(
        object
            .row_groups()
            .iter()
            .map(|group| (group.ordinal, group.first_row, group.rows))
            .collect::<Vec<_>>(),
        [(0, 0, 1), (1, 1, 1)]
    );
    assert!(
        object
            .row_groups()
            .iter()
            .all(|group| group.compressed_bytes > 0)
    );
    assert_eq!(
        object
            .row_groups()
            .iter()
            .map(|group| group.rows)
            .sum::<u64>(),
        relation.rows
    );

    let batches = object
        .scan_projected(&["support_ref", "event_id"])
        .expect("projected scan opens")
        .collect::<Result<Vec<_>, _>>()
        .expect("projected scan succeeds");
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
    for batch in &batches {
        assert_eq!(batch.num_columns(), 2);
        assert!(batch.column_by_name("event_id").is_some());
        assert!(batch.column_by_name("support_ref").is_some());
        assert!(batch.column_by_name("typed_event_json").is_none());
    }
    let second = object
        .scan_row_group(1, &["event_id"])
        .expect("second group opens")
        .collect::<Result<Vec<_>, _>>()
        .expect("second group succeeds");
    let ids = second[0]
        .column_by_name("event_id")
        .expect("event id")
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("UTF-8");
    assert_eq!(ids.value(0), "evt_2");
    assert_eq!(object.digest_validation_count(), 1);
}

#[test]
fn admitted_row_group_readers_are_independent_and_concurrent() {
    let fixture = Fixture::write();
    let reader = LocalSnapshotReader::open(fixture.root.path()).expect("snapshot opens");
    let relation = reader.typed_relations().next().expect("typed relation");
    let object = Arc::new(reader.admit_object(relation).expect("object admits"));

    let values = std::thread::scope(|scope| {
        let readers = (0..2)
            .map(|ordinal| {
                let object = Arc::clone(&object);
                scope.spawn(move || {
                    let batches = object
                        .scan_row_group(ordinal, &["event_id"])
                        .expect("row group opens")
                        .collect::<Result<Vec<_>, _>>()
                        .expect("row group succeeds");
                    let ids = batches[0]
                        .column_by_name("event_id")
                        .expect("event id")
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .expect("UTF-8");
                    ids.value(0).to_owned()
                })
            })
            .collect::<Vec<_>>();
        readers
            .into_iter()
            .map(|reader| reader.join().expect("reader thread"))
            .collect::<Vec<_>>()
    });
    assert_eq!(values, ["evt_1", "evt_2"]);
    assert_eq!(object.digest_validation_count(), 1);
}

#[test]
fn admitted_object_rejects_unknown_groups_and_incomplete_projection() {
    let fixture = Fixture::write();
    let reader = LocalSnapshotReader::open(fixture.root.path()).expect("snapshot opens");
    let relation = reader.typed_relations().next().expect("typed relation");
    let object = reader.admit_object(relation).expect("object admits");

    assert!(matches!(
        object.scan_row_group(2, &["event_id"]),
        Err(OcsfError::UnknownRowGroup { ordinal: 2, .. })
    ));
    assert!(matches!(
        object.scan_projected(&["missing"]),
        Err(OcsfError::ProjectionColumn { column, .. }) if column == "missing"
    ));
    assert!(matches!(
        object.scan_projected(&[]),
        Err(OcsfError::InvalidProjection(_))
    ));
}

#[test]
fn rejects_parent_absolute_and_symlink_escape_paths() {
    for bad in [
        "../outside.parquet",
        "/tmp/outside.parquet",
        "semantic/../outside.parquet",
    ] {
        let fixture = Fixture::write();
        fixture.mutate_receipt(|receipt| {
            receipt["snapshot_manifest"]["objects"][1]["path"] = json!(bad);
        });
        assert!(matches!(
            LocalSnapshotReader::open(fixture.root.path()),
            Err(OcsfError::UnsafePath(_)) | Err(OcsfError::UnexpectedRelationPath { .. })
        ));
    }

    let fixture = Fixture::write();
    let outside = tempfile::NamedTempFile::new().expect("outside file");
    std::fs::remove_file(
        fixture
            .root
            .path()
            .join("semantic/ocsf_process_activity.parquet"),
    )
    .expect("remove relation");
    #[cfg(unix)]
    std::os::unix::fs::symlink(
        outside.path(),
        fixture
            .root
            .path()
            .join("semantic/ocsf_process_activity.parquet"),
    )
    .expect("symlink");
    assert!(matches!(
        LocalSnapshotReader::open(fixture.root.path()),
        Err(OcsfError::UnsafePath(_))
    ));
}

#[test]
fn rejects_manifest_and_parquet_row_count_mismatch() {
    let fixture = Fixture::write();
    fixture.mutate_receipt(|receipt| {
        let object = receipt["snapshot_manifest"]["objects"]
            .as_array_mut()
            .expect("objects")
            .iter_mut()
            .find(|object| object["relation"] == "ocsf_process_activity")
            .expect("typed object");
        object["rows"] = json!(3);
    });
    assert!(matches!(
        LocalSnapshotReader::open(fixture.root.path()),
        Err(OcsfError::RowCount {
            expected: 3,
            actual: 2,
            ..
        })
    ));
}

#[test]
fn rejects_a_snapshot_missing_a_required_core_relation() {
    let fixture = Fixture::write();
    fixture.mutate_receipt(|receipt| {
        receipt["snapshot_manifest"]["objects"]
            .as_array_mut()
            .expect("objects")
            .retain(|object| object["relation"] != "relationships");
    });
    assert!(matches!(
        LocalSnapshotReader::open(fixture.root.path()),
        Err(OcsfError::MissingRelation("relationships"))
    ));
}

#[test]
fn rejects_partial_typed_inventory_and_receipt_identity_drift() {
    let fixture = Fixture::write();
    fixture.mutate_receipt(|receipt| {
        receipt["snapshot_manifest"]["objects"]
            .as_array_mut()
            .expect("objects")
            .retain(|object| object["relation"] != "ocsf_process_activity");
    });
    assert!(matches!(
        LocalSnapshotReader::open(fixture.root.path()),
        Err(OcsfError::InvalidReceipt("normalized event count closure"))
    ));

    let fixture = Fixture::write();
    fixture.mutate_receipt(|receipt| {
        receipt["output_logical_sha256"] = json!(hex('f'));
    });
    assert!(matches!(
        LocalSnapshotReader::open(fixture.root.path()),
        Err(OcsfError::InvalidReceipt(
            "snapshot receipt identity closure"
        ))
    ));

    let fixture = Fixture::write();
    fixture.mutate_receipt(|receipt| {
        receipt["closure"]["unresolved_provenance_fields"] = json!(1);
    });
    assert!(matches!(
        LocalSnapshotReader::open(fixture.root.path()),
        Err(OcsfError::InvalidReceipt(
            "snapshot receipt identity closure"
        ))
    ));
}

#[test]
fn rejects_typed_relation_missing_required_column() {
    let fixture = Fixture::write();
    let path = fixture
        .root
        .path()
        .join("semantic/ocsf_process_activity.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("event_id", DataType::Utf8, false),
        Field::new("support_ref", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["evt_1", "evt_2"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["sup_1", "sup_2"])) as ArrayRef,
        ],
    )
    .expect("batch");
    write_batch(path, schema, batch);

    assert!(matches!(
        LocalSnapshotReader::open(fixture.root.path()),
        Err(OcsfError::RequiredColumn {
            column: "typed_event_json",
            ..
        })
    ));
}

#[test]
fn refuses_a_forged_relation_descriptor_at_scan_time() {
    let fixture = Fixture::write();
    let reader = LocalSnapshotReader::open(fixture.root.path()).expect("snapshot opens");
    let mut relation = reader
        .typed_relations()
        .next()
        .expect("typed relation")
        .clone();
    relation.rows += 1;
    assert!(matches!(
        reader.scan(&relation),
        Err(OcsfError::RelationBinding(_))
    ));
}

#[test]
fn fixture_objects_are_real_parquet_files() {
    let fixture = Fixture::write();
    let bytes = std::fs::read(
        fixture
            .root
            .path()
            .join("semantic/ocsf_process_activity.parquet"),
    )
    .expect("read Parquet");
    assert_eq!(&bytes[..4], b"PAR1");
    let digest = format!("{:x}", Sha256Hasher::digest(&bytes));
    assert_eq!(digest.len(), 64);
}

#[test]
fn schema_v2_distinguishes_envelope_text_from_residual_ocsf_json() {
    // M44 carries an envelope-level `resources` text value as well as an OCSF
    // class field with the same final path segment but a residual JSON type.
    // The physical Arrow type is Utf8 for both, so the complete path is part
    // of the decoding contract.
    assert!(!residual_json_column("event.resources"));
    assert!(residual_json_column("event.ocsf.resources"));
    assert!(residual_json_column("event.ocsf.unmapped"));
    assert!(residual_json_column("event.semantics"));
}

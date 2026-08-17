//! Optional, bounded check against a locally available M45 snapshot.

use arrow_array::StringArray;
use rag_ocsf::{LocalSnapshotReader, SnapshotReader};
use rag_projection::{ComponentRef, ProjectionContext, ProjectionInput, project_m45_command};

#[test]
#[ignore = "requires LIVEFIRE_OCSF_SNAPSHOT to name the local M45 snapshot"]
fn surviving_m45_sysmon_bash_and_api_fields_are_projected() {
    let root = std::env::var_os("LIVEFIRE_OCSF_SNAPSHOT")
        .expect("set LIVEFIRE_OCSF_SNAPSHOT to the M45 snapshot root");
    let reader = LocalSnapshotReader::open(root).expect("M45 snapshot admission");
    let identity = reader.identity();
    assert_eq!(identity.schema_version, 3);
    assert_eq!(identity.snapshot_version, "45");
    assert_eq!(identity.mapping_id, "botsv3-ocsf-m45");
    assert!(identity.snapshot_capabilities_sha256.is_some());
    let context = ProjectionContext {
        snapshot: ComponentRef {
            id: identity.snapshot_id.clone(),
            version: identity.snapshot_version.clone(),
            sha256: identity.snapshot_sha256.to_string(),
            uri: None,
        },
        mapping_pack: ComponentRef {
            id: identity.mapping_id.clone(),
            version: identity.mapping_version.clone(),
            sha256: identity.mapping_sha256.to_string(),
            uri: None,
        },
    };

    let process = reader
        .typed_relations()
        .find(|relation| relation.name == "ocsf_process_activity")
        .expect("process relation");
    let process_object = reader
        .admit_object(process)
        .expect("process object admission");
    let mut found_sysmon = false;
    let mut found_bash = false;
    let mut rows_checked = 0_usize;
    for group in process_object.row_groups().iter().take(8) {
        let mut batches = process_object
            .scan_row_group(
                group.ordinal,
                &["event_id", "typed_event_json", "support_ref"],
            )
            .expect("bounded process scan");
        if let Some(batch) = batches.next() {
            let batch = batch.expect("process batch");
            let event_ids = strings(&batch, "event_id");
            let events = strings(&batch, "typed_event_json");
            let support_refs = strings(&batch, "support_ref");
            rows_checked += batch.num_rows();
            for row in 0..batch.num_rows() {
                let Some(output) = project_m45_command(ProjectionInput {
                    relation_name: &process.name,
                    event_id: event_ids.value(row),
                    typed_event_json: events.value(row),
                    support_ref: support_refs.value(row),
                    context: &context,
                })
                .expect("process projection") else {
                    continue;
                };
                assert_eq!(output.occurrence.event_id, event_ids.value(row));
                assert_eq!(output.occurrence.support_ref, support_refs.value(row));
                let text = &output.document.expect("command document").semantic_text;
                found_sysmon |= text.contains("sysmon_command_line");
                found_bash |= text.contains("bash_history_tokens");
            }
        }
        if found_sysmon && found_bash {
            break;
        }
    }
    assert!(
        rows_checked <= 8 * 8_192,
        "smoke scan exceeded its row bound"
    );
    assert!(
        found_sysmon,
        "bounded M45 sample did not retain a Sysmon command line"
    );
    assert!(
        found_bash,
        "bounded M45 sample did not retain bash history tokens"
    );

    let api = reader
        .typed_relations()
        .find(|relation| relation.name == "ocsf_api_activity")
        .expect("API relation");
    let api_object = reader.admit_object(api).expect("API object admission");
    let first_group = api_object.row_groups().first().expect("API row group");
    let batch = api_object
        .scan_row_group(
            first_group.ordinal,
            &["event_id", "typed_event_json", "support_ref"],
        )
        .expect("bounded API scan")
        .next()
        .expect("API batch")
        .expect("API rows");
    let event_ids = strings(&batch, "event_id");
    let events = strings(&batch, "typed_event_json");
    let support_refs = strings(&batch, "support_ref");
    assert!(batch.num_rows() <= 8_192);
    assert!((0..batch.num_rows()).any(|row| {
        project_m45_command(ProjectionInput {
            relation_name: &api.name,
            event_id: event_ids.value(row),
            typed_event_json: events.value(row),
            support_ref: support_refs.value(row),
            context: &context,
        })
        .expect("API projection")
        .and_then(|output| output.document)
        .is_some_and(|document| document.semantic_text.contains("normalized_api_operation"))
    }));
}

fn strings<'a>(batch: &'a arrow_array::RecordBatch, name: &str) -> &'a StringArray {
    batch
        .column_by_name(name)
        .expect("required column")
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("UTF-8 column")
}

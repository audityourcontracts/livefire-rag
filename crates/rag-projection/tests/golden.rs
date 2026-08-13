use pretty_assertions::assert_eq;
use rag_projection::{
    ComponentRef, ProjectionContext, ProjectionInput, TerminalDisposition, project,
};
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    relation_name: String,
    event_id: String,
    support_ref: String,
    typed_event: Value,
    expected: Expected,
}

#[derive(Deserialize)]
struct Expected {
    document_kind: String,
    semantic_group_sha256: String,
    semantic_text: Option<String>,
    event_time: Option<String>,
    terminal_disposition: String,
}

fn component(id: &str, digit: char) -> ComponentRef {
    ComponentRef {
        id: id.to_owned(),
        version: "1".to_owned(),
        sha256: digit.to_string().repeat(64),
        uri: None,
    }
}

fn context() -> ProjectionContext {
    ProjectionContext {
        snapshot: component("fixture.snapshot", 'a'),
        mapping_pack: component("fixture.mapping", 'b'),
    }
}

#[test]
fn matches_python_projection_oracle_for_representative_families() {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../rust-fixtures/projection/golden.v1.json"
    ))
    .unwrap();
    let context = context();
    for case in fixture.cases {
        let typed_event = serde_json::to_string(&case.typed_event).unwrap();
        let output = project(ProjectionInput {
            relation_name: &case.relation_name,
            event_id: &case.event_id,
            typed_event_json: &typed_event,
            support_ref: &case.support_ref,
            context: &context,
        })
        .unwrap();
        assert_eq!(
            serde_json::to_value(output.occurrence.document_kind).unwrap(),
            Value::String(case.expected.document_kind),
            "{} kind",
            case.name
        );
        assert_eq!(
            output.occurrence.semantic_group_sha256, case.expected.semantic_group_sha256,
            "{} group identity",
            case.name
        );
        assert_eq!(
            output.occurrence.event_time, case.expected.event_time,
            "{} time",
            case.name
        );
        assert_eq!(
            serde_json::to_value(output.occurrence.terminal_disposition).unwrap(),
            Value::String(case.expected.terminal_disposition),
            "{} disposition",
            case.name
        );
        assert_eq!(
            output
                .document
                .as_ref()
                .map(|row| row.semantic_text.clone()),
            case.expected.semantic_text,
            "{} semantic text",
            case.name
        );
        assert_eq!(output.occurrence.snapshot, context.snapshot);
        assert_eq!(output.occurrence.mapping_pack, context.mapping_pack);
    }
}

#[test]
fn hostile_values_are_absent_from_embedding_text_and_exact_attributes() {
    let context = context();
    let event = serde_json::json!({
        "activity_name": "Call",
        "api_key": "AKIAABCDEFGHIJKLMNOP",
        "password": "hunter2",
        "message": "password=swordfish Bearer abcdefghijklmnop victim@example.com 192.168.1.2",
        "actor": {"email": "victim@example.com"},
        "resource": {"arn": "arn:aws:s3:::private-bucket"}
    });
    let event_json = serde_json::to_string(&event).unwrap();
    let output = project(ProjectionInput {
        relation_name: "ocsf_api_activity",
        event_id: "hostile",
        typed_event_json: &event_json,
        support_ref: "support:hostile",
        context: &context,
    })
    .unwrap();
    let text = &output.document.unwrap().semantic_text;
    for forbidden in [
        "hunter2",
        "swordfish",
        "abcdefghijklmnop",
        "victim@example.com",
        "192.168.1.2",
        "private-bucket",
    ] {
        assert!(
            !text.contains(forbidden),
            "semantic text leaked {forbidden}: {text}"
        );
    }
    assert!(text.contains("<redacted:secret>"));
    let paths: Vec<_> = output
        .occurrence
        .exact_attributes
        .iter()
        .map(|row| row.path.as_str())
        .collect();
    assert!(!paths.contains(&"/api_key"));
    assert!(!paths.contains(&"/password"));
    assert!(!paths.contains(&"/message"));
    assert!(
        output
            .occurrence
            .exact_attribute_projection
            .source_hydration_required
    );
}

#[test]
fn malformed_and_unknown_input_remain_structured_only_occurrences() {
    let context = context();
    for (relation, json) in [
        ("ocsf_api_activity", "{"),
        ("custom_relation", "{\"action\":\"test\"}"),
    ] {
        let output = project(ProjectionInput {
            relation_name: relation,
            event_id: "event",
            typed_event_json: json,
            support_ref: "support",
            context: &context,
        })
        .unwrap();
        assert!(output.document.is_none());
        assert_eq!(
            output.occurrence.terminal_disposition,
            TerminalDisposition::StructuredOnlyOccurrence
        );
        assert!(
            output
                .occurrence
                .exact_attribute_projection
                .source_hydration_required
                || relation == "custom_relation"
        );
    }
}

#[test]
fn event_identity_does_not_change_semantic_group_identity() {
    let context = context();
    let event = r#"{"activity_name":"Delete","resource":{"name":"example"},"time":1710000000}"#;
    let project_one = |event_id: &str, support_ref: &str| {
        project(ProjectionInput {
            relation_name: "ocsf_api_activity",
            event_id,
            typed_event_json: event,
            support_ref,
            context: &context,
        })
        .unwrap()
    };
    let first = project_one("one", "support:one");
    let second = project_one("two", "support:two");
    assert_eq!(
        first.occurrence.semantic_group_id,
        second.occurrence.semantic_group_id
    );
    assert_ne!(first.occurrence.event_id, second.occurrence.event_id);
}

#[test]
fn every_current_typed_relation_has_a_scenario_blind_terminal_classification() {
    let context = context();
    let activity = [
        "ocsf_api_activity",
        "ocsf_application_lifecycle",
        "ocsf_authentication",
        "ocsf_datastore_activity",
        "ocsf_dns_activity",
        "ocsf_email_activity",
        "ocsf_entity_management",
        "ocsf_event_log_activity",
        "ocsf_file_activity",
        "ocsf_http_activity",
        "ocsf_network_activity",
        "ocsf_process_activity",
    ];
    let state = [
        "ocsf_cloud_resources_inventory_info",
        "ocsf_ext_livefire_configuration_snapshot",
        "ocsf_inventory_info",
        "ocsf_user_inventory",
    ];
    for (relation, expected) in activity
        .into_iter()
        .map(|relation| (relation, "activity"))
        .chain(state.into_iter().map(|relation| (relation, "state")))
        .chain([("ocsf_detection_finding", "detection")])
    {
        let output = project(ProjectionInput {
            relation_name: relation,
            event_id: relation,
            typed_event_json: "{}",
            support_ref: "support",
            context: &context,
        })
        .unwrap();
        assert_eq!(
            serde_json::to_value(output.occurrence.document_kind).unwrap(),
            Value::String(expected.to_owned()),
            "{relation}"
        );
        assert!(output.document.is_some(), "{relation} should be searchable");
    }
    let metric = project(ProjectionInput {
        relation_name: "ocsf_ext_livefire_system_metric",
        event_id: "metric",
        typed_event_json: "{}",
        support_ref: "support",
        context: &context,
    })
    .unwrap();
    assert!(metric.document.is_none());
    assert_eq!(
        metric.occurrence.terminal_disposition,
        TerminalDisposition::StructuredOnlyOccurrence
    );
}

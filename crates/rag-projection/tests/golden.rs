use pretty_assertions::assert_eq;
use rag_projection::{
    ComponentRef, ProjectionContext, ProjectionInput, TerminalDisposition, project,
    project_document_summary, project_event_time,
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
        let summary = project_document_summary(&case.relation_name, &typed_event, &context)
            .expect("summary projection");
        assert_eq!(summary.document, output.document, "{} summary", case.name);
        assert_eq!(
            (summary.event_time, summary.event_time_availability),
            (
                output.occurrence.event_time.clone(),
                output.occurrence.event_time_availability
            ),
            "{} summary time",
            case.name
        );
        assert_eq!(
            project_event_time(&typed_event),
            (
                output.occurrence.event_time.clone(),
                output.occurrence.event_time_availability
            ),
            "{} fast event-time projection",
            case.name
        );
    }
}

#[test]
fn fast_event_time_projection_matches_total_projection_for_unavailable_payloads() {
    let context = context();
    for typed_event_json in ["null", "[]", "{not-json", "{}"] {
        let output = project(ProjectionInput {
            relation_name: "ocsf_ext_livefire_system_metric",
            event_id: "metric",
            typed_event_json,
            support_ref: "support:metric",
            context: &context,
        })
        .unwrap();
        assert_eq!(
            project_event_time(typed_event_json),
            (
                output.occurrence.event_time.clone(),
                output.occurrence.event_time_availability
            ),
            "{typed_event_json}"
        );
        let summary = project_document_summary(
            "ocsf_ext_livefire_system_metric",
            typed_event_json,
            &context,
        )
        .expect("summary projection");
        assert_eq!(summary.document, output.document);
        assert_eq!(
            (summary.event_time, summary.event_time_availability),
            (
                output.occurrence.event_time.clone(),
                output.occurrence.event_time_availability
            )
        );
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
fn camel_case_time_and_identifier_values_do_not_split_semantic_groups() {
    let context = context();
    let project_row = |calendar_time: &str, host_identifier: &str, unix_time: u64| {
        let typed_event = serde_json::json!({
            "semantic_class": "process",
            "ocsf": {
                "activity_id": 99,
                "class_uid": 1007,
                "time": 1534762063000_u64,
                "unmapped": {
                    "action": "added",
                    "calendarTime": calendar_time,
                    "hostIdentifier": host_identifier,
                    "unixTime": unix_time,
                    "columns": {"cmdline": "\"awk\" --version", "path": "/bin/gawk"},
                }
            }
        });
        project(ProjectionInput {
            relation_name: "ocsf_process_activity",
            event_id: "event",
            typed_event_json: &serde_json::to_string(&typed_event).unwrap(),
            support_ref: "support:event",
            context: &context,
        })
        .unwrap()
    };

    let first = project_row(
        "Mon Aug 20 10:47:43 2018 UTC",
        "gacrux.i-0920036c8ca91e501",
        1_534_762_063,
    );
    let second = project_row("another event time", "another-host", 1_700_000_000);

    assert_eq!(
        first.occurrence.semantic_group_id,
        second.occurrence.semantic_group_id
    );
    for forbidden in ["calendarTime", "hostIdentifier", "unixTime", "another-host"] {
        assert!(
            !first
                .document
                .as_ref()
                .unwrap()
                .semantic_text
                .contains(forbidden)
        );
        assert!(
            !second
                .document
                .as_ref()
                .unwrap()
                .semantic_text
                .contains(forbidden)
        );
    }
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

#[test]
fn every_searchable_relation_preserves_schema_shaped_behavior_and_outcome() {
    let context = context();
    let cases = [
        (
            "ocsf_api_activity",
            serde_json::json!({"api":{"operation":"CreateResource"},"status":"Denied"}),
            ["createresource", "denied"],
        ),
        (
            "ocsf_application_lifecycle",
            serde_json::json!({"activity_name":"Install","application":{"type":"Service"},"status":"Success"}),
            ["install", "success"],
        ),
        (
            "ocsf_authentication",
            serde_json::json!({"activity_name":"Logon","auth_protocol":"Kerberos","status":"Failure"}),
            ["logon", "failure"],
        ),
        (
            "ocsf_cloud_resources_inventory_info",
            serde_json::json!({"resource":{"type":"ObjectStorage"},"state":"Public"}),
            ["objectstorage", "public"],
        ),
        (
            "ocsf_datastore_activity",
            serde_json::json!({"activity_name":"Update","databucket":{"type":"Table"},"status":"Success"}),
            ["update", "success"],
        ),
        (
            "ocsf_detection_finding",
            serde_json::json!({"finding":{"type":"MaliciousExecution"},"severity":"High"}),
            ["maliciousexecution", "high"],
        ),
        (
            "ocsf_dns_activity",
            serde_json::json!({"query_type":"TXT","query":{"class":"Internet"},"status":"NoError"}),
            ["txt", "noerror"],
        ),
        (
            "ocsf_email_activity",
            serde_json::json!({"activity_name":"Deliver","message":{"type":"Attachment"},"status":"Quarantined"}),
            ["deliver", "quarantined"],
        ),
        (
            "ocsf_entity_management",
            serde_json::json!({"activity_name":"Create","entity":{"type":"Account"},"status":"Success"}),
            ["create", "success"],
        ),
        (
            "ocsf_event_log_activity",
            serde_json::json!({"event_name":"ScriptExecution","message":"interpreter changed audit policy","status":"Blocked"}),
            ["scriptexecution", "blocked"],
        ),
        (
            "ocsf_ext_livefire_configuration_snapshot",
            serde_json::json!({"snapshot_kind":"FirewallPolicy","subject_kind":"Endpoint","state":"Disabled"}),
            ["firewallpolicy", "disabled"],
        ),
        (
            "ocsf_file_activity",
            serde_json::json!({"activity_name":"Create","file":{"type":"Executable"},"status":"Success"}),
            ["create", "success"],
        ),
        (
            "ocsf_http_activity",
            serde_json::json!({"method":"POST","request":{"type":"Upload"},"status_code":403}),
            ["post", "403"],
        ),
        (
            "ocsf_inventory_info",
            serde_json::json!({"entity":{"type":"Package"},"state":"Outdated"}),
            ["package", "outdated"],
        ),
        (
            "ocsf_network_activity",
            serde_json::json!({"activity_name":"Connect","protocol":"TLS","status":"Blocked"}),
            ["connect", "blocked"],
        ),
        (
            "ocsf_process_activity",
            serde_json::json!({"activity_name":"Launch","process":{"command_line":"interpreter --encoded-input"},"status":"Success"}),
            ["launch", "success"],
        ),
        (
            "ocsf_user_inventory",
            serde_json::json!({"user":{"type":"ServiceAccount"},"state":"Enabled"}),
            ["serviceaccount", "enabled"],
        ),
    ];

    for (relation, typed_event, expected_terms) in cases {
        let typed_event_json = serde_json::to_string(&typed_event).unwrap();
        let output = project(ProjectionInput {
            relation_name: relation,
            event_id: "evt_fixture",
            typed_event_json: &typed_event_json,
            support_ref: "support:fixture",
            context: &context,
        })
        .unwrap();
        let text = output
            .document
            .expect("a supported non-metric relation is searchable")
            .semantic_text
            .to_lowercase();
        for expected in expected_terms {
            assert!(
                text.contains(expected),
                "{relation} lost semantic term {expected}: {text}"
            );
        }
        assert_eq!(
            output.occurrence.terminal_disposition,
            TerminalDisposition::DirectSemanticDocument,
            "{relation}"
        );
    }
}

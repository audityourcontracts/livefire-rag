use std::{
    io::{BufRead, BufReader, BufWriter, Write},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
};

use rag_embedding::EmbeddingProfile;
use rag_index::{BuildScope, FastDocument, FastOccurrence, SourceBinding, write_fast_index};
use rag_provider::{PROTOCOL, provider_ref, tool_ref};
use serde::Deserialize;
use serde_json::{Value, json};

const SNAPSHOT_SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const MAPPING_SHA: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const INDEX_SHA: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
const LOCK_SHA: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

#[derive(Deserialize)]
struct Fixture {
    documents: Vec<FastDocument>,
    occurrences: Vec<FastOccurrence>,
    vectors: Vec<Vec<f32>>,
    pointer_query: String,
    pointer_document_id: String,
    miss_query: String,
}

struct Process {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
}

impl Process {
    fn start() -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_rag-provider"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn provider");
        let stdin = BufWriter::new(child.stdin.take().expect("provider stdin"));
        let stdout = BufReader::new(child.stdout.take().expect("provider stdout"));
        Self {
            child,
            stdin,
            stdout,
        }
    }

    fn exchange(&mut self, request: Value) -> Value {
        let response = self.exchange_raw(request.clone());
        assert_eq!(response.get("id"), request.get("id"));
        assert!(
            response.get("error").is_none(),
            "provider error: {response}"
        );
        assert!(response.get("result").is_some());
        response
    }

    fn exchange_raw(&mut self, request: Value) -> Value {
        serde_json::to_writer(&mut self.stdin, &request).expect("write request");
        self.stdin.write_all(b"\n").expect("frame request");
        self.stdin.flush().expect("flush request");
        let mut line = String::new();
        assert_ne!(self.stdout.read_line(&mut line).expect("read response"), 0);
        let response: Value = serde_json::from_str(&line).expect("response JSON");
        assert_eq!(
            response.get("protocol"),
            Some(&Value::String(PROTOCOL.into()))
        );
        assert_eq!(response.as_object().expect("response object").len(), 3);
        response
    }

    fn finish(mut self) {
        self.stdin.flush().expect("final flush");
        drop(self.stdin);
        let status = self.child.wait().expect("wait provider");
        assert!(status.success(), "provider status {status}");
    }
}

fn request(id: &str, method: &str, params: Value) -> Value {
    json!({
        "protocol":PROTOCOL,
        "id":id,
        "method":method,
        "params":params,
        "context":{"trace_id":"provider-subprocess-test","deadline_unix_ms":9_999_999_999_999_u64}
    })
}

fn reference(id: &str, sha256: &str) -> Value {
    json!({"id":id,"version":"1","sha256":sha256})
}

#[test]
#[ignore = "superseded by packaged SDK bundle/admission lifecycle acceptance"]
fn direct_unadmitted_lifecycle_returns_lexical_pointer_and_explicit_miss() {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../rust-fixtures/provider/lexical-cases.v1.json"
    ))
    .expect("fixture");
    let directory = tempfile::tempdir().expect("tempdir");
    let index_path = directory.path().join("index");
    write_fast_index(
        &index_path,
        SourceBinding {
            snapshot_sha256: SNAPSHOT_SHA.into(),
            mapping_sha256: MAPPING_SHA.into(),
        },
        BuildScope::Sample,
        &fixture.documents,
        &fixture.occurrences,
        &fixture.vectors,
        EmbeddingProfile {
            id: "fixture.embedding".into(),
            version: "1".into(),
            sha256: "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee".into(),
            model: "fixture-not-called-for-lexical".into(),
            dimensions: 2,
            normalization: "l2".into(),
            query_instruction: None,
            query_composition: None,
        },
    )
    .expect("write fixture index");

    let index_ref = reference("fixture.fast-index", INDEX_SHA);
    let source_ref = reference("fixture.snapshot", SNAPSHOT_SHA);
    let mut process = Process::start();
    let handshake = process.exchange(request("1", "handshake", json!({})));
    assert_eq!(handshake.pointer("/result/provider"), Some(&provider_ref()));
    assert_eq!(handshake.pointer("/result/tools/0"), Some(&tool_ref()));

    let opened = process.exchange(request(
        "2",
        "open",
        json!({
            "provider":provider_ref(),
            "tools":[tool_ref()],
            "indexes":[index_ref.clone()],
            "source_snapshots":[source_ref],
            "binding_lock_sha256":LOCK_SHA,
            "query_time_contract":{},
            "limits":{"request_bytes":2_048,"result_bytes":1_048_576,"wall_time_ms":5_000,"memory_bytes":268_435_456,"max_candidates":20},
            "mounts":[{
                "logical_name":"evidence-index",
                "role":"index",
                "component":index_ref,
                "access":"read_only",
                "process_path":index_path.to_str().expect("UTF-8 path")
            }]
        }),
    ));
    let session_id = opened
        .pointer("/result/session_id")
        .and_then(Value::as_str)
        .expect("session id")
        .to_owned();
    assert_eq!(
        opened.pointer("/result/binding_lock_sha256"),
        Some(&Value::String(LOCK_SHA.into()))
    );

    let pointer = process.exchange(request(
        "3",
        "call",
        json!({
            "session_id":session_id,
            "tool":tool_ref(),
            "arguments":{
                "schema_version":"livefire.rag.fast-search.input/1",
                "query":fixture.pointer_query,
                "mode":"lexical",
                "top_n":10
            }
        }),
    ));
    assert_eq!(
        pointer.pointer("/result/output/kind"),
        Some(&Value::String("pointer".into()))
    );
    assert_eq!(
        pointer.pointer("/result/output/candidates/0/document_id"),
        Some(&Value::String(fixture.pointer_document_id))
    );
    assert_eq!(
        pointer.pointer("/result/output/candidates/0/evidence/0/event_id"),
        Some(&Value::String("event-powershell".into()))
    );
    assert!(
        pointer
            .pointer("/result/output/candidates/0/preview")
            .is_none()
    );
    assert_eq!(
        pointer.pointer("/result/output/coverage/status"),
        Some(&Value::String("partial".into()))
    );
    assert!(
        pointer
            .pointer("/result/output/coverage/reason_codes")
            .and_then(Value::as_array)
            .is_some_and(
                |codes| codes.contains(&Value::String("sample_not_corpus_coverage".into()))
            )
    );
    assert!(
        pointer
            .pointer("/result/output/candidates/0/evidence/0/exact_attributes_json")
            .is_none()
    );

    let miss = process.exchange(request(
        "4",
        "call",
        json!({
            "session_id":session_id,
            "tool":tool_ref(),
            "arguments":{
                "schema_version":"livefire.rag.fast-search.input/1",
                "query":fixture.miss_query,
                "mode":"lexical",
                "top_n":10
            }
        }),
    ));
    assert_eq!(
        miss.pointer("/result/output/kind"),
        Some(&Value::String("miss".into()))
    );
    assert_eq!(
        miss.pointer("/result/output/miss/reason"),
        Some(&Value::String("no_ranked_candidates".into()))
    );
    assert!(miss.pointer("/result/output/candidates").is_none());

    let oversized = process.exchange_raw(request(
        "5",
        "call",
        json!({
            "session_id":session_id,
            "tool":tool_ref(),
            "arguments":{
                "schema_version":"livefire.rag.fast-search.input/1",
                "query":"x".repeat(3_000),
                "mode":"lexical",
                "top_n":10
            }
        }),
    ));
    assert_eq!(
        oversized.pointer("/error/code"),
        Some(&Value::String("resource_exhausted".into()))
    );

    let health = process.exchange(request("6", "health", json!({"session_id":session_id})));
    assert_eq!(
        health.pointer("/result/binding_lock_sha256"),
        Some(&Value::String(LOCK_SHA.into()))
    );
    let closed = process.exchange(request("7", "close", json!({"session_id":session_id})));
    assert_eq!(closed.pointer("/result/closed"), Some(&Value::Bool(true)));
    process.finish();
}

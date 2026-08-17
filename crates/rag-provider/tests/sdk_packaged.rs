use std::{fs, path::Path, process::Command};

use serde_json::Value;
use tempfile::tempdir;

#[test]
fn native_bundle_closes_license_and_profiles_without_fake_sbom_entries() {
    let sdk = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../livefire-sdk/specs");
    if !sdk.join("schema-set.lock.json").is_file() {
        eprintln!("adjacent livefire-sdk specs are absent; bundle closure is covered by smoke");
        return;
    }
    let root = tempdir().expect("temporary bundle root");
    let bundle = root.path().join("bundle");
    let status = Command::new(env!("CARGO_BIN_EXE_rag-package-tool"))
        .args([
            "--provider",
            env!("CARGO_BIN_EXE_rag-provider"),
            "--sdk-specs",
        ])
        .arg(&sdk)
        .arg("--out")
        .arg(&bundle)
        .status()
        .expect("run bundle packager");
    assert!(status.success());

    let plugin: Value = serde_json::from_slice(&fs::read(bundle.join("plugin.json")).unwrap())
        .expect("plugin manifest");
    assert!(
        plugin["artifacts"]
            .as_array()
            .unwrap()
            .iter()
            .all(|item| item["kind"] != "sbom")
    );
    let format_versions = plugin["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|item| item["kind"] == "index_format")
        .map(|item| item["component"]["version"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(format_versions, ["0.2.0", "0.3.0"]);
    let v3_descriptor: Value = serde_json::from_slice(
        &fs::read(bundle.join("descriptors/fast-index-format.v3.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        v3_descriptor.pointer("/objects/4/media_type"),
        Some(&Value::String("application/vnd.sqlite3".into()))
    );
    let lock: Value =
        serde_json::from_slice(&fs::read(bundle.join("provider.objects.lock.json")).unwrap())
            .expect("provider object lock");
    let paths = lock["objects"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item["path"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        paths,
        [
            "LICENSE",
            "bin/rag-provider",
            "profiles/fast-lexical-profile.v1.json",
            "profiles/fast-lexical-profile.v2.json",
            "profiles/fast-occurrence-lookup-profile.v1.json",
            "profiles/fast-vector-binary-profile.v1.json",
            "profiles/physical-profile.json",
            "profiles/physical-profile.v3.json",
            "profiles/retrieval-policy.json",
            "profiles/similarity-policy.json",
            "profiles/validator-profile.json",
        ]
    );
    assert_eq!(
        fs::read(bundle.join("LICENSE")).unwrap(),
        fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../LICENSE")).unwrap()
    );
}

#[test]
fn packaged_contract_exposes_only_hydration_references() {
    let sdk = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../livefire-sdk");
    if !sdk.join("Cargo.toml").is_file() {
        eprintln!(
            "adjacent livefire-sdk is absent; packaged acceptance is exercised by tools/run_rust_smoke.py"
        );
        return;
    }
    // The executable integration is intentionally driven by the repository
    // smoke script (and is required by the release checklist):
    // it builds a fresh typed OCSF fixture with the real model, packages the
    // current executable, creates SDK index/admission/binding artifacts, and
    // invokes this provider through the adjacent SDK harness. Keep this test as
    // a contract guard that the output schema exposes hydration refs, not source
    // records or semantic previews.
    let output: Value = serde_json::from_slice(
        &fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../specs/fast-evidence-search.output.v1.schema.json"),
        )
        .expect("output schema"),
    )
    .expect("output schema JSON");
    assert!(
        output
            .pointer("/$defs/evidence_ref/$ref")
            .and_then(Value::as_str)
            .is_some_and(|value| value.ends_with("ocsf-hydration-ref.v1.schema.json"))
    );
    let similar: Value = serde_json::from_slice(
        &fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../specs/fast-evidence-similar.output.v1.schema.json"),
        )
        .expect("similar output schema"),
    )
    .expect("similar output schema JSON");
    assert_eq!(
        similar.pointer("/$defs/candidate/properties/evidence/items/$ref"),
        Some(&Value::String(
            "https://livefire.dev/rag/ocsf-hydration-ref.v1.schema.json".into()
        ))
    );
    let hydration: Value = serde_json::from_slice(
        &fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../specs/ocsf-hydration-ref.v1.schema.json"),
        )
        .expect("hydration schema"),
    )
    .expect("hydration schema JSON");
    assert_eq!(hydration["properties"].get("semantic_text"), None);
    assert_eq!(hydration["properties"].get("exact_attributes_json"), None);
}

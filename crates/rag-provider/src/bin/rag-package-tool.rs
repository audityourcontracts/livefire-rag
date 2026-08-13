use std::{
    fs,
    path::{Path, PathBuf},
};

use clap::Parser;
use rag_provider::{
    FORMAT_ID, PROTOCOL, PROVIDER_ID, VERSION, hydration_ref_schema_ref, index_format_descriptor,
    input_schema_ref, output_schema_ref, physical_profile, physical_profile_ref, retrieval_policy,
    retrieval_policy_ref, tool_descriptor, validator_profile, validator_ref,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

#[derive(Parser)]
#[command(
    about = "Package the Rust evidence.search provider as a content-closed Livefire SDK bundle"
)]
struct Arguments {
    #[arg(long)]
    provider: PathBuf,
    #[arg(long)]
    sdk_specs: PathBuf,
    #[arg(long)]
    out: PathBuf,
    /// SDK artifact target triple for the supplied provider executable.
    #[arg(long, default_value = current_target())]
    target: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = Arguments::parse();
    if arguments.out.exists() {
        return Err("refusing to overwrite bundle output".into());
    }
    let schema_lock: Value = read_json(&arguments.sdk_specs.join("schema-set.lock.json"))?;
    fs::create_dir_all(arguments.out.join("bin"))?;
    fs::create_dir_all(arguments.out.join("descriptors"))?;
    fs::create_dir_all(arguments.out.join("schemas"))?;
    fs::create_dir_all(arguments.out.join("profiles"))?;

    let executable_path = arguments.out.join("bin/rag-provider");
    fs::copy(&arguments.provider, &executable_path)?;
    let executable = artifact(
        &executable_path,
        "bin/rag-provider",
        "application/x-executable",
    )?;
    let mut inventory = Vec::new();
    let mut provider_objects = vec![executable.clone()];
    let target = arguments.target.as_str();
    inventory.push(item(
        component(
            "com.ayc.livefire-rag.fast-evidence-provider.executable",
            VERSION,
            executable["sha256"].as_str().unwrap(),
        ),
        "tool_provider",
        executable.clone(),
        target,
    ));
    let tool = tool_descriptor();
    let descriptor_artifact = write_json_artifact(
        &arguments.out,
        "descriptors/fast-evidence-search.json",
        &tool,
        "application/json",
    )?;
    inventory.push(item(
        tool["tool"].clone(),
        "tool_descriptor",
        descriptor_artifact.clone(),
        target,
    ));
    let format = index_format_descriptor();
    let format_artifact = write_json_artifact(
        &arguments.out,
        "descriptors/fast-index-format.json",
        &format,
        "application/json",
    )?;
    inventory.push(item(
        format["format"].clone(),
        "index_format",
        format_artifact,
        target,
    ));

    let repo_specs = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../specs");
    for name in [
        "fast-evidence-search.input.v1.schema.json",
        "fast-evidence-search.output.v1.schema.json",
        "ocsf-hydration-ref.v1.schema.json",
        "fast-index-manifest.v2.schema.json",
        "fast-document-row.v1.schema.json",
        "fast-occurrence-row.v1.schema.json",
        "fast-build-report.v1.schema.json",
    ] {
        let value = read_json(&repo_specs.join(name))?;
        let reference = component(
            value["$id"].as_str().ok_or("schema id")?,
            "1",
            &canonical_sha256(&value),
        );
        let relative = format!("schemas/{name}");
        let artifact =
            write_json_artifact(&arguments.out, &relative, &value, "application/schema+json")?;
        inventory.push(item(reference, "schema", artifact, target));
    }
    for (name, id, value, reference) in [
        (
            "physical-profile.json",
            "com.ayc.livefire-rag.fast-index-physical-profile",
            physical_profile(),
            physical_profile_ref(),
        ),
        (
            "validator-profile.json",
            "com.ayc.livefire-rag.fast-index-validator",
            validator_profile(),
            validator_ref(),
        ),
        (
            "retrieval-policy.json",
            "com.ayc.livefire-rag.fast-retrieval-policy",
            retrieval_policy(),
            retrieval_policy_ref(),
        ),
    ] {
        let artifact = write_json_artifact(
            &arguments.out,
            &format!("profiles/{name}"),
            &value,
            "application/json",
        )?;
        debug_assert_eq!(reference["id"], id);
        debug_assert_eq!(reference["sha256"], canonical_sha256(&value));
        provider_objects.push(artifact);
    }
    for (name, id) in [
        (
            "fast-vector-binary-profile.v1.json",
            "com.ayc.livefire-rag.fast-vector-binary-profile",
        ),
        (
            "fast-lexical-profile.v1.json",
            "com.ayc.livefire-rag.fast-lexical-profile",
        ),
        (
            "fast-occurrence-lookup-profile.v1.json",
            "com.ayc.livefire-rag.fast-occurrence-lookup-profile",
        ),
    ] {
        let value = read_json(&repo_specs.join(name))?;
        let reference = component(id, "1", &canonical_sha256(&value));
        let artifact = write_json_artifact(
            &arguments.out,
            &format!("profiles/{name}"),
            &value,
            "application/json",
        )?;
        debug_assert_eq!(reference["sha256"], canonical_sha256(&value));
        provider_objects.push(artifact);
    }
    let license_path = arguments.out.join("LICENSE");
    fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../LICENSE"),
        &license_path,
    )?;
    provider_objects.push(artifact(&license_path, "LICENSE", "text/plain")?);
    provider_objects.sort_by(|left, right| left["path"].as_str().cmp(&right["path"].as_str()));
    let provider_lock =
        json!({"schema_version":"livefire.object-lock/1","objects":provider_objects});
    let provider_ref = component(PROVIDER_ID, VERSION, &canonical_sha256(&provider_lock));
    let provider_lock_artifact = write_json_artifact(
        &arguments.out,
        "provider.objects.lock.json",
        &provider_lock,
        "application/vnd.livefire.object-lock+json",
    )?;
    inventory.push(item(
        provider_ref.clone(),
        "tool_provider",
        provider_lock_artifact,
        target,
    ));
    inventory.sort_by(|left, right| {
        left["artifact"]["path"]
            .as_str()
            .cmp(&right["artifact"]["path"].as_str())
    });
    let mut manifest = json!({
        "schema_version":"livefire.plugin/1",
        "plugin":{"id":"com.ayc.livefire-rag.fast-evidence","version":VERSION,"sha256":""},
        "sdk_compatibility":{"tool_protocol":PROTOCOL,"schema_set_sha256":schema_lock["schema_set_sha256"]},
        "artifacts":inventory,
        "entrypoints":{"provider":{"component":provider_ref,"executable":executable}},
        "tools":[{
            "descriptor":tool["tool"],"descriptor_artifact":descriptor_artifact,
            "name":tool["name"],"description":tool["description"],
            "input_schema":input_schema_ref(),"output_schema":output_schema_ref(),
            "effects":["index.read","embedding.loopback","ocsf.hydration_handoff"],
            "required_indexes":[FORMAT_ID]
        }],
        "permissions":{"tool_provider":{"network":["loopback:lmstudio"],"secret_handles":[],"source_mount":"none","index_mount":"read_only","staging_mount":"none","scratch_bytes":268435456}}
    });
    let digest = canonical_sha256_omitting(&manifest, "/plugin/sha256");
    manifest["plugin"]["sha256"] = Value::String(digest);
    write_canonical(arguments.out.join("plugin.json"), &manifest)?;
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"bundle":arguments.out,"plugin":manifest["plugin"],"provider":provider_ref,"tool":tool["tool"],"index_format":format["format"],"hydration_ref":hydration_ref_schema_ref(),"admission":"bundle_only_not_index_admission"})
        )?
    );
    Ok(())
}

fn current_target() -> &'static str {
    if cfg!(all(target_arch = "aarch64", target_os = "macos")) {
        "aarch64-apple-darwin"
    } else if cfg!(all(target_arch = "x86_64", target_os = "macos")) {
        "x86_64-apple-darwin"
    } else if cfg!(all(target_arch = "x86_64", target_os = "linux")) {
        "x86_64-unknown-linux-gnu"
    } else if cfg!(all(target_arch = "aarch64", target_os = "linux")) {
        "aarch64-unknown-linux-gnu"
    } else {
        "unknown-local-target"
    }
}

fn item(component: Value, kind: &str, artifact: Value, target: &str) -> Value {
    json!({"component":component,"kind":kind,"target":target,"artifact":artifact})
}
fn component(id: &str, version: &str, digest: &str) -> Value {
    json!({"id":id,"version":version,"sha256":digest})
}
fn read_json(path: &Path) -> Result<Value, Box<dyn std::error::Error>> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn write_json_artifact(
    root: &Path,
    relative: &str,
    value: &Value,
    media: &str,
) -> Result<Value, Box<dyn std::error::Error>> {
    let path = root.join(relative);
    write_canonical(&path, value)?;
    artifact(&path, relative, media)
}
fn write_canonical(
    path: impl AsRef<Path>,
    value: &Value,
) -> Result<(), Box<dyn std::error::Error>> {
    fs::write(path, serde_json_canonicalizer::to_vec(value)?)?;
    Ok(())
}
fn artifact(path: &Path, relative: &str, media: &str) -> Result<Value, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    Ok(json!({"path":relative,"media_type":media,"sha256":sha256(&bytes),"bytes":bytes.len()}))
}
fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn canonical_sha256(value: &Value) -> String {
    sha256(&serde_json_canonicalizer::to_vec(value).expect("canonical JSON"))
}
fn canonical_sha256_omitting(value: &Value, pointer: &str) -> String {
    let mut copy = value.clone();
    let (parent, field) = pointer.rsplit_once('/').unwrap();
    copy.pointer_mut(parent)
        .unwrap()
        .as_object_mut()
        .unwrap()
        .remove(field);
    canonical_sha256(&copy)
}

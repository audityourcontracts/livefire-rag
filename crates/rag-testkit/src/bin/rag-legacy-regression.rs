//! Non-production runner for the frozen Q1-Q9 and S1/S2 command regression.

use std::{
    collections::BTreeMap,
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    process::ExitCode,
    time::Duration,
};

use clap::{Parser, Subcommand};
use rag_embedding::{IdentifiedEmbedder, LmStudioEmbedder, adapt_model_vector, try_compose_query};
use rag_index::{FastIndex, SearchFilters, SearchMode};
use rag_testkit::legacy_regression::{
    CaseKind, FileReceipt, Suite, evaluate, file_receipt, parse_result_calls, parse_suite,
};
use serde_json::{Value, json};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser)]
#[command(about = "Run or check the frozen command-search regression without Python")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Check captured provider or Rust CLI JSON without contacting a model.
    Check {
        #[arg(long)]
        suite: PathBuf,
        #[arg(long)]
        results: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Run dense search and stored-document similarity against one Rust index.
    Run {
        #[arg(long)]
        suite: PathBuf,
        #[arg(long)]
        index: PathBuf,
        #[arg(long, default_value = "http://127.0.0.1:1234")]
        embedding_endpoint: String,
        /// Bind a diagnostic case to an exact indexed document, for example
        /// `--similar-seed S1=sha256:...`. Supply every S case exactly once.
        #[arg(long, value_name = "CASE_ID=DOCUMENT_ID")]
        similar_seed: Vec<String>,
        /// Maximum duration of each of the bounded Q1-Q9 embedding requests.
        #[arg(long, default_value_t = 120, value_parser = clap::value_parser!(u64).range(1..=300))]
        embedding_timeout_seconds: u64,
        /// New directory for exact requests, results, evaluation, and receipts.
        #[arg(long)]
        out: PathBuf,
    },
}

#[tokio::main]
async fn main() -> ExitCode {
    match execute(Cli::parse()).await {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(error) => {
            eprintln!("rag-legacy-regression: {error}");
            ExitCode::from(2)
        }
    }
}

async fn execute(cli: Cli) -> Result<bool> {
    match cli.command {
        Command::Check {
            suite,
            results,
            out,
        } => check(&suite, &results, &out),
        Command::Run {
            suite,
            index,
            embedding_endpoint,
            similar_seed,
            embedding_timeout_seconds,
            out,
        } => {
            run(
                &suite,
                &index,
                &embedding_endpoint,
                &similar_seed,
                embedding_timeout_seconds,
                &out,
            )
            .await
        }
    }
}

fn check(suite_path: &Path, results_path: &Path, out: &Path) -> Result<bool> {
    let suite_bytes = fs::read(suite_path)?;
    let results_bytes = fs::read(results_path)?;
    let suite = parse_suite(&suite_bytes).map_err(invalid_data)?;
    let calls = parse_result_calls(&results_bytes).map_err(invalid_data)?;
    let report = evaluate(
        &suite,
        receipt(suite_path, &suite_bytes),
        &calls,
        receipt(results_path, &results_bytes),
    )
    .map_err(invalid_data)?;
    let report_bytes = pretty_json(&report)?;
    write_new_file(out, &report_bytes)?;
    let output_receipt = receipt(out, &report_bytes);
    println!(
        "{}",
        serde_json::to_string(&json!({
            "status": report["status"],
            "report": output_receipt
        }))?
    );
    Ok(report["status"] == "pass")
}

async fn run(
    suite_path: &Path,
    index_path: &Path,
    endpoint: &str,
    seed_arguments: &[String],
    timeout_seconds: u64,
    out: &Path,
) -> Result<bool> {
    if out.exists() {
        return Err(invalid_data("run output already exists").into());
    }
    let suite_bytes = fs::read(suite_path)?;
    let suite = parse_suite(&suite_bytes).map_err(invalid_data)?;
    let seeds = parse_seeds(&suite, seed_arguments)?;
    let index = FastIndex::open(index_path)?;
    let embedder = LmStudioEmbedder::with_timeout(
        endpoint,
        &index.manifest.embedding_profile.model,
        Duration::from_secs(timeout_seconds),
    )?;
    let mut requests = Vec::with_capacity(suite.cases.len());
    let mut result_calls = Vec::with_capacity(suite.cases.len());
    let mut returned_models = BTreeMap::<String, String>::new();

    for case in &suite.cases {
        match case.kind {
            CaseKind::Acceptance => {
                let query = case
                    .query
                    .as_deref()
                    .ok_or_else(|| invalid_data("search query"))?;
                requests.push(json!({
                    "case_id": case.case_id,
                    "tool": "cli.search",
                    "query": query,
                    "mode": "dense",
                    "top_n": suite.policy.top_n,
                    "filters": {"relations": [], "time_start_ms": null, "time_end_ms": null}
                }));
                let composed = try_compose_query(&index.manifest.embedding_profile, query)?;
                let mut response = embedder
                    .embed_identified(std::slice::from_ref(&composed))
                    .await?;
                if response.returned_model != index.manifest.embedding_profile.model
                    || response.vectors.len() != 1
                {
                    return Err(invalid_data(format!(
                        "{} returned an unexpected model or vector count",
                        case.case_id
                    ))
                    .into());
                }
                returned_models.insert(case.case_id.clone(), response.returned_model);
                let vector = adapt_model_vector(
                    &index.manifest.embedding_profile,
                    response
                        .vectors
                        .pop()
                        .ok_or_else(|| invalid_data("missing query vector"))?,
                )?;
                index.validate_query_vector(&vector)?;
                let hits = index.search(
                    SearchMode::Dense,
                    query,
                    Some(&vector),
                    &SearchFilters::default(),
                    suite.policy.top_n,
                )?;
                result_calls.push(json!({
                    "case_id": case.case_id,
                    "output": {
                        "schema_version": "livefire.rag.fast-search-result/1",
                        "query": query,
                        "hits": hits
                    }
                }));
            }
            CaseKind::Diagnostic => {
                let document_id = seeds
                    .get(case.case_id.as_str())
                    .ok_or_else(|| invalid_data(format!("missing seed for {}", case.case_id)))?;
                requests.push(json!({
                    "case_id": case.case_id,
                    "tool": "cli.similar",
                    "seed_document_id": document_id,
                    "top_n": suite.policy.top_n,
                    "exclude_seed": true,
                    "filters": {"relations": [], "time_start_ms": null, "time_end_ms": null}
                }));
                let hits = index
                    .similar(
                        document_id,
                        &SearchFilters::default(),
                        suite.policy.top_n,
                        true,
                    )?
                    .ok_or_else(|| {
                        invalid_data(format!(
                            "{} seed document is not in the index: {document_id}",
                            case.case_id
                        ))
                    })?;
                result_calls.push(json!({
                    "case_id": case.case_id,
                    "output": {
                        "schema_version": "livefire.rag.fast-similar-result/1",
                        "seed_document_id": document_id,
                        "seed_excluded": true,
                        "hits": hits
                    }
                }));
            }
        }
    }

    let request_bytes = json_lines(&requests)?;
    let result_bytes = json_lines(&result_calls)?;
    let request_receipt = file_receipt("requests.jsonl", &request_bytes);
    let result_receipt = file_receipt("results.jsonl", &result_bytes);
    let report = evaluate(
        &suite,
        file_receipt(suite_path.to_string_lossy().into_owned(), &suite_bytes),
        &result_calls,
        result_receipt.clone(),
    )
    .map_err(invalid_data)?;
    let report_bytes = pretty_json(&report)?;
    let report_receipt = file_receipt("acceptance.json", &report_bytes);
    let manifest = json!({
        "schema_version": "livefire.rag.legacy-regression-run/1",
        "status": report["status"],
        "non_production_test_tooling": true,
        "suite_id": suite.suite_id,
        "index": {
            "component_sha256": index.manifest.component_sha256,
            "embedding_profile_sha256": index.manifest.embedding_profile.sha256,
            "documents": index.manifest.documents.rows,
            "test_only": index.manifest.test_only
        },
        "execution": {
            "case_order": "suite_order",
            "search_mode": "dense",
            "search_embedding_requests": returned_models.len(),
            "search_embedding_concurrency": 1,
            "top_n": suite.policy.top_n,
            "filters": "none",
            "similarity_uses_stored_seed_vector": true,
            "similarity_excludes_seed": true,
            "returned_models": returned_models
        },
        "receipts": {
            "requests": request_receipt,
            "results": result_receipt,
            "acceptance": report_receipt
        }
    });
    let manifest_bytes = pretty_json(&manifest)?;

    fs::create_dir(out)?;
    fs::write(out.join("requests.jsonl"), request_bytes)?;
    fs::write(out.join("results.jsonl"), result_bytes)?;
    fs::write(out.join("acceptance.json"), report_bytes)?;
    fs::write(out.join("manifest.json"), &manifest_bytes)?;
    let manifest_receipt = file_receipt("manifest.json", &manifest_bytes);
    println!(
        "{}",
        serde_json::to_string(&json!({
            "status": report["status"],
            "run": out,
            "manifest": manifest_receipt
        }))?
    );
    Ok(report["status"] == "pass")
}

fn parse_seeds<'a>(suite: &Suite, arguments: &'a [String]) -> Result<BTreeMap<&'a str, &'a str>> {
    let expected = suite
        .cases
        .iter()
        .filter(|case| case.kind == CaseKind::Diagnostic)
        .map(|case| case.case_id.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    let mut seeds = BTreeMap::new();
    for argument in arguments {
        let (case_id, document_id) = argument
            .split_once('=')
            .filter(|(case_id, document_id)| !case_id.is_empty() && !document_id.is_empty())
            .ok_or_else(|| invalid_data("similar seed must be CASE_ID=DOCUMENT_ID"))?;
        if !expected.contains(case_id) {
            return Err(invalid_data(format!("unknown diagnostic seed case: {case_id}")).into());
        }
        if seeds.insert(case_id, document_id).is_some() {
            return Err(invalid_data(format!("duplicate seed for {case_id}")).into());
        }
    }
    if seeds.len() != expected.len() {
        let missing = expected
            .into_iter()
            .filter(|case_id| !seeds.contains_key(case_id))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(invalid_data(format!("missing diagnostic seeds: {missing}")).into());
    }
    Ok(seeds)
}

fn json_lines(rows: &[Value]) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    for row in rows {
        serde_json::to_writer(&mut bytes, row)?;
        bytes.write_all(b"\n")?;
    }
    Ok(bytes)
}

fn pretty_json(value: &Value) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn receipt(path: &Path, bytes: &[u8]) -> FileReceipt {
    file_receipt(path.to_string_lossy().into_owned(), bytes)
}

fn write_new_file(path: &Path, bytes: &[u8]) -> Result<()> {
    if path.exists() {
        return Err(
            io::Error::new(io::ErrorKind::AlreadyExists, path.display().to_string()).into(),
        );
    }
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, bytes)?;
    Ok(())
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_seed_bindings_are_exact_and_complete() {
        let suite = parse_suite(include_bytes!(
            "../../../../fixtures/provider-poc/acceptance-suite.v1.json"
        ))
        .unwrap();
        let values = ["S1=one".to_owned(), "S2=two".to_owned()];
        let seeds = parse_seeds(&suite, &values).unwrap();
        assert_eq!(seeds["S1"], "one");
        assert_eq!(seeds["S2"], "two");
        assert!(parse_seeds(&suite, &values[..1]).is_err());
    }

    #[test]
    fn json_lines_always_has_a_final_line_feed() {
        let bytes = json_lines(&[json!({"case_id":"Q1"})]).unwrap();
        assert_eq!(bytes.last(), Some(&b'\n'));
    }
}

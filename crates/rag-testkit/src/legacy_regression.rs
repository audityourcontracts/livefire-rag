//! Native evaluation of the frozen command-search regression fixture.
//!
//! These literal checks preserve the historical Q1-Q9 and S1/S2 smoke-test
//! semantics. They identify retrieval candidates only. They do not establish
//! that a result is evidence, malicious, causally related, chronologically
//! ordered, or complete enough for an aggregate answer.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

pub const SUITE_SCHEMA: &str = "livefire.rag.provider-poc-acceptance-suite/1";
pub const REPORT_SCHEMA: &str = "livefire.rag.legacy-regression-report/1";
pub const MAX_CASES: usize = 32;
pub const MAX_CANDIDATES: usize = 100;

pub type Result<T> = std::result::Result<T, String>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Suite {
    pub schema_version: String,
    pub suite_id: String,
    pub description: String,
    pub policy: SuitePolicy,
    pub cases: Vec<Case>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SuitePolicy {
    pub top_n: usize,
    pub require_every_case: bool,
    pub reject_unknown_cases: bool,
    pub candidate_text_fields: Vec<String>,
    pub matcher_semantics: String,
    pub candidate_boundary: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub case_id: String,
    pub kind: CaseKind,
    pub tool: String,
    #[serde(default)]
    pub query: Option<String>,
    #[serde(default)]
    pub seed: Option<String>,
    pub expected_disposition: String,
    #[serde(default)]
    pub behaviors: Vec<Rule>,
    #[serde(default)]
    pub hard_negatives: Vec<Rule>,
    #[serde(default)]
    pub boundary: Option<Boundary>,
    #[serde(default)]
    pub boundary_note: Option<String>,
    #[serde(default)]
    pub diagnostic_note: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaseKind {
    Acceptance,
    Diagnostic,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rule {
    pub label: String,
    #[serde(default)]
    pub mandatory: bool,
    pub max_rank: usize,
    #[serde(default = "one")]
    pub min_matches: usize,
    pub matcher: Matcher,
}

const fn one() -> usize {
    1
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Matcher {
    #[serde(default)]
    pub all: Vec<String>,
    #[serde(default)]
    pub any: Vec<String>,
    #[serde(default)]
    pub none: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Boundary {
    pub label: String,
    pub required_present: BoundaryRule,
    pub required_absent: BoundaryRule,
    pub follow_up: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundaryRule {
    pub max_rank: usize,
    pub matcher: Matcher,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Candidate {
    rank: usize,
    command_id: String,
    preview: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FileReceipt {
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
}

/// Parse and fully validate the frozen suite before any query is run.
pub fn parse_suite(bytes: &[u8]) -> Result<Suite> {
    let suite: Suite = serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
    validate_suite(&suite)?;
    Ok(suite)
}

pub fn validate_suite(suite: &Suite) -> Result<()> {
    if suite.schema_version != SUITE_SCHEMA {
        return Err(format!(
            "unsupported suite schema: {}",
            suite.schema_version
        ));
    }
    if suite.cases.is_empty() || suite.cases.len() > MAX_CASES {
        return Err("suite must contain between 1 and 32 cases".into());
    }
    if suite.policy.top_n == 0 || suite.policy.top_n > MAX_CANDIDATES {
        return Err("suite top_n must be between 1 and 100".into());
    }
    if !suite.policy.require_every_case || !suite.policy.reject_unknown_cases {
        return Err(
            "legacy regression requires every declared case and rejects unknown cases".into(),
        );
    }
    if suite.policy.candidate_text_fields != ["preview", "command_id"] {
        return Err("legacy candidate text fields must be preview and command_id".into());
    }
    let mut ids = BTreeSet::new();
    for case in &suite.cases {
        if !ids.insert(case.case_id.as_str()) {
            return Err(format!("duplicate suite case: {}", case.case_id));
        }
        match (case.tool.as_str(), case.kind, &case.query, &case.seed) {
            ("cli.search", CaseKind::Acceptance, Some(query), None) if !query.is_empty() => {}
            ("cli.similar", CaseKind::Diagnostic, None, Some(seed)) if !seed.is_empty() => {}
            _ => {
                return Err(format!(
                    "{} has an invalid tool or input shape",
                    case.case_id
                ));
            }
        }
        for rule in case.behaviors.iter().chain(&case.hard_negatives) {
            validate_rule(
                &case.case_id,
                rule.max_rank,
                rule.min_matches,
                &rule.matcher,
            )?;
        }
        if let Some(boundary) = &case.boundary {
            validate_rule(
                &case.case_id,
                boundary.required_present.max_rank,
                1,
                &boundary.required_present.matcher,
            )?;
            validate_rule(
                &case.case_id,
                boundary.required_absent.max_rank,
                1,
                &boundary.required_absent.matcher,
            )?;
        }
    }
    Ok(())
}

fn validate_rule(
    case_id: &str,
    max_rank: usize,
    min_matches: usize,
    matcher: &Matcher,
) -> Result<()> {
    if max_rank == 0 || max_rank > MAX_CANDIDATES || min_matches == 0 || min_matches > max_rank {
        return Err(format!("{case_id} has an invalid rank or match bound"));
    }
    if matcher.all.is_empty() && matcher.any.is_empty() && matcher.none.is_empty() {
        return Err(format!("{case_id} has an empty matcher"));
    }
    if matcher
        .all
        .iter()
        .chain(&matcher.any)
        .chain(&matcher.none)
        .any(String::is_empty)
    {
        return Err(format!("{case_id} has an empty matcher term"));
    }
    Ok(())
}

/// Accept either one JSON result envelope or JSON Lines with one case per row.
/// Each case may contain historical provider output or unmodified `rag query`
/// / `rag similar` JSON under `response` or `output`.
pub fn parse_result_calls(bytes: &[u8]) -> Result<Vec<Value>> {
    if let Ok(value) = serde_json::from_slice::<Value>(bytes) {
        if let Some(calls) = value.get("calls").and_then(Value::as_array) {
            return Ok(calls.clone());
        }
        if let Some(calls) = value.as_array() {
            return Ok(calls.clone());
        }
        if value.get("case_id").and_then(Value::as_str).is_some() {
            return Ok(vec![value]);
        }
    }
    let text = std::str::from_utf8(bytes).map_err(|error| error.to_string())?;
    let mut calls = Vec::new();
    for (line_number, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            return Err(format!("results line {} is empty", line_number + 1));
        }
        let row = serde_json::from_str(line)
            .map_err(|error| format!("results line {}: {error}", line_number + 1))?;
        calls.push(row);
    }
    if calls.is_empty() {
        return Err("results contain no calls".into());
    }
    Ok(calls)
}

/// Evaluate every declared case and bind the report to the exact input bytes.
pub fn evaluate(
    suite: &Suite,
    suite_receipt: FileReceipt,
    result_calls: &[Value],
    results_receipt: FileReceipt,
) -> Result<Value> {
    validate_suite(suite)?;
    let declared = suite
        .cases
        .iter()
        .map(|case| (case.case_id.as_str(), case))
        .collect::<BTreeMap<_, _>>();
    let mut observed = BTreeMap::new();
    for call in result_calls {
        let case_id = call
            .get("case_id")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .ok_or("every result call needs a non-empty case_id")?;
        if observed.insert(case_id, call).is_some() {
            return Err(format!("duplicate result for {case_id}"));
        }
    }
    let missing = declared
        .keys()
        .filter(|case_id| !observed.contains_key(**case_id))
        .copied()
        .collect::<Vec<_>>();
    let unknown = observed
        .keys()
        .filter(|case_id| !declared.contains_key(**case_id))
        .copied()
        .collect::<Vec<_>>();
    let mut cases = Vec::new();
    for case in &suite.cases {
        if let Some(call) = observed.get(case.case_id.as_str()) {
            cases.push(evaluate_case(case, call)?);
        }
    }
    let all_passed = cases.iter().all(|row| row["passed"] == true);
    let status = if missing.is_empty() && unknown.is_empty() && all_passed {
        "pass"
    } else {
        "fail"
    };
    let acceptance_total = suite
        .cases
        .iter()
        .filter(|case| case.kind == CaseKind::Acceptance)
        .count();
    let diagnostic_total = suite.cases.len() - acceptance_total;
    let acceptance_passed = cases
        .iter()
        .filter(|row| row["kind"] == "acceptance" && row["passed"] == true)
        .count();
    let diagnostics_passed = cases
        .iter()
        .filter(|row| row["kind"] == "diagnostic" && row["passed"] == true)
        .count();
    Ok(json!({
        "schema_version": REPORT_SCHEMA,
        "suite_id": suite.suite_id,
        "status": status,
        "semantics": {
            "matcher": suite.policy.matcher_semantics,
            "candidate_boundary": suite.policy.candidate_boundary,
            "q1_q9_denominator_only": true,
            "diagnostics_do_not_enlarge_denominator": true
        },
        "receipts": {"suite": suite_receipt, "results": results_receipt},
        "coverage": {
            "declared_cases": suite.cases.len(),
            "observed_cases": observed.len(),
            "missing_case_ids": missing,
            "unknown_case_ids": unknown,
            "complete": missing.is_empty() && unknown.is_empty()
        },
        "summary": {
            "acceptance_passed": acceptance_passed,
            "acceptance_total": acceptance_total,
            "diagnostics_passed": diagnostics_passed,
            "diagnostics_total": diagnostic_total
        },
        "cases": cases
    }))
}

fn evaluate_case(case: &Case, call: &Value) -> Result<Value> {
    let candidates = candidates_for(call, &case.tool, &case.case_id)?;
    let behaviors = case
        .behaviors
        .iter()
        .map(|rule| evaluate_rule(rule, &candidates))
        .collect::<Vec<_>>();
    let hard_negatives = case
        .hard_negatives
        .iter()
        .map(|rule| evaluate_rule(rule, &candidates))
        .collect::<Vec<_>>();
    let mut checks = behaviors
        .iter()
        .map(|row| row["passed"] == true)
        .collect::<Vec<_>>();
    checks.extend(
        case.hard_negatives
            .iter()
            .zip(&hard_negatives)
            .filter(|(rule, _)| rule.mandatory)
            .map(|(_, row)| row["passed"] == true),
    );
    let boundary = case.boundary.as_ref().map(|boundary| {
        let present =
            evaluate_boundary_rule("required_present", &boundary.required_present, &candidates);
        let absent_probe =
            evaluate_boundary_rule("required_absent", &boundary.required_absent, &candidates);
        let absent_passed = absent_probe["observed_matches"] == 0;
        let mut absent = absent_probe;
        absent["passed"] = Value::Bool(absent_passed);
        let passed = present["passed"] == true && absent_passed;
        checks.push(passed);
        json!({
            "label": boundary.label,
            "passed": passed,
            "required_present": present,
            "required_absent": absent,
            "follow_up": boundary.follow_up
        })
    });
    let passed = checks.into_iter().all(|value| value);
    let outcome = if case.expected_disposition == "expected_boundary_failure" {
        if passed {
            "expected_boundary_failure"
        } else {
            "boundary_not_reproduced"
        }
    } else if passed {
        "pass"
    } else {
        "fail"
    };
    Ok(json!({
        "case_id": case.case_id,
        "kind": case.kind,
        "tool": case.tool,
        "expected_disposition": case.expected_disposition,
        "outcome": outcome,
        "passed": passed,
        "returned_candidates": candidates.len(),
        "behaviors": behaviors,
        "hard_negatives": hard_negatives,
        "boundary": boundary
    }))
}

fn evaluate_rule(rule: &Rule, candidates: &[Candidate]) -> Value {
    evaluate_matcher(
        &rule.label,
        rule.max_rank,
        rule.min_matches,
        &rule.matcher,
        candidates,
    )
}

fn evaluate_boundary_rule(label: &str, rule: &BoundaryRule, candidates: &[Candidate]) -> Value {
    evaluate_matcher(label, rule.max_rank, 1, &rule.matcher, candidates)
}

fn evaluate_matcher(
    label: &str,
    max_rank: usize,
    required_matches: usize,
    matcher: &Matcher,
    candidates: &[Candidate],
) -> Value {
    let matched_ranks = candidates
        .iter()
        .filter(|candidate| candidate.rank <= max_rank && matcher_matches(candidate, matcher))
        .map(|candidate| candidate.rank)
        .collect::<Vec<_>>();
    json!({
        "label": label,
        "passed": matched_ranks.len() >= required_matches,
        "matched_ranks": matched_ranks,
        "observed_matches": matched_ranks.len(),
        "required_matches": required_matches,
        "max_rank": max_rank
    })
}

fn matcher_matches(candidate: &Candidate, matcher: &Matcher) -> bool {
    let text = format!("{}\n{}", candidate.preview, candidate.command_id).to_lowercase();
    matcher
        .all
        .iter()
        .all(|term| text.contains(&term.to_lowercase()))
        && (matcher.any.is_empty()
            || matcher
                .any
                .iter()
                .any(|term| text.contains(&term.to_lowercase())))
        && !matcher
            .none
            .iter()
            .any(|term| text.contains(&term.to_lowercase()))
}

fn candidates_for(call: &Value, expected_tool: &str, case_id: &str) -> Result<Vec<Candidate>> {
    let mut response = call
        .get("response")
        .or_else(|| call.get("output"))
        .ok_or_else(|| format!("{case_id}: missing response/output object"))?;
    for _ in 0..4 {
        let recognized_fast_result = matches!(
            response.get("schema_version").and_then(Value::as_str),
            Some("livefire.rag.fast-search-result/1" | "livefire.rag.fast-similar-result/1")
        );
        if response.get("tool").and_then(Value::as_str).is_some() || recognized_fast_result {
            break;
        }
        if let Some(next) = response.get("output").or_else(|| response.get("result")) {
            response = next;
        } else {
            break;
        }
    }
    let candidates = if let Some(schema) = response.get("schema_version").and_then(Value::as_str) {
        let actual_tool = match schema {
            "livefire.rag.fast-search-result/1" => "cli.search",
            "livefire.rag.fast-similar-result/1" => "cli.similar",
            _ => response.get("tool").and_then(Value::as_str).unwrap_or(""),
        };
        if actual_tool != expected_tool {
            return Err(format!(
                "{case_id}: expected {expected_tool}, got {actual_tool}"
            ));
        }
        response
            .get("hits")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("{case_id}: current Rust CLI output needs a hits array"))?
    } else {
        let actual_tool = response.get("tool").and_then(Value::as_str).unwrap_or("");
        if actual_tool != expected_tool {
            return Err(format!(
                "{case_id}: expected {expected_tool}, got {actual_tool}"
            ));
        }
        if response.get("kind") == Some(&Value::String("miss".into())) {
            return Ok(Vec::new());
        }
        if let Some(pointers) = response.get("pointers").and_then(Value::as_array) {
            pointers
        } else {
            let ranking_name = if expected_tool == "cli.search" {
                "semantic_search"
            } else {
                "similar_command"
            };
            let rankings = response
                .get("rankings")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("{case_id}: output needs pointers, hits, or rankings"))?;
            let matches = rankings
                .iter()
                .filter(|row| row.get("ranking").and_then(Value::as_str) == Some(ranking_name))
                .collect::<Vec<_>>();
            if matches.len() != 1 {
                return Err(format!("{case_id}: expected one {ranking_name} ranking"));
            }
            matches[0]
                .get("candidates")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("{case_id}: ranking needs candidates"))?
        }
    };
    if candidates.len() > MAX_CANDIDATES {
        return Err(format!("{case_id}: more than 100 candidates"));
    }
    let mut output = Vec::with_capacity(candidates.len());
    let mut ids = BTreeSet::new();
    for (offset, row) in candidates.iter().enumerate() {
        let rank = row
            .get("rank")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| format!("{case_id}: candidate needs an integer rank"))?;
        if rank != offset + 1 {
            return Err(format!(
                "{case_id}: candidate ranks must be contiguous from 1"
            ));
        }
        let command_id = row
            .get("command_id")
            .or_else(|| row.get("document_id"))
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| format!("{case_id}: candidate needs command_id or document_id"))?
            .to_owned();
        if !ids.insert(command_id.clone()) {
            return Err(format!("{case_id}: duplicate command_id"));
        }
        let preview = row
            .get("preview")
            .or_else(|| row.get("semantic_text"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        output.push(Candidate {
            rank,
            command_id,
            preview,
        });
    }
    Ok(output)
}

#[must_use]
pub fn file_receipt(path: impl Into<String>, bytes: &[u8]) -> FileReceipt {
    FileReceipt {
        path: path.into(),
        sha256: format!("{:x}", Sha256::digest(bytes)),
        bytes: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SUITE: &[u8] = include_bytes!("../../../fixtures/provider-poc/acceptance-suite.v1.json");
    const PASSING: &[u8] =
        include_bytes!("../../../fixtures/provider-poc/synthetic-provider-results.pass.json");

    #[test]
    fn historical_synthetic_fixture_passes_natively() {
        let suite = parse_suite(SUITE).unwrap();
        let calls = parse_result_calls(PASSING).unwrap();
        let report = evaluate(
            &suite,
            file_receipt("suite", SUITE),
            &calls,
            file_receipt("results", PASSING),
        )
        .unwrap();
        assert_eq!(report["status"], "pass");
        assert_eq!(report["summary"]["acceptance_passed"], 9);
        assert_eq!(report["summary"]["diagnostics_passed"], 2);
    }

    #[test]
    fn current_rust_cli_json_is_accepted_without_rewriting() {
        let suite = parse_suite(SUITE).unwrap();
        let q1 = json!({
            "case_id":"Q1",
            "output":{
                "schema_version":"livefire.rag.fast-search-result/1",
                "query":"query",
                "hits":[
                    {"rank":1,"document_id":"log","semantic_text":"cachedGroupPolicySettings ScriptBlockLogging"},
                    {"rank":2,"document_id":"enc","semantic_text":"powershell.exe -enc value"}
                ]
            }
        });
        let candidates = candidates_for(&q1, "cli.search", "Q1").unwrap();
        assert_eq!(candidates[0].command_id, "log");
        assert_eq!(candidates[1].preview, "powershell.exe -enc value");
        assert!(matcher_matches(
            &candidates[0],
            &suite.cases[0].behaviors[0].matcher
        ));
    }

    #[test]
    fn exact_input_receipts_change_when_results_change() {
        let first = file_receipt("results", b"one\n");
        let second = file_receipt("results", b"two\n");
        assert_ne!(first.sha256, second.sha256);
        assert_eq!(first.bytes, 4);
    }
}

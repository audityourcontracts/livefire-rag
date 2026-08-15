use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::json;

use super::{
    BENCHMARK_SELECTION_SCHEMA, ComponentRef, DatasetIdentity, Digest, PipelineError,
    PreparedCorpusManifest, PreparedDocumentRow, Result, canonical_digest, component_digest,
    digest_bytes, require_safe_u64, require_text, validate_prepared_documents,
};

pub const STANDARD_BENCHMARK_SIZES: [u64; 3] = [512, 2_000, 10_000];
const POLICY_SCHEMA: &str = "livefire.rag.benchmark-selection-policy/1";
const SELECTION_KEY_DOMAIN: &str = "livefire.rag.benchmark-selection-key/1";
const CANDIDATE_UNIVERSE_DOMAIN: &[u8] = b"livefire.rag.benchmark-candidate-universe/1\0";
const SELECTION_PREFIX_DOMAIN: &[u8] = b"livefire.rag.benchmark-selection-prefix/1\0";
const MEMBERSHIP_DOMAIN: &[u8] = b"livefire.rag.benchmark-membership/1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkLengthStratum {
    pub id: String,
    pub minimum_utf8_bytes: u64,
    pub maximum_utf8_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkStratumQuota {
    pub relation: String,
    pub length_stratum: String,
    pub documents: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkTargetQuota {
    pub document_count: u64,
    pub quotas: Vec<BenchmarkStratumQuota>,
}

/// The complete, content-bound sampling policy. Quotas are cumulative: the
/// 2,000-document quota for a stratum includes its 512-document selections.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkSelectionPolicy {
    pub schema_version: String,
    pub component_sha256: Digest,
    pub algorithm: String,
    pub selection_seed: String,
    pub length_strata: Vec<BenchmarkLengthStratum>,
    pub targets: Vec<BenchmarkTargetQuota>,
}

impl BenchmarkSelectionPolicy {
    pub fn validate(&self, dataset: &DatasetIdentity) -> Result<()> {
        dataset.validate()?;
        if self.schema_version != POLICY_SCHEMA || self.algorithm != "staged_stratified_sha256_v1" {
            return Err(PipelineError::Invalid("benchmark selection policy schema"));
        }
        require_text(&self.selection_seed)?;
        validate_length_strata(&self.length_strata)?;
        if self.targets.len() != STANDARD_BENCHMARK_SIZES.len() {
            return Err(PipelineError::Invalid("standard benchmark target sizes"));
        }

        let expected_cells = dataset
            .included_relations
            .iter()
            .flat_map(|relation| {
                self.length_strata
                    .iter()
                    .map(move |stratum| (relation.as_str(), stratum.id.as_str()))
            })
            .collect::<Vec<_>>();
        let mut previous = BTreeMap::<(&str, &str), u64>::new();
        for (target, expected_size) in self.targets.iter().zip(STANDARD_BENCHMARK_SIZES) {
            require_safe_u64(target.document_count)?;
            if target.document_count != expected_size || target.quotas.len() != expected_cells.len()
            {
                return Err(PipelineError::Invalid("standard benchmark target sizes"));
            }
            let actual_cells = target
                .quotas
                .iter()
                .map(|quota| (quota.relation.as_str(), quota.length_stratum.as_str()))
                .collect::<Vec<_>>();
            if actual_cells != expected_cells {
                return Err(PipelineError::Invalid("benchmark quota coverage or order"));
            }
            let mut total = 0_u64;
            for quota in &target.quotas {
                require_safe_u64(quota.documents)?;
                let key = (quota.relation.as_str(), quota.length_stratum.as_str());
                if quota.documents < previous.get(&key).copied().unwrap_or(0) {
                    return Err(PipelineError::Invalid("benchmark quota nesting"));
                }
                total = total
                    .checked_add(quota.documents)
                    .ok_or(PipelineError::Invalid("benchmark quota total"))?;
                previous.insert(key, quota.documents);
            }
            if total != expected_size {
                return Err(PipelineError::Invalid("benchmark quota total"));
            }
        }
        if self.component_sha256 != component_digest(self)? {
            return Err(PipelineError::Invalid("benchmark policy component digest"));
        }
        Ok(())
    }

    pub fn seal(&mut self, dataset: &DatasetIdentity) -> Result<()> {
        self.component_sha256 = component_digest(self)?;
        self.validate(dataset)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkSelectionCandidate {
    pub document_id: String,
    pub document_sha256: Digest,
    pub semantic_text_sha256: Digest,
    pub semantic_text_utf8_bytes: u64,
    pub primary_relation: String,
}

impl BenchmarkSelectionCandidate {
    pub fn from_prepared(row: &PreparedDocumentRow) -> Result<Self> {
        row.validate()?;
        let semantic_text_utf8_bytes = u64::try_from(row.semantic_text.len())
            .map_err(|_| PipelineError::Invalid("semantic text byte length"))?;
        require_safe_u64(semantic_text_utf8_bytes)?;
        Ok(Self {
            document_id: row.document_id.clone(),
            document_sha256: row.document_sha256.clone(),
            semantic_text_sha256: row.semantic_text_sha256.clone(),
            semantic_text_utf8_bytes,
            primary_relation: row.primary_relation.clone(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkSelectionRow {
    pub selection_rank: u64,
    pub document_id: String,
    pub document_sha256: Digest,
    pub semantic_text_sha256: Digest,
    pub semantic_text_utf8_bytes: u64,
    pub primary_relation: String,
    pub length_stratum: String,
    pub selection_key_sha256: Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkPreparedCorpusIdentity {
    pub prepared_corpus_sha256: Digest,
    pub dataset_sha256: Digest,
    pub projection_policy_sha256: Digest,
    pub document_count: u64,
    pub occurrence_count: u64,
    pub document_order_sha256: Digest,
    pub embedding_input_order_sha256: Digest,
    pub selected_documents_sha256: Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkPublishedCorpus {
    pub document_count: u64,
    pub selection_prefix_sha256: Digest,
    pub prepared_corpus: BenchmarkPreparedCorpusIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkSelectionTarget {
    pub document_count: u64,
    pub quotas: Vec<BenchmarkStratumQuota>,
    pub selection_prefix_sha256: Digest,
    pub prepared_corpus: BenchmarkPreparedCorpusIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkSelectionManifest {
    pub schema_version: String,
    pub component_sha256: Digest,
    pub dataset: DatasetIdentity,
    pub dataset_sha256: Digest,
    pub projection_policy: ComponentRef,
    pub selection_implementation: ComponentRef,
    pub selection_policy: BenchmarkSelectionPolicy,
    pub candidate_count: u64,
    pub candidate_universe_sha256: Digest,
    pub selection_count: u64,
    pub selection_order_sha256: Digest,
    pub selections: Vec<BenchmarkSelectionRow>,
    pub targets: Vec<BenchmarkSelectionTarget>,
}

impl BenchmarkSelectionManifest {
    pub fn validate(&self) -> Result<()> {
        for value in [self.candidate_count, self.selection_count] {
            require_safe_u64(value)?;
        }
        if self.schema_version != BENCHMARK_SELECTION_SCHEMA
            || self.dataset_sha256 != canonical_digest(&self.dataset)?
            || self.selection_count != STANDARD_BENCHMARK_SIZES[2]
            || u64::try_from(self.selections.len()).ok() != Some(self.selection_count)
            || self.candidate_count < self.selection_count
        {
            return Err(PipelineError::Invalid(
                "benchmark selection manifest binding",
            ));
        }
        self.dataset.validate()?;
        self.projection_policy.validate()?;
        self.selection_implementation.validate()?;
        self.selection_policy.validate(&self.dataset)?;
        validate_selection_rows(
            &self.selections,
            &self.dataset_sha256,
            &self.projection_policy.sha256,
            &self.selection_policy,
        )?;
        if self.selection_order_sha256 != selection_prefix_digest(&self.selections) {
            return Err(PipelineError::Invalid("benchmark selection order digest"));
        }
        if self.targets.len() != STANDARD_BENCHMARK_SIZES.len() {
            return Err(PipelineError::Invalid("benchmark target coverage"));
        }
        for ((target, quota), expected_size) in self
            .targets
            .iter()
            .zip(&self.selection_policy.targets)
            .zip(STANDARD_BENCHMARK_SIZES)
        {
            if target.document_count != expected_size
                || quota.document_count != expected_size
                || target.quotas != quota.quotas
            {
                return Err(PipelineError::Invalid("benchmark target binding"));
            }
            let end = usize::try_from(expected_size)
                .map_err(|_| PipelineError::Invalid("benchmark target size"))?;
            let prefix = &self.selections[..end];
            if target.selection_prefix_sha256 != selection_prefix_digest(prefix) {
                return Err(PipelineError::Invalid("benchmark target prefix digest"));
            }
            validate_prefix_quotas(prefix, &target.quotas)?;
            validate_prepared_identity(
                &target.prepared_corpus,
                expected_size,
                &self.dataset_sha256,
                &self.projection_policy.sha256,
                prefix,
            )?;
        }
        if self.component_sha256 != component_digest(self)? {
            return Err(PipelineError::Invalid(
                "benchmark selection component digest",
            ));
        }
        Ok(())
    }

    /// Rebuild the selection from the admitted candidate universe. This proves
    /// that no search result, query scenario, or later evaluation label could
    /// have influenced the published ranks.
    pub fn validate_against_candidates(
        &self,
        candidates: &[BenchmarkSelectionCandidate],
    ) -> Result<()> {
        self.validate()?;
        let (candidate_count, candidate_universe_sha256, selections) = select_benchmark_documents(
            &self.dataset,
            &self.projection_policy,
            &self.selection_policy,
            candidates,
        )?;
        if candidate_count != self.candidate_count
            || candidate_universe_sha256 != self.candidate_universe_sha256
            || selections != self.selections
        {
            return Err(PipelineError::Invalid(
                "benchmark candidate universe binding",
            ));
        }
        Ok(())
    }

    /// Load each standard prepared corpus and prove that its normal
    /// document-ID order contains exactly the selected prefix. Prepared object
    /// paths, sizes, and hashes are validated by the prepared manifest.
    pub fn validate_prepared_corpora(
        &self,
        prepared_corpora: &[(&PreparedCorpusManifest, &[PreparedDocumentRow])],
    ) -> Result<()> {
        self.validate()?;
        if prepared_corpora.len() != self.targets.len() {
            return Err(PipelineError::Invalid("benchmark prepared corpus coverage"));
        }
        for (target, (prepared, documents)) in
            self.targets.iter().zip(prepared_corpora.iter().copied())
        {
            let actual = bind_benchmark_prepared_corpus(
                target.document_count,
                &self.selections,
                &self.dataset,
                &self.projection_policy,
                prepared,
                documents,
            )?;
            if actual.document_count != target.document_count
                || actual.selection_prefix_sha256 != target.selection_prefix_sha256
                || actual.prepared_corpus != target.prepared_corpus
            {
                return Err(PipelineError::Invalid("benchmark prepared corpus identity"));
            }
        }
        Ok(())
    }
}

/// Select all standard nested benchmark documents. Candidates contain only
/// source-derived identity, relation, and text-length fields; the API has no
/// query, relevance, or result inputs.
pub fn select_benchmark_documents(
    dataset: &DatasetIdentity,
    projection_policy: &ComponentRef,
    policy: &BenchmarkSelectionPolicy,
    candidates: &[BenchmarkSelectionCandidate],
) -> Result<(u64, Digest, Vec<BenchmarkSelectionRow>)> {
    dataset.validate()?;
    projection_policy.validate()?;
    policy.validate(dataset)?;
    let candidate_count = u64::try_from(candidates.len())
        .map_err(|_| PipelineError::Invalid("benchmark candidate count"))?;
    require_safe_u64(candidate_count)?;
    if candidate_count < STANDARD_BENCHMARK_SIZES[2] {
        return Err(PipelineError::Invalid("benchmark candidate capacity"));
    }
    let dataset_sha256 = canonical_digest(dataset)?;
    let mut ids = BTreeSet::new();
    let mut keys = BTreeSet::new();
    let mut groups = BTreeMap::<(String, String), Vec<BenchmarkSelectionRow>>::new();
    for candidate in candidates {
        validate_candidate(candidate, dataset)?;
        if !ids.insert(candidate.document_id.as_str()) {
            return Err(PipelineError::Invalid("duplicate benchmark candidate id"));
        }
        let stratum =
            find_length_stratum(&policy.length_strata, candidate.semantic_text_utf8_bytes)?;
        let key = selection_key(
            &dataset_sha256,
            &projection_policy.sha256,
            &policy.component_sha256,
            &policy.selection_seed,
            candidate,
            &stratum.id,
        )?;
        if !keys.insert(key.clone()) {
            return Err(PipelineError::Invalid("duplicate benchmark selection key"));
        }
        groups
            .entry((candidate.primary_relation.clone(), stratum.id.clone()))
            .or_default()
            .push(BenchmarkSelectionRow {
                selection_rank: 0,
                document_id: candidate.document_id.clone(),
                document_sha256: candidate.document_sha256.clone(),
                semantic_text_sha256: candidate.semantic_text_sha256.clone(),
                semantic_text_utf8_bytes: candidate.semantic_text_utf8_bytes,
                primary_relation: candidate.primary_relation.clone(),
                length_stratum: stratum.id.clone(),
                selection_key_sha256: key,
            });
    }
    for rows in groups.values_mut() {
        rows.sort_by(|left, right| {
            left.selection_key_sha256
                .cmp(&right.selection_key_sha256)
                .then_with(|| left.document_id.cmp(&right.document_id))
        });
    }
    let candidate_universe_sha256 = candidate_universe_digest(candidates)?;
    let mut selected = Vec::with_capacity(STANDARD_BENCHMARK_SIZES[2] as usize);
    let mut previous = BTreeMap::<(String, String), usize>::new();
    for target in &policy.targets {
        let mut additions = Vec::new();
        for quota in &target.quotas {
            let cell = (quota.relation.clone(), quota.length_stratum.clone());
            let start = previous.get(&cell).copied().unwrap_or(0);
            let end = usize::try_from(quota.documents)
                .map_err(|_| PipelineError::Invalid("benchmark quota size"))?;
            let Some(candidates) = groups.get(&cell) else {
                if start == 0 && end == 0 {
                    previous.insert(cell, 0);
                    continue;
                }
                return Err(PipelineError::Invalid(
                    "benchmark stratum has no candidates",
                ));
            };
            if end > candidates.len() || start > end {
                return Err(PipelineError::Invalid("benchmark stratum capacity"));
            }
            additions.extend_from_slice(&candidates[start..end]);
            previous.insert(cell, end);
        }
        additions.sort_by(|left, right| {
            left.selection_key_sha256
                .cmp(&right.selection_key_sha256)
                .then_with(|| left.document_id.cmp(&right.document_id))
        });
        for mut row in additions {
            row.selection_rank = u64::try_from(selected.len())
                .map_err(|_| PipelineError::Invalid("benchmark selection rank"))?;
            selected.push(row);
        }
        if u64::try_from(selected.len()).ok() != Some(target.document_count) {
            return Err(PipelineError::Invalid("benchmark target selection count"));
        }
    }
    Ok((candidate_count, candidate_universe_sha256, selected))
}

/// Bind one selected prefix to the ordinary prepared-corpus manifest and its
/// exact document rows. The prepared rows may use their normal document-ID
/// order; membership is compared independently from selection rank.
pub fn bind_benchmark_prepared_corpus(
    document_count: u64,
    selections: &[BenchmarkSelectionRow],
    dataset: &DatasetIdentity,
    projection_policy: &ComponentRef,
    prepared: &PreparedCorpusManifest,
    documents: &[PreparedDocumentRow],
) -> Result<BenchmarkPublishedCorpus> {
    if !STANDARD_BENCHMARK_SIZES.contains(&document_count) {
        return Err(PipelineError::Invalid("standard benchmark target size"));
    }
    prepared.validate()?;
    validate_prepared_documents(prepared, documents)?;
    let end = usize::try_from(document_count)
        .map_err(|_| PipelineError::Invalid("benchmark target size"))?;
    let prefix = selections
        .get(..end)
        .ok_or(PipelineError::Invalid("benchmark selection prefix"))?;
    if prepared.dataset != *dataset
        || prepared.projection_policy != *projection_policy
        || prepared.document_count != document_count
    {
        return Err(PipelineError::Invalid("benchmark prepared corpus binding"));
    }
    let selected_membership = selected_membership_digest(prefix);
    let prepared_membership = prepared_membership_digest(documents);
    if selected_membership != prepared_membership {
        return Err(PipelineError::Invalid(
            "benchmark prepared document membership",
        ));
    }
    Ok(BenchmarkPublishedCorpus {
        document_count,
        selection_prefix_sha256: selection_prefix_digest(prefix),
        prepared_corpus: BenchmarkPreparedCorpusIdentity {
            prepared_corpus_sha256: prepared.component_sha256.clone(),
            dataset_sha256: canonical_digest(dataset)?,
            projection_policy_sha256: projection_policy.sha256.clone(),
            document_count,
            occurrence_count: prepared.occurrence_count,
            document_order_sha256: prepared.document_order_sha256.clone(),
            embedding_input_order_sha256: prepared.embedding_input_order_sha256.clone(),
            selected_documents_sha256: selected_membership,
        },
    })
}

#[allow(clippy::too_many_arguments)]
pub fn build_benchmark_selection_manifest(
    dataset: DatasetIdentity,
    projection_policy: ComponentRef,
    selection_implementation: ComponentRef,
    selection_policy: BenchmarkSelectionPolicy,
    candidate_count: u64,
    candidate_universe_sha256: Digest,
    selections: Vec<BenchmarkSelectionRow>,
    published_corpora: Vec<BenchmarkPublishedCorpus>,
) -> Result<BenchmarkSelectionManifest> {
    if published_corpora.len() != STANDARD_BENCHMARK_SIZES.len() {
        return Err(PipelineError::Invalid(
            "benchmark published corpus coverage",
        ));
    }
    let targets = published_corpora
        .into_iter()
        .zip(&selection_policy.targets)
        .map(|(published, quota)| BenchmarkSelectionTarget {
            document_count: published.document_count,
            quotas: quota.quotas.clone(),
            selection_prefix_sha256: published.selection_prefix_sha256,
            prepared_corpus: published.prepared_corpus,
        })
        .collect();
    let mut manifest = BenchmarkSelectionManifest {
        schema_version: BENCHMARK_SELECTION_SCHEMA.into(),
        component_sha256: digest_bytes(b"unsealed benchmark selection"),
        dataset_sha256: canonical_digest(&dataset)?,
        dataset,
        projection_policy,
        selection_implementation,
        selection_policy,
        candidate_count,
        candidate_universe_sha256,
        selection_count: u64::try_from(selections.len())
            .map_err(|_| PipelineError::Invalid("benchmark selection count"))?,
        selection_order_sha256: selection_prefix_digest(&selections),
        selections,
        targets,
    };
    manifest.component_sha256 = component_digest(&manifest)?;
    manifest.validate()?;
    Ok(manifest)
}

fn validate_length_strata(strata: &[BenchmarkLengthStratum]) -> Result<()> {
    if strata.is_empty() {
        return Err(PipelineError::Invalid("benchmark length strata"));
    }
    let mut ids = BTreeSet::new();
    let mut next = 0_u64;
    for (index, stratum) in strata.iter().enumerate() {
        require_text(&stratum.id)?;
        require_safe_u64(stratum.minimum_utf8_bytes)?;
        if !ids.insert(&stratum.id) || stratum.minimum_utf8_bytes != next {
            return Err(PipelineError::Invalid("benchmark length strata"));
        }
        match stratum.maximum_utf8_bytes {
            Some(maximum) => {
                require_safe_u64(maximum)?;
                if maximum < stratum.minimum_utf8_bytes || index + 1 == strata.len() {
                    return Err(PipelineError::Invalid("benchmark length strata"));
                }
                next = maximum
                    .checked_add(1)
                    .ok_or(PipelineError::Invalid("benchmark length strata"))?;
            }
            None if index + 1 == strata.len() => {}
            None => return Err(PipelineError::Invalid("benchmark length strata")),
        }
    }
    Ok(())
}

fn validate_candidate(
    candidate: &BenchmarkSelectionCandidate,
    dataset: &DatasetIdentity,
) -> Result<()> {
    require_text(&candidate.document_id)?;
    require_text(&candidate.primary_relation)?;
    require_safe_u64(candidate.semantic_text_utf8_bytes)?;
    if candidate.semantic_text_utf8_bytes == 0
        || !dataset
            .included_relations
            .contains(&candidate.primary_relation)
    {
        return Err(PipelineError::Invalid("benchmark candidate fields"));
    }
    Ok(())
}

fn find_length_stratum(
    strata: &[BenchmarkLengthStratum],
    bytes: u64,
) -> Result<&BenchmarkLengthStratum> {
    strata
        .iter()
        .find(|stratum| {
            bytes >= stratum.minimum_utf8_bytes
                && stratum
                    .maximum_utf8_bytes
                    .is_none_or(|maximum| bytes <= maximum)
        })
        .ok_or(PipelineError::Invalid("semantic text length stratum"))
}

fn selection_key(
    dataset_sha256: &Digest,
    projection_policy_sha256: &Digest,
    policy_sha256: &Digest,
    selection_seed: &str,
    candidate: &BenchmarkSelectionCandidate,
    length_stratum: &str,
) -> Result<Digest> {
    canonical_digest(&json!({
        "schema_version": SELECTION_KEY_DOMAIN,
        "dataset_sha256": dataset_sha256,
        "projection_policy_sha256": projection_policy_sha256,
        "selection_policy_sha256": policy_sha256,
        "selection_seed": selection_seed,
        "document_id": candidate.document_id,
        "document_sha256": candidate.document_sha256,
        "semantic_text_sha256": candidate.semantic_text_sha256,
        "semantic_text_utf8_bytes": candidate.semantic_text_utf8_bytes,
        "primary_relation": candidate.primary_relation,
        "length_stratum": length_stratum,
    }))
}

fn validate_selection_rows(
    rows: &[BenchmarkSelectionRow],
    dataset_sha256: &Digest,
    projection_policy_sha256: &Digest,
    policy: &BenchmarkSelectionPolicy,
) -> Result<()> {
    let mut ids = BTreeSet::new();
    let mut keys = BTreeSet::new();
    let mut stage_start = 0_usize;
    for (rank, row) in rows.iter().enumerate() {
        require_safe_u64(row.selection_rank)?;
        require_safe_u64(row.semantic_text_utf8_bytes)?;
        if usize::try_from(row.selection_rank).ok() != Some(rank)
            || !ids.insert(&row.document_id)
            || !keys.insert(&row.selection_key_sha256)
        {
            return Err(PipelineError::Invalid(
                "benchmark selection rank or identity",
            ));
        }
        let stratum = find_length_stratum(&policy.length_strata, row.semantic_text_utf8_bytes)?;
        let candidate = BenchmarkSelectionCandidate {
            document_id: row.document_id.clone(),
            document_sha256: row.document_sha256.clone(),
            semantic_text_sha256: row.semantic_text_sha256.clone(),
            semantic_text_utf8_bytes: row.semantic_text_utf8_bytes,
            primary_relation: row.primary_relation.clone(),
        };
        if stratum.id != row.length_stratum
            || selection_key(
                dataset_sha256,
                projection_policy_sha256,
                &policy.component_sha256,
                &policy.selection_seed,
                &candidate,
                &row.length_stratum,
            )? != row.selection_key_sha256
        {
            return Err(PipelineError::Invalid("benchmark selection key"));
        }
    }
    for size in STANDARD_BENCHMARK_SIZES {
        let stage_end = usize::try_from(size)
            .map_err(|_| PipelineError::Invalid("benchmark selection stage"))?;
        if rows[stage_start..stage_end].windows(2).any(|pair| {
            (
                pair[0].selection_key_sha256.as_str(),
                pair[0].document_id.as_str(),
            ) >= (
                pair[1].selection_key_sha256.as_str(),
                pair[1].document_id.as_str(),
            )
        }) {
            return Err(PipelineError::Invalid("benchmark selection stage order"));
        }
        stage_start = stage_end;
    }
    Ok(())
}

fn validate_prefix_quotas(
    rows: &[BenchmarkSelectionRow],
    quotas: &[BenchmarkStratumQuota],
) -> Result<()> {
    let actual = rows.iter().fold(BTreeMap::new(), |mut counts, row| {
        *counts
            .entry((row.primary_relation.as_str(), row.length_stratum.as_str()))
            .or_insert(0_u64) += 1;
        counts
    });
    for quota in quotas {
        if actual
            .get(&(quota.relation.as_str(), quota.length_stratum.as_str()))
            .copied()
            .unwrap_or(0)
            != quota.documents
        {
            return Err(PipelineError::Invalid("benchmark target quota"));
        }
    }
    Ok(())
}

fn validate_prepared_identity(
    identity: &BenchmarkPreparedCorpusIdentity,
    expected_size: u64,
    dataset_sha256: &Digest,
    projection_policy_sha256: &Digest,
    selections: &[BenchmarkSelectionRow],
) -> Result<()> {
    for value in [identity.document_count, identity.occurrence_count] {
        require_safe_u64(value)?;
    }
    if identity.document_count != expected_size
        || identity.dataset_sha256 != *dataset_sha256
        || identity.projection_policy_sha256 != *projection_policy_sha256
        || identity.selected_documents_sha256 != selected_membership_digest(selections)
    {
        return Err(PipelineError::Invalid("benchmark prepared corpus identity"));
    }
    Ok(())
}

fn candidate_universe_digest(candidates: &[BenchmarkSelectionCandidate]) -> Result<Digest> {
    let mut ordered = candidates.iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| left.document_id.cmp(&right.document_id));
    let mut bytes = CANDIDATE_UNIVERSE_DOMAIN.to_vec();
    for candidate in ordered {
        bytes.extend_from_slice(
            canonical_digest(&json!({
                "document_id": candidate.document_id,
                "document_sha256": candidate.document_sha256,
                "semantic_text_sha256": candidate.semantic_text_sha256,
                "semantic_text_utf8_bytes": candidate.semantic_text_utf8_bytes,
                "primary_relation": candidate.primary_relation,
            }))?
            .as_str()
            .as_bytes(),
        );
        bytes.push(0);
    }
    Ok(digest_bytes(&bytes))
}

fn selection_prefix_digest(rows: &[BenchmarkSelectionRow]) -> Digest {
    let mut bytes = SELECTION_PREFIX_DOMAIN.to_vec();
    for row in rows {
        bytes.extend_from_slice(row.selection_rank.to_string().as_bytes());
        bytes.push(0);
        for value in [
            row.document_id.as_str(),
            row.document_sha256.as_str(),
            row.semantic_text_sha256.as_str(),
            row.selection_key_sha256.as_str(),
        ] {
            bytes.extend_from_slice(value.as_bytes());
            bytes.push(0);
        }
    }
    digest_bytes(&bytes)
}

fn selected_membership_digest(rows: &[BenchmarkSelectionRow]) -> Digest {
    let mut ordered = rows.iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| left.document_id.cmp(&right.document_id));
    let mut bytes = MEMBERSHIP_DOMAIN.to_vec();
    for row in ordered {
        for value in [
            row.document_id.as_str(),
            row.document_sha256.as_str(),
            row.semantic_text_sha256.as_str(),
        ] {
            bytes.extend_from_slice(value.as_bytes());
            bytes.push(0);
        }
    }
    digest_bytes(&bytes)
}

fn prepared_membership_digest(rows: &[PreparedDocumentRow]) -> Digest {
    let mut bytes = MEMBERSHIP_DOMAIN.to_vec();
    for row in rows {
        for value in [
            row.document_id.as_str(),
            row.document_sha256.as_str(),
            row.semantic_text_sha256.as_str(),
        ] {
            bytes.extend_from_slice(value.as_bytes());
            bytes.push(0);
        }
    }
    digest_bytes(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(label: &str) -> Digest {
        digest_bytes(label.as_bytes())
    }

    fn component(id: &str) -> ComponentRef {
        ComponentRef {
            id: id.into(),
            version: "1".into(),
            sha256: digest(id),
        }
    }

    fn dataset() -> DatasetIdentity {
        DatasetIdentity {
            id: "benchmark-fixture".into(),
            version: "1".into(),
            source_snapshot: component("snapshot"),
            mapping: component("mapping"),
            included_relations: vec!["relation_a".into(), "relation_b".into()],
            excluded_relations: vec!["network".into()],
            structured_only_relations: vec![],
        }
    }

    fn policy() -> BenchmarkSelectionPolicy {
        let cells = [
            ("relation_a", "short"),
            ("relation_a", "long"),
            ("relation_b", "short"),
            ("relation_b", "long"),
        ];
        let target = |document_count, per_cell| BenchmarkTargetQuota {
            document_count,
            quotas: cells
                .iter()
                .map(|(relation, length_stratum)| BenchmarkStratumQuota {
                    relation: (*relation).into(),
                    length_stratum: (*length_stratum).into(),
                    documents: per_cell,
                })
                .collect(),
        };
        let mut policy = BenchmarkSelectionPolicy {
            schema_version: POLICY_SCHEMA.into(),
            component_sha256: digest("unsealed"),
            algorithm: "staged_stratified_sha256_v1".into(),
            selection_seed: "checked-in-v1".into(),
            length_strata: vec![
                BenchmarkLengthStratum {
                    id: "short".into(),
                    minimum_utf8_bytes: 0,
                    maximum_utf8_bytes: Some(9),
                },
                BenchmarkLengthStratum {
                    id: "long".into(),
                    minimum_utf8_bytes: 10,
                    maximum_utf8_bytes: None,
                },
            ],
            targets: vec![target(512, 128), target(2_000, 500), target(10_000, 2_500)],
        };
        policy.seal(&dataset()).unwrap();
        policy
    }

    fn candidates() -> Vec<BenchmarkSelectionCandidate> {
        let mut result = Vec::new();
        for relation in ["relation_a", "relation_b"] {
            for (length_name, bytes) in [("short", 5_u64), ("long", 15_u64)] {
                for ordinal in 0..3_000_u64 {
                    let id = format!("{relation}-{length_name}-{ordinal:04}");
                    result.push(BenchmarkSelectionCandidate {
                        document_id: id.clone(),
                        document_sha256: digest(&format!("document:{id}")),
                        semantic_text_sha256: digest(&format!("input:{id}")),
                        semantic_text_utf8_bytes: bytes,
                        primary_relation: relation.into(),
                    });
                }
            }
        }
        result
    }

    fn published(rows: &[BenchmarkSelectionRow], document_count: u64) -> BenchmarkPublishedCorpus {
        let end = usize::try_from(document_count).unwrap();
        let prefix = &rows[..end];
        BenchmarkPublishedCorpus {
            document_count,
            selection_prefix_sha256: selection_prefix_digest(prefix),
            prepared_corpus: BenchmarkPreparedCorpusIdentity {
                prepared_corpus_sha256: digest(&format!("prepared:{document_count}")),
                dataset_sha256: canonical_digest(&dataset()).unwrap(),
                projection_policy_sha256: component("projection").sha256,
                document_count,
                occurrence_count: document_count * 2,
                document_order_sha256: digest(&format!("document-order:{document_count}")),
                embedding_input_order_sha256: digest(&format!("input-order:{document_count}")),
                selected_documents_sha256: selected_membership_digest(prefix),
            },
        }
    }

    fn manifest() -> (BenchmarkSelectionManifest, Vec<BenchmarkSelectionCandidate>) {
        let candidates = candidates();
        let policy = policy();
        let (candidate_count, universe, rows) =
            select_benchmark_documents(&dataset(), &component("projection"), &policy, &candidates)
                .unwrap();
        let publications = STANDARD_BENCHMARK_SIZES
            .map(|size| published(&rows, size))
            .into_iter()
            .collect();
        let manifest = build_benchmark_selection_manifest(
            dataset(),
            component("projection"),
            component("benchmark-selector"),
            policy,
            candidate_count,
            universe,
            rows,
            publications,
        )
        .unwrap();
        (manifest, candidates)
    }

    #[test]
    fn deterministic_selection_is_nested_and_input_order_independent() {
        let (manifest, mut candidates) = manifest();
        manifest.validate_against_candidates(&candidates).unwrap();
        candidates.reverse();
        manifest.validate_against_candidates(&candidates).unwrap();
        assert_eq!(manifest.targets[0].document_count, 512);
        assert_eq!(manifest.targets[1].document_count, 2_000);
        assert_eq!(manifest.targets[2].document_count, 10_000);
        assert_eq!(manifest.selections[511].selection_rank, 511);
        assert_eq!(manifest.selections[1_999].selection_rank, 1_999);
        assert_eq!(manifest.selections[9_999].selection_rank, 9_999);
    }

    #[test]
    fn golden_selection_identity_is_stable() {
        let (manifest, _) = manifest();
        assert_eq!(
            manifest.component_sha256.as_str(),
            "5acd33490716b32cf0f26126c611ba0eb741bf2066a640cc396f21642be49cd6"
        );
        assert_eq!(
            manifest.candidate_universe_sha256.as_str(),
            "db3d8d497360a0f62e87424bb2018e44cb43a9d1a7380d28564bea71ddf3ca4f"
        );
        assert_eq!(
            manifest.selections[..3]
                .iter()
                .map(|row| row.document_id.as_str())
                .collect::<Vec<_>>(),
            vec![
                "relation_a-short-2234",
                "relation_a-long-0272",
                "relation_b-short-2391",
            ]
        );
    }

    #[test]
    fn rejects_rank_duplicate_binding_and_safe_integer_mutations() {
        let (manifest, _) = manifest();

        let mut rank_gap = manifest.clone();
        rank_gap.selections[7].selection_rank = 8;
        rank_gap.component_sha256 = component_digest(&rank_gap).unwrap();
        assert!(rank_gap.validate().is_err());

        let mut duplicate = manifest.clone();
        duplicate.selections[1].document_id = duplicate.selections[0].document_id.clone();
        duplicate.component_sha256 = component_digest(&duplicate).unwrap();
        assert!(duplicate.validate().is_err());

        let mut wrong_prepared = manifest.clone();
        wrong_prepared.targets[0]
            .prepared_corpus
            .selected_documents_sha256 = digest("wrong-membership");
        wrong_prepared.component_sha256 = component_digest(&wrong_prepared).unwrap();
        assert!(wrong_prepared.validate().is_err());

        let mut unsafe_count = manifest;
        unsafe_count.candidate_count = super::super::MAX_SAFE_JSON_INTEGER + 1;
        unsafe_count.component_sha256 = component_digest(&unsafe_count).unwrap();
        assert!(unsafe_count.validate().is_err());
    }

    #[test]
    fn rejects_non_nested_or_under_capacity_quota() {
        let dataset = dataset();
        let mut policy = policy();
        policy.targets[1].quotas[0].documents = 100;
        policy.component_sha256 = component_digest(&policy).unwrap();
        assert!(policy.validate(&dataset).is_err());

        let mut candidates = candidates();
        candidates.retain(|candidate| {
            if candidate.primary_relation != "relation_a" || candidate.semantic_text_utf8_bytes != 5
            {
                return true;
            }
            candidate
                .document_id
                .rsplit('-')
                .next()
                .unwrap()
                .parse::<u64>()
                .unwrap()
                < 2_499
        });
        assert!(
            select_benchmark_documents(
                &dataset,
                &component("projection"),
                &crate::benchmark_selection::tests::policy(),
                &candidates,
            )
            .is_err()
        );
    }

    #[test]
    fn empty_length_cell_is_valid_only_with_zero_quota() {
        let mut dataset = dataset();
        dataset.included_relations = vec!["relation_a".into()];
        let target = |document_count| BenchmarkTargetQuota {
            document_count,
            quotas: vec![
                BenchmarkStratumQuota {
                    relation: "relation_a".into(),
                    length_stratum: "short".into(),
                    documents: document_count,
                },
                BenchmarkStratumQuota {
                    relation: "relation_a".into(),
                    length_stratum: "long".into(),
                    documents: 0,
                },
            ],
        };
        let mut policy = BenchmarkSelectionPolicy {
            schema_version: POLICY_SCHEMA.into(),
            component_sha256: digest("unsealed"),
            algorithm: "staged_stratified_sha256_v1".into(),
            selection_seed: "empty-cell-fixture".into(),
            length_strata: vec![
                BenchmarkLengthStratum {
                    id: "short".into(),
                    minimum_utf8_bytes: 0,
                    maximum_utf8_bytes: Some(9),
                },
                BenchmarkLengthStratum {
                    id: "long".into(),
                    minimum_utf8_bytes: 10,
                    maximum_utf8_bytes: None,
                },
            ],
            targets: STANDARD_BENCHMARK_SIZES.map(target).into_iter().collect(),
        };
        policy.seal(&dataset).unwrap();
        let candidates = (0..10_000)
            .map(|ordinal| {
                let id = format!("short-{ordinal:05}");
                BenchmarkSelectionCandidate {
                    document_id: id.clone(),
                    document_sha256: digest(&format!("document:{id}")),
                    semantic_text_sha256: digest(&format!("input:{id}")),
                    semantic_text_utf8_bytes: 5,
                    primary_relation: "relation_a".into(),
                }
            })
            .collect::<Vec<_>>();
        let (_, _, selected) =
            select_benchmark_documents(&dataset, &component("projection"), &policy, &candidates)
                .unwrap();
        assert_eq!(selected.len(), 10_000);
        assert!(selected.iter().all(|row| row.length_stratum == "short"));

        for target in &mut policy.targets {
            target.quotas[0].documents -= 1;
            target.quotas[1].documents = 1;
        }
        policy.seal(&dataset).unwrap();
        assert!(
            select_benchmark_documents(&dataset, &component("projection"), &policy, &candidates,)
                .is_err()
        );
    }

    #[test]
    fn schemas_are_valid_json_with_expected_ids() {
        for (text, expected_id) in [
            (
                include_str!("../../../specs/benchmark-selection-row.v1.schema.json"),
                "https://livefire.dev/rag/benchmark-selection-row.v1.schema.json",
            ),
            (
                include_str!("../../../specs/benchmark-selection-manifest.v1.schema.json"),
                "https://livefire.dev/rag/benchmark-selection-manifest.v1.schema.json",
            ),
        ] {
            let schema: serde_json::Value = serde_json::from_str(text).unwrap();
            assert_eq!(schema["$id"], expected_id);
            assert_eq!(
                schema["$schema"],
                "https://json-schema.org/draft/2020-12/schema"
            );
        }
    }
}

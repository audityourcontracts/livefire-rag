use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::{
    ComponentRef, DATASET_CATALOGUE_SCHEMA, DERIVED_RESULT_SET_SCHEMA, DatasetIdentity, Digest,
    EMBEDDING_PLAN_V2_SCHEMA, EmbeddingPlanV2, EmbeddingResultSetManifest, PipelineError,
    PreparedCorpusManifest, RESULT_SET_SCHEMA, Result, SafeRelativePath, TEST_RESULT_SET_SCHEMA,
    canonical_digest, component_digest, require_safe_u64, require_text,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CatalogueMode {
    Normal,
    TestOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogueArtifactRef {
    pub path: SafeRelativePath,
    #[serde(flatten)]
    pub component: ComponentRef,
}

impl CatalogueArtifactRef {
    fn validate(&self) -> Result<()> {
        self.component.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogueDatasetEntry {
    pub dataset: DatasetIdentity,
    pub dataset_sha256: Digest,
    pub projection_policy: ComponentRef,
    pub prepared_corpus: CatalogueArtifactRef,
    pub embedding_plan: CatalogueArtifactRef,
    pub embedding_result_set: CatalogueArtifactRef,
    pub embedding_profile: ComponentRef,
    pub final_index: CatalogueArtifactRef,
    pub searchable_document_count: u64,
    pub searchable_reference_count: u64,
    pub test_only: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RelationOverlapAllowance {
    pub relation: String,
    pub dataset_ids: Vec<String>,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetCatalogue {
    pub schema_version: String,
    pub component_sha256: Digest,
    pub mode: CatalogueMode,
    pub source_snapshot: ComponentRef,
    pub mapping: ComponentRef,
    pub projection_policy: ComponentRef,
    pub embedding_profile: ComponentRef,
    pub query_compatibility: String,
    pub rank_merge: String,
    pub datasets: Vec<CatalogueDatasetEntry>,
    pub allowed_relation_overlaps: Vec<RelationOverlapAllowance>,
}

impl DatasetCatalogue {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != DATASET_CATALOGUE_SCHEMA
            || self.query_compatibility != "single_embedding_profile"
            || self.rank_merge != "reciprocal_rank_fusion_v1"
            || self.datasets.is_empty()
        {
            return Err(PipelineError::Invalid("dataset catalogue contract"));
        }
        self.source_snapshot.validate()?;
        self.mapping.validate()?;
        self.projection_policy.validate()?;
        self.embedding_profile.validate()?;

        let mut dataset_ids = BTreeSet::new();
        let mut artifact_paths = BTreeSet::new();
        let mut relation_owners = BTreeMap::<&str, Vec<&str>>::new();
        let mut previous_dataset_id: Option<&str> = None;
        for entry in &self.datasets {
            entry.dataset.validate()?;
            for value in [
                entry.searchable_document_count,
                entry.searchable_reference_count,
            ] {
                require_safe_u64(value)?;
            }
            if entry.dataset.included_relations.is_empty()
                || entry.searchable_document_count == 0
                || entry.searchable_reference_count == 0
                || entry.searchable_reference_count < entry.searchable_document_count
                || previous_dataset_id.is_some_and(|previous| previous >= entry.dataset.id.as_str())
                || !dataset_ids.insert(entry.dataset.id.as_str())
                || entry.dataset_sha256 != canonical_digest(&entry.dataset)?
                || entry.dataset.source_snapshot != self.source_snapshot
                || entry.dataset.mapping != self.mapping
                || entry.projection_policy != self.projection_policy
                || entry.embedding_profile != self.embedding_profile
                || matches!(self.mode, CatalogueMode::Normal) && entry.test_only
                || matches!(self.mode, CatalogueMode::TestOnly) && !entry.test_only
            {
                return Err(PipelineError::Invalid("dataset catalogue entry binding"));
            }
            previous_dataset_id = Some(&entry.dataset.id);
            for artifact in [
                &entry.prepared_corpus,
                &entry.embedding_plan,
                &entry.embedding_result_set,
                &entry.final_index,
            ] {
                artifact.validate()?;
                if !artifact_paths.insert(&artifact.path) {
                    return Err(PipelineError::Invalid("duplicate catalogue artifact path"));
                }
            }
            for relation in &entry.dataset.included_relations {
                relation_owners
                    .entry(relation)
                    .or_default()
                    .push(&entry.dataset.id);
            }
        }

        validate_overlap_allowances(
            &self.allowed_relation_overlaps,
            &relation_owners,
            &dataset_ids,
        )?;
        if self.component_sha256 != component_digest(self)? {
            return Err(PipelineError::Invalid("dataset catalogue component digest"));
        }
        Ok(())
    }

    pub fn seal(&mut self) -> Result<()> {
        self.component_sha256 = component_digest(self)?;
        self.validate()
    }

    #[must_use]
    pub fn dataset(&self, dataset_id: &str) -> Option<&CatalogueDatasetEntry> {
        self.datasets
            .binary_search_by(|entry| entry.dataset.id.as_str().cmp(dataset_id))
            .ok()
            .map(|index| &self.datasets[index])
    }
}

fn validate_overlap_allowances(
    allowances: &[RelationOverlapAllowance],
    relation_owners: &BTreeMap<&str, Vec<&str>>,
    dataset_ids: &BTreeSet<&str>,
) -> Result<()> {
    let actual_overlaps = relation_owners
        .iter()
        .filter(|(_, owners)| owners.len() > 1)
        .map(|(relation, owners)| (*relation, owners.as_slice()))
        .collect::<BTreeMap<_, _>>();
    let mut previous_relation: Option<&str> = None;
    for allowance in allowances {
        require_text(&allowance.relation)?;
        require_text(&allowance.reason)?;
        if previous_relation.is_some_and(|previous| previous >= allowance.relation.as_str())
            || allowance.dataset_ids.len() < 2
            || allowance.dataset_ids.iter().any(String::is_empty)
            || allowance
                .dataset_ids
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || allowance
                .dataset_ids
                .iter()
                .any(|id| !dataset_ids.contains(id.as_str()))
        {
            return Err(PipelineError::Invalid("relation overlap allowance"));
        }
        let owners =
            actual_overlaps
                .get(allowance.relation.as_str())
                .ok_or(PipelineError::Invalid(
                    "unnecessary relation overlap allowance",
                ))?;
        if owners.len() != allowance.dataset_ids.len()
            || owners
                .iter()
                .zip(&allowance.dataset_ids)
                .any(|(owner, allowed)| *owner != allowed)
        {
            return Err(PipelineError::Invalid(
                "relation overlap allowance coverage",
            ));
        }
        previous_relation = Some(&allowance.relation);
    }
    if allowances.len() != actual_overlaps.len() {
        return Err(PipelineError::Invalid("unapproved relation overlap"));
    }
    Ok(())
}

/// Check the pipeline components loaded from one catalogue entry. Final-index
/// readers must additionally compare the loaded index component digest and its
/// document/reference counts to the entry before opening it for search.
pub fn validate_dataset_pipeline_binding(
    entry: &CatalogueDatasetEntry,
    prepared: &PreparedCorpusManifest,
    plan: &EmbeddingPlanV2,
    result_set: &EmbeddingResultSetManifest,
) -> Result<()> {
    prepared.validate()?;
    plan.validate_manifest_binding(prepared)?;
    let receipt_entries = result_set
        .receipts
        .iter()
        .map(|entry| (entry.task_id.as_str(), entry))
        .collect::<BTreeMap<_, _>>();
    if !matches!(
        (
            result_set.schema_version.as_str(),
            result_set.test_only,
            result_set.derivation.is_some()
        ),
        (RESULT_SET_SCHEMA, false, false)
            | (TEST_RESULT_SET_SCHEMA, true, false)
            | (DERIVED_RESULT_SET_SCHEMA, false, true)
    ) || result_set.component_sha256 != component_digest(result_set)?
        || result_set.plan_sha256 != plan.component_sha256
        || result_set.prepared_corpus_sha256 != prepared.component_sha256
        || result_set.embedding_profile_sha256 != plan.embedding_profile.component.sha256
        || result_set.document_count != prepared.document_count
        || result_set.document_order_sha256 != prepared.document_order_sha256
        || plan.schema_version != EMBEDDING_PLAN_V2_SCHEMA
        || entry.dataset != prepared.dataset
        || entry.dataset_sha256 != canonical_digest(&prepared.dataset)?
        || entry.projection_policy != prepared.projection_policy
        || entry.prepared_corpus.component.sha256 != prepared.component_sha256
        || entry.embedding_plan.component.sha256 != plan.component_sha256
        || entry.embedding_result_set.component.sha256 != result_set.component_sha256
        || entry.embedding_profile != plan.embedding_profile.component
        || entry.searchable_document_count != prepared.document_count
        || entry.searchable_reference_count != prepared.occurrence_count
        || receipt_entries.len() != result_set.receipts.len()
        || receipt_entries.len() != plan.tasks.len()
        || entry.test_only != result_set.test_only
    {
        return Err(PipelineError::Invalid(
            "catalogue pipeline component binding",
        ));
    }
    for task in &plan.tasks {
        let receipt = receipt_entries
            .get(task.task_id.as_str())
            .ok_or(PipelineError::Invalid("catalogue result task coverage"))?;
        if receipt.path != task.receipt_path {
            return Err(PipelineError::Invalid("catalogue result task binding"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::digest_bytes;

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

    fn artifact(dataset_id: &str, kind: &str) -> CatalogueArtifactRef {
        CatalogueArtifactRef {
            path: SafeRelativePath::new(format!("datasets/{dataset_id}/{kind}/manifest.json"))
                .unwrap(),
            component: component(&format!("{dataset_id}-{kind}")),
        }
    }

    fn dataset(id: &str, relation: &str) -> DatasetIdentity {
        DatasetIdentity {
            id: id.into(),
            version: "1".into(),
            source_snapshot: component("snapshot"),
            mapping: component("mapping"),
            included_relations: vec![relation.into()],
            excluded_relations: vec![],
            structured_only_relations: vec![],
        }
    }

    fn entry(id: &str, relation: &str, test_only: bool) -> CatalogueDatasetEntry {
        let dataset = dataset(id, relation);
        CatalogueDatasetEntry {
            dataset_sha256: canonical_digest(&dataset).unwrap(),
            dataset,
            projection_policy: component("projection"),
            prepared_corpus: artifact(id, "prepared"),
            embedding_plan: artifact(id, "plan"),
            embedding_result_set: artifact(id, "results"),
            embedding_profile: component("profile"),
            final_index: artifact(id, "index"),
            searchable_document_count: 10,
            searchable_reference_count: 20,
            test_only,
        }
    }

    fn catalogue() -> DatasetCatalogue {
        let mut catalogue = DatasetCatalogue {
            schema_version: DATASET_CATALOGUE_SCHEMA.into(),
            component_sha256: digest("unsealed"),
            mode: CatalogueMode::Normal,
            source_snapshot: component("snapshot"),
            mapping: component("mapping"),
            projection_policy: component("projection"),
            embedding_profile: component("profile"),
            query_compatibility: "single_embedding_profile".into(),
            rank_merge: "reciprocal_rank_fusion_v1".into(),
            datasets: vec![
                entry("dataset-a", "relation_a", false),
                entry("dataset-b", "relation_b", false),
            ],
            allowed_relation_overlaps: vec![],
        };
        catalogue.seal().unwrap();
        catalogue
    }

    #[test]
    fn valid_catalogue_is_sorted_searchable_and_golden() {
        let catalogue = catalogue();
        catalogue.validate().unwrap();
        assert_eq!(
            catalogue
                .dataset("dataset-b")
                .unwrap()
                .dataset
                .included_relations,
            ["relation_b"]
        );
        assert_eq!(
            catalogue.component_sha256.as_str(),
            "8d102ad1c3c0d3a61f86b67f58f39185072be59813e19572e8b4380fc8a9d1e5"
        );
    }

    #[test]
    fn rejects_duplicate_unsorted_and_unsafe_entries() {
        let valid = catalogue();

        let mut duplicate = valid.clone();
        duplicate.datasets[1].dataset.id = "dataset-a".into();
        duplicate.datasets[1].dataset_sha256 =
            canonical_digest(&duplicate.datasets[1].dataset).unwrap();
        assert!(duplicate.seal().is_err());

        let mut unsorted = valid.clone();
        unsorted.datasets.swap(0, 1);
        assert!(unsorted.seal().is_err());

        let mut duplicate_path = valid;
        duplicate_path.datasets[1].final_index.path =
            duplicate_path.datasets[0].final_index.path.clone();
        assert!(duplicate_path.seal().is_err());

        let unsafe_artifact = serde_json::json!({
            "path": "../outside/index.json",
            "id": "index",
            "version": "1",
            "sha256": "a".repeat(64),
        });
        assert!(serde_json::from_value::<CatalogueArtifactRef>(unsafe_artifact).is_err());
        let missing_path = serde_json::json!({
            "id": "index",
            "version": "1",
            "sha256": "a".repeat(64),
        });
        assert!(serde_json::from_value::<CatalogueArtifactRef>(missing_path).is_err());
    }

    #[test]
    fn rejects_mixed_source_projection_and_profile() {
        for mutation in 0..4 {
            let mut catalogue = catalogue();
            match mutation {
                0 => catalogue.datasets[1].dataset.source_snapshot = component("other-snapshot"),
                1 => catalogue.datasets[1].dataset.mapping = component("other-mapping"),
                2 => catalogue.datasets[1].projection_policy = component("other-projection"),
                _ => catalogue.datasets[1].embedding_profile = component("other-profile"),
            }
            catalogue.datasets[1].dataset_sha256 =
                canonical_digest(&catalogue.datasets[1].dataset).unwrap();
            assert!(catalogue.seal().is_err(), "mutation {mutation}");
        }
    }

    #[test]
    fn relation_overlap_requires_one_exact_explicit_allowance() {
        let mut overlapping = catalogue();
        overlapping.datasets[1] = entry("dataset-b", "relation_a", false);
        assert!(overlapping.seal().is_err());

        overlapping.allowed_relation_overlaps = vec![RelationOverlapAllowance {
            relation: "relation_a".into(),
            dataset_ids: vec!["dataset-a".into(), "dataset-b".into()],
            reason: "intentional migration comparison".into(),
        }];
        overlapping.seal().unwrap();

        let mut incomplete = overlapping;
        incomplete.allowed_relation_overlaps[0].dataset_ids =
            vec!["dataset-a".into(), "missing".into()];
        assert!(incomplete.seal().is_err());

        let mut unnecessary = catalogue();
        unnecessary.allowed_relation_overlaps = vec![RelationOverlapAllowance {
            relation: "relation_a".into(),
            dataset_ids: vec!["dataset-a".into(), "dataset-b".into()],
            reason: "not actually overlapping".into(),
        }];
        assert!(unnecessary.seal().is_err());
    }

    #[test]
    fn normal_and_test_only_catalogues_cannot_mix_index_kinds() {
        let mut normal = catalogue();
        normal.datasets[0].test_only = true;
        assert!(normal.seal().is_err());

        let mut testing = catalogue();
        testing.mode = CatalogueMode::TestOnly;
        assert!(testing.seal().is_err());
        for entry in &mut testing.datasets {
            entry.test_only = true;
        }
        testing.seal().unwrap();
    }

    #[test]
    fn rejects_invalid_counts_and_self_digest() {
        let valid = catalogue();
        let mut too_many = valid.clone();
        too_many.datasets[0].searchable_document_count = super::super::MAX_SAFE_JSON_INTEGER + 1;
        assert!(too_many.seal().is_err());

        let mut fewer_references = valid.clone();
        fewer_references.datasets[0].searchable_reference_count = 9;
        assert!(fewer_references.seal().is_err());

        let mut wrong_digest = valid;
        wrong_digest.component_sha256 = digest("wrong");
        assert!(wrong_digest.validate().is_err());
    }

    #[test]
    fn schema_is_valid_json_with_expected_id() {
        let schema: serde_json::Value = serde_json::from_str(include_str!(
            "../../../specs/dataset-catalogue.v1.schema.json"
        ))
        .unwrap();
        assert_eq!(
            schema["$id"],
            "https://livefire.dev/rag/dataset-catalogue.v1.schema.json"
        );
        assert_eq!(
            schema["$schema"],
            "https://json-schema.org/draft/2020-12/schema"
        );
    }
}

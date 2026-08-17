//! Optional smoke test for a locally available released snapshot. The normal
//! suite is fully synthetic; this test lets an operator check format parity
//! without encoding a sibling checkout or corpus path in the crate.

use rag_ocsf::{LocalSnapshotReader, SnapshotReader};

#[test]
#[ignore = "requires LIVEFIRE_OCSF_SNAPSHOT to name a local snapshot root"]
fn opens_and_reads_one_batch_from_a_local_snapshot() {
    let root = std::env::var_os("LIVEFIRE_OCSF_SNAPSHOT")
        .expect("set LIVEFIRE_OCSF_SNAPSHOT to a build-receipt.json parent directory");
    let reader = LocalSnapshotReader::open(root).expect("local snapshot must pass fast admission");
    let requested = std::env::var("LIVEFIRE_OCSF_RELATION").ok();
    let relation = reader
        .typed_relations()
        .find(|relation| {
            requested
                .as_deref()
                .is_none_or(|name| relation.name == name)
        })
        .expect("snapshot must materialize the requested typed semantic relation");
    let first = reader
        .scan(relation)
        .expect("typed relation scan opens")
        .next()
        .expect("typed relation must contain a batch")
        .expect("first batch reads");
    assert!(first.num_rows() > 0);
}

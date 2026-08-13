//! Shared Rust conformance helpers will grow from the fixture vertical slice.

/// Returns the experimental format version exercised by the workspace tests.
#[must_use]
pub const fn fast_index_version() -> &'static str {
    "livefire.rag.fast-index/1"
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn version_is_frozen() {
        assert_eq!(fast_index_version(), "livefire.rag.fast-index/1");
    }
}

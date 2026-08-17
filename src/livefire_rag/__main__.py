"""Retired Python command entry point.

Historical submodules remain importable for tests and comparisons. Production
operators must use the native Rust binaries.
"""

raise SystemExit(
    "the Python livefire-rag command is retired; use the Rust `rag` and "
    "`rag-provider` binaries"
)

"""Analysis tools for the Rust fast experimental RAG index."""

from .evaluate import evaluate_retrieval_run
from .geometry import write_pca_report
from .index import FastIndex, FastIndexError, VectorHeader, document_order_sha256

__all__ = [
    "FastIndex",
    "FastIndexError",
    "VectorHeader",
    "document_order_sha256",
    "evaluate_retrieval_run",
    "write_pca_report",
]

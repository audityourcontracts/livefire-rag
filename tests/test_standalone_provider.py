from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from livefire_rag.builder import build_fixture
from livefire_rag.canonical import canonical_sha256_omitting, component_ref
from livefire_rag.contracts import (
    PROTOCOL,
    PROVIDER_OBJECT_LOCK,
    PROVIDER_REF,
    SEARCH_TOOL_DESCRIPTOR,
    SEARCH_TOOL_ID,
    SIMILAR_TOOL_ID,
    TOOL_REFS,
)
from livefire_rag.index import IndexCorrupt, SemanticIndex, manifest_identity
from livefire_rag.provider import Provider
from livefire_rag.service import DeadlineExceeded, SemanticService


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures/semantic-index/small.v1.json"
BINDING = "c" * 64


class FakeEmbeddingHandler(BaseHTTPRequestHandler):
    vector = [0.0, 1.0, 0.0, 0.0]

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        json.loads(self.rfile.read(length))
        body = json.dumps(
            {"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": self.vector}]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


class StandaloneProviderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = Path(tempfile.mkdtemp())
        self.index_dir = self.temp / "index"
        self.manifest = build_fixture(FIXTURE, self.index_dir)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeEmbeddingHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        shutil.rmtree(self.temp)

    def test_fixture_build_is_byte_reproducible_and_verifiable(self) -> None:
        second = self.temp / "second"
        second_manifest = build_fixture(FIXTURE, second)
        self.assertEqual(self.manifest, second_manifest)
        for name in ("manifest.json", "objects.lock.json", "documents.jsonl", "vectors.f32"):
            self.assertEqual((self.index_dir / name).read_bytes(), (second / name).read_bytes())
        opened = SemanticIndex.open(self.index_dir)
        self.assertEqual(len(opened.documents), 4)

    def test_builder_sorts_documents_and_their_vectors_as_pairs(self) -> None:
        fixture = json.loads(FIXTURE.read_text())
        fixture["rows"] = list(reversed(fixture["rows"]))
        shuffled_fixture = self.temp / "shuffled.json"
        shuffled_fixture.write_text(json.dumps(fixture))
        shuffled_index = self.temp / "shuffled-index"
        shuffled_manifest = build_fixture(shuffled_fixture, shuffled_index)
        self.assertEqual(shuffled_manifest, self.manifest)
        opened = SemanticIndex.open(shuffled_index)
        self.assertEqual(opened.documents[0]["command_id"], "cmd-a")
        self.assertEqual(opened.vectors[0].tolist(), [1.0, 0.0, 0.0, 0.0])

    def test_corruption_fails_closed(self) -> None:
        path = self.index_dir / "documents.jsonl"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaises(IndexCorrupt):
            SemanticIndex.open(self.index_dir)

    def test_component_identities_use_normative_rfc8785_material(self) -> None:
        self.assertEqual(
            self.manifest["component"]["sha256"], manifest_identity(self.manifest)
        )
        self.assertEqual(
            SEARCH_TOOL_DESCRIPTOR["tool"]["sha256"],
            canonical_sha256_omitting(SEARCH_TOOL_DESCRIPTOR, ("tool", "sha256")),
        )
        self.assertEqual(
            PROVIDER_REF,
            component_ref(PROVIDER_REF["id"], PROVIDER_REF["version"], PROVIDER_OBJECT_LOCK),
        )

    def test_pointer_requires_full_snapshot_membership_and_closed_locator(self) -> None:
        fixture = json.loads(FIXTURE.read_text())
        fixture["rows"][0]["document"]["source_pointer"]["snapshot"]["id"] = "different.snapshot"
        changed = self.temp / "changed-snapshot.json"
        changed.write_text(json.dumps(fixture))
        with self.assertRaisesRegex(ValueError, "undeclared source snapshot"):
            build_fixture(changed, self.temp / "changed-index")

        fixture = json.loads(FIXTURE.read_text())
        fixture["rows"][0]["document"]["source_pointer"]["locator"] = {
            "kind": "jsonl_record",
            "object_sha256": "G" * 64,
            "line_ordinal": 0,
        }
        invalid = self.temp / "invalid-locator.json"
        invalid.write_text(json.dumps(fixture))
        with self.assertRaisesRegex(ValueError, "object_sha256"):
            build_fixture(invalid, self.temp / "invalid-index")

    def test_similar_exact_tie_break_filters_and_miss(self) -> None:
        service = SemanticService(SemanticIndex.open(self.index_dir), self.endpoint)
        result = service.similar(
            {
                "schema_version": "livefire.rag.cli-similar.input/1",
                "command_id": "cmd-a",
                "top_n": 3,
            },
            int(time.time() * 1000) + 5000,
        )
        self.assertEqual(result["kind"], "pointer")
        self.assertEqual([row["command_id"] for row in result["pointers"]], ["cmd-b", "cmd-c", "cmd-d"])
        self.assertEqual(result["pointers"][1]["cosine_distance_millionths"], 1_000_000)
        self.assertEqual(result["pointers"][2]["cosine_distance_millionths"], 1_000_000)

        miss = service.similar(
            {
                "schema_version": "livefire.rag.cli-similar.input/1",
                "command_id": "cmd-a",
                "top_n": 2,
                "filters": {"host_ids": ["does-not-exist"]},
            },
            int(time.time() * 1000) + 5000,
        )
        self.assertEqual(miss["kind"], "miss")
        self.assertEqual(miss["coverage"]["eligible_commands"], 0)

    def test_search_loopback_and_closed_filters(self) -> None:
        service = SemanticService(SemanticIndex.open(self.index_dir), self.endpoint)
        result = service.search(
            {
                "schema_version": "livefire.rag.cli-search.input/1",
                "query": "disable firewall",
                "time_range": {"start": "2024-01-01T00:00:00Z", "end_exclusive": "2024-01-04T00:00:00Z"},
                "top_n": 2,
                "filters": {"shell_families": ["powershell"]},
            },
            int(time.time() * 1000) + 5000,
        )
        self.assertEqual(result["kind"], "pointer")
        self.assertEqual([row["command_id"] for row in result["pointers"]], ["cmd-c", "cmd-a"])
        self.assertEqual(result["coverage"]["eligible_commands"], 2)

    def test_expired_deadline_is_error_not_miss(self) -> None:
        service = SemanticService(SemanticIndex.open(self.index_dir), self.endpoint)
        with self.assertRaises(DeadlineExceeded):
            service.similar(
                {"schema_version": "livefire.rag.cli-similar.input/1", "command_id": "cmd-a", "top_n": 1},
                int(time.time() * 1000) - 1,
            )

    def _request(self, request_id: str, method: str, params: dict) -> dict:
        return {
            "protocol": PROTOCOL,
            "id": request_id,
            "method": method,
            "params": params,
            "context": {"trace_id": f"trace-{request_id}", "deadline_unix_ms": int(time.time() * 1000) + 5000},
        }

    def test_provider_handshake_open_call_health_close(self) -> None:
        provider = Provider(self.endpoint)
        handshake = provider.handle(self._request("1", "handshake", {}))
        self.assertEqual(handshake["response_kind"], "handshake")
        self.assertEqual(handshake["provider"], PROVIDER_REF)
        opened = provider.handle(
            self._request(
                "2",
                "open",
                {
                    "provider": PROVIDER_REF,
                    "tools": [TOOL_REFS[SEARCH_TOOL_ID], TOOL_REFS[SIMILAR_TOOL_ID]],
                    "indexes": [self.manifest["component"]],
                    "source_snapshots": self.manifest["source_snapshots"],
                    "binding_lock_sha256": BINDING,
                    "query_time_contract": {"embedding_endpoint": self.endpoint},
                    "limits": {"result_bytes": 1048576, "wall_time_ms": 5000},
                    "mounts": [{
                        "logical_name": "commands",
                        "role": "index",
                        "component": self.manifest["component"],
                        "access": "read_only",
                        "process_path": str(self.index_dir),
                    }],
                },
            )
        )
        session_id = opened["session_id"]
        called = provider.handle(
            self._request(
                "3",
                "call",
                {
                    "session_id": session_id,
                    "tool": TOOL_REFS[SIMILAR_TOOL_ID],
                    "arguments": {"schema_version": "livefire.rag.cli-similar.input/1", "command_id": "cmd-a", "top_n": 1},
                },
            )
        )
        self.assertEqual(called["output"]["kind"], "pointer")
        health = provider.handle(self._request("4", "health", {"session_id": session_id}))
        self.assertEqual(health, {"response_kind": "health", "status": "ready", "binding_lock_sha256": BINDING})
        closed = provider.handle(self._request("5", "close", {"session_id": session_id}))
        self.assertEqual(closed, {"response_kind": "close", "closed": True})

    def test_jsonl_subprocess_has_clean_stdout_and_full_lifecycle(self) -> None:
        process = subprocess.Popen(
            [sys.executable, "-m", "livefire_rag.cli", "provider", "--embedding-endpoint", self.endpoint],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=ROOT,
        )
        assert process.stdin is not None
        assert process.stdout is not None

        def exchange(request: dict) -> dict:
            process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            process.stdin.flush()
            line = process.stdout.readline()
            self.assertTrue(line.endswith("\n"))
            return json.loads(line)

        handshake = exchange(self._request("wire-1", "handshake", {}))
        self.assertEqual(handshake["result"]["response_kind"], "handshake")
        opened = exchange(
            self._request(
                "wire-2",
                "open",
                {
                    "provider": PROVIDER_REF,
                    "tools": [TOOL_REFS[SIMILAR_TOOL_ID]],
                    "indexes": [self.manifest["component"]],
                    "source_snapshots": self.manifest["source_snapshots"],
                    "binding_lock_sha256": BINDING,
                    "query_time_contract": {},
                    "limits": {"result_bytes": 1048576},
                    "mounts": [{
                        "logical_name": "commands",
                        "role": "index",
                        "component": self.manifest["component"],
                        "access": "read_only",
                        "process_path": str(self.index_dir),
                    }],
                },
            )
        )
        session_id = opened["result"]["session_id"]
        called = exchange(
            self._request(
                "wire-3",
                "call",
                {
                    "session_id": session_id,
                    "tool": TOOL_REFS[SIMILAR_TOOL_ID],
                    "arguments": {"schema_version": "livefire.rag.cli-similar.input/1", "command_id": "cmd-a", "top_n": 1},
                },
            )
        )
        self.assertEqual(called["result"]["output"]["pointers"][0]["command_id"], "cmd-b")
        self.assertEqual(exchange(self._request("wire-4", "health", {"session_id": session_id}))["result"]["status"], "ready")
        self.assertTrue(exchange(self._request("wire-5", "close", {"session_id": session_id}))["result"]["closed"])
        process.stdin.close()
        self.assertEqual(process.wait(timeout=5), 0)
        assert process.stderr is not None
        self.assertEqual(process.stderr.read(), "")
        process.stdout.close()
        process.stderr.close()


if __name__ == "__main__":
    unittest.main()

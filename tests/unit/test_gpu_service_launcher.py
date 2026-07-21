import hashlib
import http.client
import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_PATH = REPO_ROOT / "workers" / "gpu-service" / "main.py"
SLURM_WRAPPER_PATH = REPO_ROOT / "deploy" / "slurm" / "gpu_service.slurm"


def load_launcher():
    spec = importlib.util.spec_from_file_location("gpu_service_launcher", LAUNCHER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


launcher = load_launcher()


class RuntimeArgumentTests(unittest.TestCase):
    def test_substitutes_each_documented_placeholder_once(self):
        deployment = {
            "model": "/models/literal-{port}",
            "quantization": "int4",
            "context_limit_tokens": 32768,
            "runtime_args": [
                "--model={model}",
                "--model-path={model_path}",
                "--quant={quantization}",
                "--context={context_limit_tokens}",
                "--host={host}",
                "--port={port}",
                "--tensor-parallel={gpu_count}",
                "--api-key={endpoint_token}",
                '{"json":"braces are preserved"}',
            ],
        }

        result = launcher.substitute_runtime_args(deployment, 8123, "secret-token", 4)

        self.assertEqual(
            result,
            [
                "--model=/models/literal-{port}",
                "--model-path=/models/literal-{port}",
                "--quant=int4",
                "--context=32768",
                "--host=127.0.0.1",
                "--port=8123",
                "--tensor-parallel=4",
                "--api-key=secret-token",
                '{"json":"braces are preserved"}',
            ],
        )

    def test_rejects_unknown_placeholder_instead_of_launching_bad_command(self):
        deployment = {
            "model": "/models/demo",
            "quantization": "int4",
            "context_limit_tokens": 32768,
            "runtime_args": ["--port={prt}"],
        }

        with self.assertRaisesRegex(ValueError, "unsupported runtime argument placeholder"):
            launcher.substitute_runtime_args(deployment, 8123, "secret-token", 1)


class _LiveProcess:
    returncode = None

    def poll(self):
        return None


class _ReadinessHandler(BaseHTTPRequestHandler):
    requests = []
    statuses = []

    def log_message(self, _message, *_args):
        return

    def do_GET(self):
        type(self).requests.append(
            {
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
            }
        )
        status = type(self).statuses.pop(0) if type(self).statuses else 200
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()


class RuntimeReadinessTests(unittest.TestCase):
    def test_waits_for_authenticated_health_instead_of_tcp_listener(self):
        _ReadinessHandler.requests = []
        _ReadinessHandler.statuses = [503, 200]
        server = ThreadingHTTPServer(("127.0.0.1", 0), _ReadinessHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        try:
            launcher.wait_for_runtime(_LiveProcess(), server.server_address[1], "runtime-secret", 2)
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=5)

        self.assertEqual(len(_ReadinessHandler.requests), 2)
        self.assertEqual(
            _ReadinessHandler.requests,
            [
                {"path": "/health", "authorization": "Bearer runtime-secret"},
                {"path": "/health", "authorization": "Bearer runtime-secret"},
            ],
        )


class RetrievalAdapterTests(unittest.TestCase):
    def test_index_upsert_search_and_rerank_fallback(self):
        embeddings = {
            "alpha beta": [1.0, 0.0],
            "gamma delta": [0.0, 1.0],
            "alpha query": [1.0, 0.0],
            "path: a.py\nalpha beta": [1.0, 0.0],
            "path: b.py\ngamma delta": [0.0, 1.0],
        }

        class FakeUpstream:
            def __init__(self):
                self.calls = []

            def post(self, path, payload):
                self.calls.append((path, payload))
                if path == "/v1/embeddings":
                    items = []
                    for index, text in enumerate(payload["input"]):
                        items.append({"index": index, "embedding": embeddings[text]})
                    return {"data": items}
                if path == "/v1/rerank":
                    raise RuntimeError("upstream /v1/rerank failed (501): Not Implemented")
                raise AssertionError(path)

        adapter = launcher.RetrievalServiceAdapter(9000, "runtime-secret")
        adapter.upstream = FakeUpstream()

        status, payload = adapter.handle(
            "/v1/indexes/status",
            {"model_profile": "retrieval", "index_fingerprint": "fp1", "document_count": 2},
        )
        self.assertEqual((status, payload), (200, {"ready": False, "document_count": 0}))

        status, payload = adapter.handle(
            "/v1/indexes/upsert",
            {
                "model": "retrieval-model",
                "model_profile": "retrieval",
                "index_fingerprint": "fp1",
                "document_count": 2,
                "replace": True,
                "finalize": True,
                "documents": [
                    {"id": "chunk_a", "text": "alpha beta"},
                    {"id": "chunk_b", "text": "gamma delta"},
                ],
            },
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload["accepted"])

        status, payload = adapter.handle(
            "/v1/indexes/status",
            {"model_profile": "retrieval", "index_fingerprint": "fp1", "document_count": 2},
        )
        self.assertEqual((status, payload), (200, {"ready": True, "document_count": 2}))

        status, payload = adapter.handle(
            "/v1/search",
            {
                "model": "retrieval-model",
                "model_profile": "retrieval",
                "index_fingerprint": "fp1",
                "query": "alpha query",
                "limit": 2,
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual([item["id"] for item in payload["results"]], ["chunk_a", "chunk_b"])

        status, payload = adapter.handle(
            "/v1/rerank",
            {
                "model": "retrieval-model",
                "query": "alpha query",
                "documents": ["path: a.py\nalpha beta", "path: b.py\ngamma delta"],
                "top_n": 2,
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual([item["index"] for item in payload["results"]], [0, 1])

    def test_persistent_semantic_cache_reloads_after_adapter_restart(self):
        embeddings = {
            "alpha beta": [1.0, 0.0],
            "alpha query": [1.0, 0.0],
        }

        class FakeUpstream:
            def __init__(self):
                self.calls = []

            def post(self, path, payload):
                self.calls.append((path, payload))
                if path == "/v1/embeddings":
                    return {
                        "data": [
                            {"index": index, "embedding": embeddings[text]}
                            for index, text in enumerate(payload["input"])
                        ]
                    }
                raise AssertionError(path)

        with tempfile.TemporaryDirectory() as temp_dir:
            upstream = FakeUpstream()
            adapter = launcher.RetrievalServiceAdapter(9000, "runtime-secret", cache_dir=temp_dir)
            adapter.upstream = upstream
            status, payload = adapter.handle(
                "/v1/indexes/upsert",
                {
                    "model": "retrieval-model",
                    "model_profile": "retrieval",
                    "index_fingerprint": "fp1",
                    "document_count": 1,
                    "replace": True,
                    "finalize": True,
                    "documents": [{"id": "chunk_a", "text": "alpha beta"}],
                },
            )
            self.assertEqual(status, 200)
            self.assertTrue(payload["accepted"])
            self.assertEqual(len(upstream.calls), 1)

            restarted = launcher.RetrievalServiceAdapter(9000, "runtime-secret", cache_dir=temp_dir)
            restarted.upstream = FakeUpstream()
            status, payload = restarted.handle(
                "/v1/indexes/status",
                {"model_profile": "retrieval", "index_fingerprint": "fp1", "document_count": 1},
            )
            self.assertEqual((status, payload), (200, {"ready": True, "document_count": 1}))

            status, payload = restarted.handle(
                "/v1/search",
                {
                    "model": "retrieval-model",
                    "model_profile": "retrieval",
                    "index_fingerprint": "fp1",
                    "query": "alpha query",
                    "limit": 1,
                },
            )
            self.assertEqual(status, 200)
            self.assertEqual([item["id"] for item in payload["results"]], ["chunk_a"])
            self.assertEqual(len(restarted.upstream.calls), 1)

    def test_finalize_prepares_cached_search_matrix_and_reuses_it_for_hot_search(self):
        embeddings = {
            "alpha beta": [1.0, 0.0],
            "gamma delta": [0.0, 1.0],
            "alpha query": [1.0, 0.0],
        }

        class FakeUpstream:
            def __init__(self):
                self.calls = []

            def post(self, path, payload):
                self.calls.append((path, payload))
                if path == "/v1/embeddings":
                    return {
                        "data": [
                            {"index": index, "embedding": embeddings[text]}
                            for index, text in enumerate(payload["input"])
                        ]
                    }
                raise AssertionError(path)

        adapter = launcher.RetrievalServiceAdapter(9000, "runtime-secret")
        adapter.upstream = FakeUpstream()
        status, payload = adapter.handle(
            "/v1/indexes/upsert",
            {
                "model": "retrieval-model",
                "model_profile": "retrieval",
                "index_fingerprint": "fp1",
                "document_count": 2,
                "replace": True,
                "finalize": True,
                "documents": [
                    {"id": "chunk_a", "text": "alpha beta"},
                    {"id": "chunk_b", "text": "gamma delta"},
                ],
            },
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload["accepted"])

        record = adapter.indexes[("retrieval", "fp1")]
        if launcher.np is not None:
            self.assertEqual(tuple(record["_search_matrix"].shape), (2, 2))
        self.assertEqual(record["_search_ids"], ["chunk_a", "chunk_b"])

        status, payload = adapter.handle(
            "/v1/search",
            {
                "model": "retrieval-model",
                "model_profile": "retrieval",
                "index_fingerprint": "fp1",
                "query": "alpha query",
                "limit": 1,
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual([item["id"] for item in payload["results"]], ["chunk_a"])
        self.assertEqual(record["_search_ids"], ["chunk_a", "chunk_b"])

    def test_search_uses_faiss_index_when_available(self):
        embeddings = {
            "alpha beta": [1.0, 0.0],
            "gamma delta": [0.0, 1.0],
            "alpha query": [1.0, 0.0],
        }

        class FakeFaissIndex:
            def __init__(self, dims):
                self.dims = dims
                self.matrix = None

            def add(self, matrix):
                self.matrix = matrix

            def search(self, query, k):
                scores = self.matrix @ query[0]
                order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))[:k]
                padded_scores = [float(scores[index]) for index in order]
                padded_indices = list(order)
                while len(padded_scores) < k:
                    padded_scores.append(float("-inf"))
                    padded_indices.append(-1)
                return launcher.np.asarray([padded_scores], dtype=launcher.np.float32), launcher.np.asarray([padded_indices], dtype=launcher.np.int64)

        class FakeFaiss:
            def __init__(self):
                self.calls = []

            def IndexFlatIP(self, dims):
                self.calls.append(dims)
                return FakeFaissIndex(dims)

        class FakeUpstream:
            def __init__(self):
                self.calls = []

            def post(self, path, payload):
                self.calls.append((path, payload))
                if path == "/v1/embeddings":
                    return {
                        "data": [
                            {"index": index, "embedding": embeddings[text]}
                            for index, text in enumerate(payload["input"])
                        ]
                    }
                raise AssertionError(path)

        fake_faiss = FakeFaiss()
        adapter = launcher.RetrievalServiceAdapter(9000, "runtime-secret")
        adapter.upstream = FakeUpstream()
        with mock.patch.object(launcher, "faiss", fake_faiss):
            status, payload = adapter.handle(
                "/v1/indexes/upsert",
                {
                    "model": "retrieval-model",
                    "model_profile": "retrieval",
                    "index_fingerprint": "fp1",
                    "document_count": 2,
                    "replace": True,
                    "finalize": True,
                    "documents": [
                        {"id": "chunk_a", "text": "alpha beta"},
                        {"id": "chunk_b", "text": "gamma delta"},
                    ],
                },
            )
            self.assertEqual(status, 200)
            self.assertTrue(payload["accepted"])

            status, payload = adapter.handle(
                "/v1/search",
                {
                    "model": "retrieval-model",
                    "model_profile": "retrieval",
                    "index_fingerprint": "fp1",
                    "query": "alpha query",
                    "limit": 1,
                },
            )

        self.assertEqual(status, 200)
        self.assertEqual([item["id"] for item in payload["results"]], ["chunk_a"])
        self.assertEqual(fake_faiss.calls, [2])

    def test_persisted_faiss_sidecar_reloads_after_restart_when_available(self):
        embeddings = {
            "alpha beta": [1.0, 0.0],
            "alpha query": [1.0, 0.0],
        }

        class FakeFaissIndex:
            def __init__(self, dims):
                self.dims = dims
                self.matrix = None

            def add(self, matrix):
                self.matrix = matrix

            def search(self, query, k):
                scores = self.matrix @ query[0]
                order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))[:k]
                padded_scores = [float(scores[index]) for index in order]
                padded_indices = list(order)
                while len(padded_scores) < k:
                    padded_scores.append(float("-inf"))
                    padded_indices.append(-1)
                return launcher.np.asarray([padded_scores], dtype=launcher.np.float32), launcher.np.asarray([padded_indices], dtype=launcher.np.int64)

        class FakeFaiss:
            def IndexFlatIP(self, dims):
                return FakeFaissIndex(dims)

            @staticmethod
            def serialize_index(index):
                return json.dumps({"dims": index.dims, "matrix": index.matrix.tolist()}).encode("utf-8")

            @staticmethod
            def deserialize_index(payload):
                decoded = json.loads(payload.decode("utf-8"))
                index = FakeFaissIndex(decoded["dims"])
                index.matrix = launcher.np.asarray(decoded["matrix"], dtype=launcher.np.float32)
                return index

        class FakeUpstream:
            def __init__(self):
                self.calls = []

            def post(self, path, payload):
                self.calls.append((path, payload))
                if path == "/v1/embeddings":
                    return {
                        "data": [
                            {"index": index, "embedding": embeddings[text]}
                            for index, text in enumerate(payload["input"])
                        ]
                    }
                raise AssertionError(path)

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(launcher, "faiss", FakeFaiss()):
            adapter = launcher.RetrievalServiceAdapter(9000, "runtime-secret", cache_dir=temp_dir)
            adapter.upstream = FakeUpstream()
            status, payload = adapter.handle(
                "/v1/indexes/upsert",
                {
                    "model": "retrieval-model",
                    "model_profile": "retrieval",
                    "index_fingerprint": "fp1",
                    "document_count": 1,
                    "replace": True,
                    "finalize": True,
                    "documents": [{"id": "chunk_a", "text": "alpha beta"}],
                },
            )
            self.assertEqual(status, 200)
            self.assertTrue(payload["accepted"])
            self.assertEqual(len(list(Path(temp_dir).glob("semantic-index-*.faiss"))), 1)
            self.assertEqual(len(list(Path(temp_dir).glob("semantic-index-*.npy"))), 0)

            restarted = launcher.RetrievalServiceAdapter(9000, "runtime-secret", cache_dir=temp_dir)
            restarted.upstream = FakeUpstream()
            with mock.patch.object(
                restarted,
                "_load_cache",
                side_effect=AssertionError("search should use persisted faiss+status sidecars before full cache reload"),
            ):
                status, payload = restarted.handle(
                    "/v1/search",
                    {
                        "model": "retrieval-model",
                        "model_profile": "retrieval",
                        "index_fingerprint": "fp1",
                        "query": "alpha query",
                        "limit": 1,
                    },
                )

        self.assertEqual(status, 200)
        self.assertEqual([item["id"] for item in payload["results"]], ["chunk_a"])
        self.assertEqual(len(restarted.upstream.calls), 1)

    def test_persisted_matrix_sidecar_reloads_after_restart_without_faiss(self):
        embeddings = {
            "alpha beta": [1.0, 0.0],
            "alpha query": [1.0, 0.0],
        }

        class FakeUpstream:
            def __init__(self):
                self.calls = []

            def post(self, path, payload):
                self.calls.append((path, payload))
                if path == "/v1/embeddings":
                    return {
                        "data": [
                            {"index": index, "embedding": embeddings[text]}
                            for index, text in enumerate(payload["input"])
                        ]
                    }
                raise AssertionError(path)

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(launcher, "faiss", None):
            adapter = launcher.RetrievalServiceAdapter(9000, "runtime-secret", cache_dir=temp_dir)
            adapter.upstream = FakeUpstream()
            status, payload = adapter.handle(
                "/v1/indexes/upsert",
                {
                    "model": "retrieval-model",
                    "model_profile": "retrieval",
                    "index_fingerprint": "fp1",
                    "document_count": 1,
                    "replace": True,
                    "finalize": True,
                    "documents": [{"id": "chunk_a", "text": "alpha beta"}],
                },
            )
            self.assertEqual(status, 200)
            self.assertTrue(payload["accepted"])
            self.assertEqual(len(list(Path(temp_dir).glob("semantic-index-*.npy"))), 1)

            restarted = launcher.RetrievalServiceAdapter(9000, "runtime-secret", cache_dir=temp_dir)
            restarted.upstream = FakeUpstream()
            with mock.patch.object(
                restarted,
                "_load_cache",
                side_effect=AssertionError("search should use persisted matrix+status sidecars before full cache reload"),
            ):
                status, payload = restarted.handle(
                    "/v1/search",
                    {
                        "model": "retrieval-model",
                        "model_profile": "retrieval",
                        "index_fingerprint": "fp1",
                        "query": "alpha query",
                        "limit": 1,
                    },
                )

        self.assertEqual(status, 200)
        self.assertEqual([item["id"] for item in payload["results"]], ["chunk_a"])
        self.assertEqual(len(restarted.upstream.calls), 1)

    def test_replacing_index_invalidates_and_rebuilds_search_cache(self):
        embeddings = {
            "alpha beta": [1.0, 0.0],
            "gamma delta": [0.0, 1.0],
            "delta query": [0.0, 1.0],
        }

        class FakeUpstream:
            def __init__(self):
                self.calls = []

            def post(self, path, payload):
                self.calls.append((path, payload))
                if path == "/v1/embeddings":
                    return {
                        "data": [
                            {"index": index, "embedding": embeddings[text]}
                            for index, text in enumerate(payload["input"])
                        ]
                    }
                raise AssertionError(path)

        adapter = launcher.RetrievalServiceAdapter(9000, "runtime-secret")
        adapter.upstream = FakeUpstream()
        adapter.handle(
            "/v1/indexes/upsert",
            {
                "model": "retrieval-model",
                "model_profile": "retrieval",
                "index_fingerprint": "fp1",
                "document_count": 1,
                "replace": True,
                "finalize": True,
                "documents": [{"id": "chunk_a", "text": "alpha beta"}],
            },
        )
        adapter.handle(
            "/v1/indexes/upsert",
            {
                "model": "retrieval-model",
                "model_profile": "retrieval",
                "index_fingerprint": "fp1",
                "document_count": 1,
                "replace": True,
                "finalize": True,
                "documents": [{"id": "chunk_b", "text": "gamma delta"}],
            },
        )

        record = adapter.indexes[("retrieval", "fp1")]
        self.assertEqual(record["_search_ids"], ["chunk_b"])
        status, payload = adapter.handle(
            "/v1/search",
            {
                "model": "retrieval-model",
                "model_profile": "retrieval",
                "index_fingerprint": "fp1",
                "query": "delta query",
                "limit": 1,
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual([item["id"] for item in payload["results"]], ["chunk_b"])

    def test_persisted_semantic_cache_omits_document_text_in_v2_format(self):
        embeddings = {
            "alpha beta": [1.0, 0.0],
        }

        class FakeUpstream:
            def __init__(self):
                self.calls = []

            def post(self, path, payload):
                self.calls.append((path, payload))
                if path == "/v1/embeddings":
                    return {
                        "data": [
                            {"index": index, "embedding": embeddings[text]}
                            for index, text in enumerate(payload["input"])
                        ]
                    }
                raise AssertionError(path)

        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = launcher.RetrievalServiceAdapter(9000, "runtime-secret", cache_dir=temp_dir)
            adapter.upstream = FakeUpstream()
            status, payload = adapter.handle(
                "/v1/indexes/upsert",
                {
                    "model": "retrieval-model",
                    "model_profile": "retrieval",
                    "index_fingerprint": "fp1",
                    "document_count": 1,
                    "replace": True,
                    "finalize": True,
                    "documents": [{"id": "chunk_a", "text": "alpha beta"}],
                },
            )
            self.assertEqual(status, 200)
            self.assertTrue(payload["accepted"])

            self.assertEqual(len(list(Path(temp_dir).glob("semantic-index-*.json.gz"))), 0)
            self.assertEqual(len(list(Path(temp_dir).glob("semantic-index-status-*.json"))), 1)
            self.assertEqual(len(list(Path(temp_dir).glob("semantic-vector-entry-*.json.gz"))), 0)
            self.assertGreaterEqual(len(list(Path(temp_dir).glob("semantic-vectors-*.json.gz"))), 1)

    def test_reuses_document_vectors_across_index_fingerprints(self):
        embeddings = {
            "alpha beta": [1.0, 0.0],
            "gamma delta": [0.0, 1.0],
            "epsilon zeta": [0.5, 0.5],
        }

        class FakeUpstream:
            def __init__(self):
                self.calls = []

            def post(self, path, payload):
                self.calls.append((path, payload))
                if path == "/v1/embeddings":
                    return {
                        "data": [
                            {"index": index, "embedding": embeddings[text]}
                            for index, text in enumerate(payload["input"])
                        ]
                    }
                raise AssertionError(path)

        with tempfile.TemporaryDirectory() as temp_dir:
            upstream = FakeUpstream()
            adapter = launcher.RetrievalServiceAdapter(9000, "runtime-secret", cache_dir=temp_dir)
            adapter.upstream = upstream
            first_documents = [
                {
                    "id": "chunk_a",
                    "text": "alpha beta",
                    "metadata": {"content_hash": "sha256:alpha"},
                },
                {
                    "id": "chunk_b",
                    "text": "gamma delta",
                    "metadata": {"content_hash": "sha256:gamma"},
                },
            ]
            status, payload = adapter.handle(
                "/v1/indexes/upsert",
                {
                    "model": "retrieval-model",
                    "model_profile": "retrieval",
                    "index_fingerprint": "fp1",
                    "document_count": 2,
                    "replace": True,
                    "finalize": True,
                    "documents": first_documents,
                },
            )
            self.assertEqual(status, 200)
            self.assertTrue(payload["accepted"])
            self.assertEqual([call[1]["input"] for call in upstream.calls], [["alpha beta", "gamma delta"]])

            status, payload = adapter.handle(
                "/v1/indexes/upsert",
                {
                    "model": "retrieval-model",
                    "model_profile": "retrieval",
                    "index_fingerprint": "fp2",
                    "document_count": 2,
                    "replace": True,
                    "finalize": True,
                    "documents": [
                        first_documents[0],
                        {
                            "id": "chunk_c",
                            "text": "epsilon zeta",
                            "metadata": {"content_hash": "sha256:epsilon"},
                        },
                    ],
                },
            )
            self.assertEqual(status, 200)
            self.assertTrue(payload["accepted"])
            self.assertEqual(
                [call[1]["input"] for call in upstream.calls],
                [["alpha beta", "gamma delta"], ["epsilon zeta"]],
            )

    def test_partial_update_rewrites_only_affected_vector_cache_shard(self):
        embeddings = {
            "alpha beta": [1.0, 0.0],
            "gamma delta": [0.0, 1.0],
            "alpha beta updated": [0.8, 0.2],
        }

        class FakeUpstream:
            def __init__(self):
                self.calls = []

            def post(self, path, payload):
                self.calls.append((path, payload))
                if path == "/v1/embeddings":
                    return {
                        "data": [
                            {"index": index, "embedding": embeddings[text]}
                            for index, text in enumerate(payload["input"])
                        ]
                    }
                raise AssertionError(path)

        with tempfile.TemporaryDirectory() as temp_dir:
            upstream = FakeUpstream()
            adapter = launcher.RetrievalServiceAdapter(9000, "runtime-secret", cache_dir=temp_dir)
            adapter.upstream = upstream
            status, payload = adapter.handle(
                "/v1/indexes/upsert",
                {
                    "model": "retrieval-model",
                    "model_profile": "retrieval",
                    "index_fingerprint": "fp1",
                    "document_count": 2,
                    "replace": True,
                    "finalize": True,
                    "documents": [
                        {"id": "chunk_a", "text": "alpha beta", "metadata": {"content_hash": "sha256:00alpha"}},
                        {"id": "chunk_b", "text": "gamma delta", "metadata": {"content_hash": "sha256:ffgamma"}},
                    ],
                },
            )
            self.assertEqual(status, 200)
            self.assertTrue(payload["accepted"])

            write_calls = []
            original_write_bytes_if_changed = launcher._write_bytes_if_changed

            def recording_write(path, data, *, prefix, suffix):
                write_calls.append(Path(path).name)
                return original_write_bytes_if_changed(path, data, prefix=prefix, suffix=suffix)

            with mock.patch.object(launcher, "_write_bytes_if_changed", side_effect=recording_write):
                status, payload = adapter.handle(
                    "/v1/indexes/upsert",
                    {
                        "model": "retrieval-model",
                        "model_profile": "retrieval",
                        "index_fingerprint": "fp2",
                        "document_count": 2,
                        "replace": True,
                        "finalize": True,
                        "documents": [
                            {"id": "chunk_a", "text": "alpha beta updated", "metadata": {"content_hash": "sha256:00updated"}},
                            {"id": "chunk_b", "text": "gamma delta", "metadata": {"content_hash": "sha256:ffgamma"}},
                        ],
                    },
                )
            self.assertEqual(status, 200)
            self.assertTrue(payload["accepted"])
            self.assertEqual(payload["embedded_documents"], 1)
            self.assertEqual(payload["reused_documents"], 1)

            vector_writes = [name for name in write_calls if name.startswith("semantic-vectors-")]
            self.assertEqual(len(vector_writes), 1)
            self.assertTrue(vector_writes[0].endswith("-00.json.gz"))

    def test_delta_upsert_reuses_base_index_documents_and_applies_removals(self):
        embeddings = {
            "alpha beta": [1.0, 0.0],
            "gamma delta": [0.0, 1.0],
            "epsilon zeta": [0.5, 0.5],
            "alpha beta query": [1.0, 0.0],
        }

        class FakeUpstream:
            def __init__(self):
                self.calls = []

            def post(self, path, payload):
                self.calls.append((path, payload))
                if path == "/v1/embeddings":
                    return {
                        "data": [
                            {"index": index, "embedding": embeddings[text]}
                            for index, text in enumerate(payload["input"])
                        ]
                    }
                raise AssertionError(path)

        adapter = launcher.RetrievalServiceAdapter(9000, "runtime-secret")
        adapter.upstream = FakeUpstream()

        status, payload = adapter.handle(
            "/v1/indexes/upsert",
            {
                "model": "retrieval-model",
                "model_profile": "retrieval",
                "index_fingerprint": "fp1",
                "document_count": 2,
                "replace": True,
                "finalize": True,
                "documents": [
                    {"id": "chunk_a", "text": "alpha beta"},
                    {"id": "chunk_b", "text": "gamma delta"},
                ],
            },
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload["accepted"])

        status, payload = adapter.handle(
            "/v1/indexes/upsert",
            {
                "model": "retrieval-model",
                "model_profile": "retrieval",
                "index_fingerprint": "fp2",
                "document_count": 2,
                "replace": True,
                "base_index_fingerprint": "fp1",
                "removed_document_ids": ["chunk_b"],
                "finalize": True,
                "documents": [{"id": "chunk_c", "text": "epsilon zeta"}],
            },
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload["accepted"])
        self.assertTrue(payload["base_reused"])
        self.assertEqual(payload["total_documents"], 2)

        status, payload = adapter.handle(
            "/v1/search",
            {
                "model": "retrieval-model",
                "model_profile": "retrieval",
                "index_fingerprint": "fp2",
                "query": "alpha beta query",
                "limit": 2,
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual({item["id"] for item in payload["results"]}, {"chunk_a", "chunk_c"})

    def test_noop_delta_finalize_clones_persisted_base_index_cache(self):
        embeddings = {
            "alpha beta": [1.0, 0.0],
        }

        class FakeUpstream:
            def __init__(self):
                self.calls = []

            def post(self, path, payload):
                self.calls.append((path, payload))
                if path == "/v1/embeddings":
                    return {
                        "data": [
                            {"index": index, "embedding": embeddings[text]}
                            for index, text in enumerate(payload["input"])
                        ]
                    }
                raise AssertionError(path)

        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = launcher.RetrievalServiceAdapter(9000, "runtime-secret", cache_dir=temp_dir)
            adapter.upstream = FakeUpstream()
            status, payload = adapter.handle(
                "/v1/indexes/upsert",
                {
                    "model": "retrieval-model",
                    "model_profile": "retrieval",
                    "index_fingerprint": "fp1",
                    "document_count": 1,
                    "replace": True,
                    "finalize": True,
                    "documents": [{"id": "chunk_a", "text": "alpha beta"}],
                },
            )
            self.assertEqual(status, 200)
            self.assertTrue(payload["accepted"])

            with mock.patch.object(adapter, "_write_cache", side_effect=AssertionError("should clone cache instead of rewriting")):
                status, payload = adapter.handle(
                    "/v1/indexes/upsert",
                    {
                        "model": "retrieval-model",
                        "model_profile": "retrieval",
                        "index_fingerprint": "fp2",
                        "document_count": 1,
                        "replace": True,
                        "base_index_fingerprint": "fp1",
                        "finalize": True,
                        "documents": [],
                    },
                )
            self.assertEqual(status, 200)
            self.assertTrue(payload["accepted"])
            self.assertTrue(payload["base_reused"])

            restarted = launcher.RetrievalServiceAdapter(9000, "runtime-secret", cache_dir=temp_dir)
            restarted.upstream = FakeUpstream()
            with mock.patch.object(
                restarted,
                "_load_cache",
                side_effect=AssertionError("alias index_status should not load full cache payload"),
            ):
                status, payload = restarted.handle(
                    "/v1/indexes/status",
                    {"model_profile": "retrieval", "index_fingerprint": "fp2", "document_count": 1},
                )
            self.assertEqual((status, payload), (200, {"ready": True, "document_count": 1}))

    def test_index_status_uses_status_sidecar_without_loading_full_index_cache(self):
        embeddings = {
            "alpha beta": [1.0, 0.0],
        }

        class FakeUpstream:
            def __init__(self):
                self.calls = []

            def post(self, path, payload):
                self.calls.append((path, payload))
                if path == "/v1/embeddings":
                    return {
                        "data": [
                            {"index": index, "embedding": embeddings[text]}
                            for index, text in enumerate(payload["input"])
                        ]
                    }
                raise AssertionError(path)

        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = launcher.RetrievalServiceAdapter(9000, "runtime-secret", cache_dir=temp_dir)
            adapter.upstream = FakeUpstream()
            status, payload = adapter.handle(
                "/v1/indexes/upsert",
                {
                    "model": "retrieval-model",
                    "model_profile": "retrieval",
                    "index_fingerprint": "fp1",
                    "document_count": 1,
                    "replace": True,
                    "finalize": True,
                    "documents": [{"id": "chunk_a", "text": "alpha beta"}],
                },
            )
            self.assertEqual(status, 200)
            self.assertTrue(payload["accepted"])

            restarted = launcher.RetrievalServiceAdapter(9000, "runtime-secret", cache_dir=temp_dir)
            restarted.upstream = FakeUpstream()
            with mock.patch.object(
                restarted,
                "_load_cache",
                side_effect=AssertionError("index_status should not load full cache payload"),
            ):
                status, payload = restarted.handle(
                    "/v1/indexes/status",
                    {
                        "model_profile": "retrieval",
                        "index_fingerprint": "fp1",
                        "document_count": 1,
                    },
                )
        self.assertEqual((status, payload), (200, {"ready": True, "document_count": 1}))

    def test_reuses_persisted_document_vectors_after_restart_for_new_index(self):
        embeddings = {
            "alpha beta": [1.0, 0.0],
            "gamma delta": [0.0, 1.0],
        }

        class FakeUpstream:
            def __init__(self):
                self.calls = []

            def post(self, path, payload):
                self.calls.append((path, payload))
                if path == "/v1/embeddings":
                    return {
                        "data": [
                            {"index": index, "embedding": embeddings[text]}
                            for index, text in enumerate(payload["input"])
                        ]
                    }
                raise AssertionError(path)

        with tempfile.TemporaryDirectory() as temp_dir:
            upstream = FakeUpstream()
            adapter = launcher.RetrievalServiceAdapter(9000, "runtime-secret", cache_dir=temp_dir)
            adapter.upstream = upstream
            status, payload = adapter.handle(
                "/v1/indexes/upsert",
                {
                    "model": "retrieval-model",
                    "model_profile": "retrieval",
                    "index_fingerprint": "fp1",
                    "document_count": 1,
                    "replace": True,
                    "finalize": True,
                    "documents": [
                        {
                            "id": "chunk_a",
                            "text": "alpha beta",
                            "metadata": {"content_hash": "sha256:alpha"},
                        }
                    ],
                },
            )
            self.assertEqual(status, 200)
            self.assertTrue(payload["accepted"])
            self.assertEqual([call[1]["input"] for call in upstream.calls], [["alpha beta"]])

            restarted = launcher.RetrievalServiceAdapter(9000, "runtime-secret", cache_dir=temp_dir)
            restarted.upstream = FakeUpstream()
            status, payload = restarted.handle(
                "/v1/indexes/upsert",
                {
                    "model": "retrieval-model",
                    "model_profile": "retrieval",
                    "index_fingerprint": "fp2",
                    "document_count": 2,
                    "replace": True,
                    "finalize": True,
                    "documents": [
                        {
                            "id": "chunk_a",
                            "text": "alpha beta",
                            "metadata": {"content_hash": "sha256:alpha"},
                        },
                        {
                            "id": "chunk_b",
                            "text": "gamma delta",
                            "metadata": {"content_hash": "sha256:gamma"},
                        },
                    ],
                },
            )
            self.assertEqual(status, 200)
            self.assertTrue(payload["accepted"])
            self.assertEqual([call[1]["input"] for call in restarted.upstream.calls], [["gamma delta"]])

    def test_restart_delta_upsert_reuses_base_index_without_full_semantic_cache_file(self):
        embeddings = {
            "alpha beta": [1.0, 0.0],
            "gamma delta": [0.0, 1.0],
        }

        class FakeUpstream:
            def __init__(self):
                self.calls = []

            def post(self, path, payload):
                self.calls.append((path, payload))
                if path == "/v1/embeddings":
                    return {
                        "data": [
                            {"index": index, "embedding": embeddings[text]}
                            for index, text in enumerate(payload["input"])
                        ]
                    }
                raise AssertionError(path)

        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = launcher.RetrievalServiceAdapter(9000, "runtime-secret", cache_dir=temp_dir)
            adapter.upstream = FakeUpstream()
            status, payload = adapter.handle(
                "/v1/indexes/upsert",
                {
                    "model": "retrieval-model",
                    "model_profile": "retrieval",
                    "index_fingerprint": "fp1",
                    "document_count": 1,
                    "replace": True,
                    "finalize": True,
                    "documents": [{"id": "chunk_a", "text": "alpha beta"}],
                },
            )
            self.assertEqual(status, 200)
            self.assertTrue(payload["accepted"])
            self.assertEqual(len(list(Path(temp_dir).glob("semantic-index-*.json.gz"))), 0)

            restarted = launcher.RetrievalServiceAdapter(9000, "runtime-secret", cache_dir=temp_dir)
            restarted.upstream = FakeUpstream()
            status, payload = restarted.handle(
                "/v1/indexes/upsert",
                {
                    "model": "retrieval-model",
                    "model_profile": "retrieval",
                    "index_fingerprint": "fp2",
                    "document_count": 2,
                    "replace": True,
                    "base_index_fingerprint": "fp1",
                    "finalize": True,
                    "documents": [{"id": "chunk_b", "text": "gamma delta"}],
                },
            )
            self.assertEqual(status, 200)
            self.assertTrue(payload["accepted"])
            self.assertTrue(payload["base_reused"])
            self.assertEqual([call[1]["input"] for call in restarted.upstream.calls], [["gamma delta"]])

    def test_reuses_general_embedding_cache_for_repeated_search_and_rerank_fallback(self):
        embeddings = {
            "alpha beta": [1.0, 0.0],
            "gamma delta": [0.0, 1.0],
            "alpha query": [1.0, 0.0],
            "path: a.py\nalpha beta": [1.0, 0.0],
            "path: b.py\ngamma delta": [0.0, 1.0],
        }

        class FakeUpstream:
            def __init__(self):
                self.calls = []

            def post(self, path, payload):
                self.calls.append((path, payload))
                if path == "/v1/embeddings":
                    return {
                        "data": [
                            {"index": index, "embedding": embeddings[text]}
                            for index, text in enumerate(payload["input"])
                        ]
                    }
                if path == "/v1/rerank":
                    raise RuntimeError("upstream /v1/rerank failed (501): Not Implemented")
                raise AssertionError(path)

        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = launcher.RetrievalServiceAdapter(9000, "runtime-secret", cache_dir=temp_dir)
            adapter.upstream = FakeUpstream()
            adapter.handle(
                "/v1/indexes/upsert",
                {
                    "model": "retrieval-model",
                    "model_profile": "retrieval",
                    "index_fingerprint": "fp1",
                    "document_count": 2,
                    "replace": True,
                    "finalize": True,
                    "documents": [
                        {"id": "chunk_a", "text": "alpha beta"},
                        {"id": "chunk_b", "text": "gamma delta"},
                    ],
                },
            )
            self.assertEqual(len([call for call in adapter.upstream.calls if call[0] == "/v1/embeddings"]), 1)

            adapter.handle(
                "/v1/search",
                {
                    "model": "retrieval-model",
                    "model_profile": "retrieval",
                    "index_fingerprint": "fp1",
                    "query": "alpha query",
                    "limit": 2,
                },
            )
            adapter.handle(
                "/v1/rerank",
                {
                    "model": "retrieval-model",
                    "query": "alpha query",
                    "documents": ["path: a.py\nalpha beta", "path: b.py\ngamma delta"],
                    "top_n": 2,
                },
            )
            first_embedding_calls = [call for call in adapter.upstream.calls if call[0] == "/v1/embeddings"]
            self.assertEqual(len(first_embedding_calls), 3)

            adapter.handle(
                "/v1/search",
                {
                    "model": "retrieval-model",
                    "model_profile": "retrieval",
                    "index_fingerprint": "fp1",
                    "query": "alpha query",
                    "limit": 2,
                },
            )
            adapter.handle(
                "/v1/rerank",
                {
                    "model": "retrieval-model",
                    "query": "alpha query",
                    "documents": ["path: a.py\nalpha beta", "path: b.py\ngamma delta"],
                    "top_n": 2,
                },
            )
            second_embedding_calls = [call for call in adapter.upstream.calls if call[0] == "/v1/embeddings"]
            self.assertEqual(len(second_embedding_calls), 3)

    def test_reuses_persisted_general_embedding_cache_after_restart(self):
        embeddings = {
            "alpha query": [1.0, 0.0],
        }

        class FakeUpstream:
            def __init__(self):
                self.calls = []

            def post(self, path, payload):
                self.calls.append((path, payload))
                if path == "/v1/embeddings":
                    return {
                        "data": [
                            {"index": index, "embedding": embeddings[text]}
                            for index, text in enumerate(payload["input"])
                        ]
                    }
                raise AssertionError(path)

        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = launcher.RetrievalServiceAdapter(9000, "runtime-secret", cache_dir=temp_dir)
            adapter.upstream = FakeUpstream()
            adapter._embed(["alpha query"], "retrieval-model")
            self.assertEqual(len(adapter.upstream.calls), 1)

            restarted = launcher.RetrievalServiceAdapter(9000, "runtime-secret", cache_dir=temp_dir)
            restarted.upstream = FakeUpstream()
            restarted._embed(["alpha query"], "retrieval-model")
            self.assertEqual(len(restarted.upstream.calls), 0)

    def test_index_upsert_persists_document_vectors_without_writing_segment_embedding_cache_files(self):
        embeddings = {
            "alpha beta": [1.0, 0.0],
            "gamma delta": [0.0, 1.0],
        }

        class FakeUpstream:
            def __init__(self):
                self.calls = []

            def post(self, path, payload):
                self.calls.append((path, payload))
                if path == "/v1/embeddings":
                    return {
                        "data": [
                            {"index": index, "embedding": embeddings[text]}
                            for index, text in enumerate(payload["input"])
                        ]
                    }
                raise AssertionError(path)

        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = launcher.RetrievalServiceAdapter(9000, "runtime-secret", cache_dir=temp_dir)
            adapter.upstream = FakeUpstream()
            status, payload = adapter.handle(
                "/v1/indexes/upsert",
                {
                    "model": "retrieval-model",
                    "model_profile": "retrieval",
                    "index_fingerprint": "fp1",
                    "document_count": 2,
                    "replace": True,
                    "finalize": True,
                    "documents": [
                        {"id": "chunk_a", "text": "alpha beta", "metadata": {"content_hash": "sha256:alpha"}},
                        {"id": "chunk_b", "text": "gamma delta", "metadata": {"content_hash": "sha256:gamma"}},
                    ],
                },
            )
            self.assertEqual(status, 200)
            self.assertTrue(payload["accepted"])

            cache_root = Path(temp_dir)
            vector_cache_files = sorted(cache_root.glob("semantic-vectors-*.json.gz"))
            vector_entry_files = sorted(cache_root.glob("semantic-vector-entry-*.json.gz"))
            embedding_cache_files = sorted(cache_root.glob("embedding-vector-*.json.gz"))
            self.assertGreaterEqual(len(vector_cache_files), 1)
            self.assertEqual(len(vector_entry_files), 0)
            self.assertEqual(len(embedding_cache_files), 0)

    def test_batch_limits_can_be_overridden_by_environment(self):
        with mock.patch.dict(
            os.environ,
            {
                "BROKER_GPU_SERVICE_EMBED_BATCH_ITEMS": "64",
                "BROKER_GPU_SERVICE_EMBED_BATCH_TOKENS": "8192",
                "BROKER_GPU_SERVICE_EMBED_SEGMENT_TOKENS": "1024",
            },
            clear=False,
        ):
            adapter = launcher.RetrievalServiceAdapter(9000, "runtime-secret")

        self.assertEqual(adapter.max_embed_batch_items, 64)
        self.assertEqual(adapter.max_embed_batch_tokens, 8192)
        self.assertEqual(adapter.max_embed_segment_tokens, 1024)

    def test_embeddings_are_split_into_safe_batches(self):
        class FakeUpstream:
            def __init__(self):
                self.calls = []

            def post(self, path, payload):
                self.calls.append((path, list(payload["input"])))
                return {
                    "data": [
                        {"index": index, "embedding": [float(index + 1), 0.0]}
                        for index, _ in enumerate(payload["input"])
                    ]
                }

        adapter = launcher.RetrievalServiceAdapter(9000, "runtime-secret")
        adapter.upstream = FakeUpstream()
        adapter.max_embed_batch_tokens = 10
        adapter.max_embed_batch_items = 2

        vectors = adapter._embed(["a" * 24, "b" * 24, "c" * 24], "retrieval-model")

        self.assertEqual(len(vectors), 3)
        self.assertEqual(len(adapter.upstream.calls), 3)

    def test_oversized_single_document_is_split_and_merged(self):
        class FakeUpstream:
            def __init__(self):
                self.calls = []

            def post(self, path, payload):
                self.calls.append((path, list(payload["input"])))
                return {
                    "data": [
                        {"index": index, "embedding": [1.0, float(index + 1)]}
                        for index, _ in enumerate(payload["input"])
                    ]
                }

        adapter = launcher.RetrievalServiceAdapter(9000, "runtime-secret")
        adapter.upstream = FakeUpstream()
        adapter.max_embed_batch_tokens = 16
        adapter.max_embed_batch_items = 4
        adapter.max_embed_segment_tokens = 4

        long_text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
        vectors = adapter._embed([long_text], "retrieval-model")

        self.assertEqual(len(vectors), 1)
        self.assertGreater(sum(len(payload) for _, payload in adapter.upstream.calls), 1)
        merged = vectors[0]
        self.assertEqual(len(merged), 2)
        self.assertAlmostEqual(sum(value * value for value in merged), 1.0, places=6)


class _UpstreamHandler(BaseHTTPRequestHandler):
    requests = []

    def log_message(self, _message, *_args):
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        type(self).requests.append(
            {
                "method": self.command,
                "path": self.path,
                "body": body,
                "headers": {key.lower(): value for key, value in self.headers.items()},
            }
        )
        payload = b'{"forwarded":true}'
        self.send_response(201)
        self.send_header("Content-Type", "application/json")
        self.send_header("Connection", "X-Upstream-Hop")
        self.send_header("X-Upstream-Hop", "must-not-escape")
        self.send_header("Proxy-Authenticate", "must-not-escape")
        self.send_header("X-Upstream", "visible")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        type(self).requests.append({"method": self.command, "path": self.path, "body": b"", "headers": {}})
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()


class AuthenticatedProxyTests(unittest.TestCase):
    def setUp(self):
        _UpstreamHandler.requests = []
        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
        self.upstream_thread = threading.Thread(target=self.upstream.serve_forever, daemon=True)
        self.upstream_thread.start()
        self.proxy = launcher.AuthenticatedProxy(
            ("127.0.0.1", 0), "proxy-secret", _LiveProcess(), self.upstream.server_address[1]
        )
        self.proxy_thread = threading.Thread(target=self.proxy.serve_forever, daemon=True)
        self.proxy_thread.start()

    def tearDown(self):
        self.proxy.shutdown()
        self.proxy.server_close()
        self.proxy_thread.join(timeout=5)
        self.upstream.shutdown()
        self.upstream.server_close()
        self.upstream_thread.join(timeout=5)

    def request(self, headers=None, body=b'{"input":"demo"}'):
        connection = http.client.HTTPConnection("127.0.0.1", self.proxy.server_address[1], timeout=5)
        try:
            connection.request("POST", "/v1/embeddings?x=1", body=body, headers=headers or {})
            response = connection.getresponse()
            payload = response.read()
            return response, payload
        finally:
            connection.close()

    def test_rejects_missing_or_incorrect_bearer_without_forwarding(self):
        response, payload = self.request()
        self.assertEqual(response.status, 401)
        self.assertEqual(json.loads(payload), {"error": "unauthorized"})

        response, _ = self.request({"Authorization": "Bearer wrong"})
        self.assertEqual(response.status, 401)
        self.assertEqual(_UpstreamHandler.requests, [])

    def test_forwards_authorized_request_with_proxy_auth_but_without_hop_headers(self):
        response, payload = self.request(
            {
                "Authorization": "Bearer proxy-secret",
                "Content-Type": "application/json",
                "X-Trace": "trace-id",
                "Connection": "X-Private-Hop",
                "X-Private-Hop": "must-not-forward",
                "Proxy-Authorization": "Basic must-not-forward",
            }
        )

        self.assertEqual(response.status, 201)
        self.assertEqual(json.loads(payload), {"forwarded": True})
        self.assertEqual(response.getheader("X-Upstream"), "visible")
        self.assertIsNone(response.getheader("X-Upstream-Hop"))
        self.assertIsNone(response.getheader("Proxy-Authenticate"))
        self.assertEqual(len(_UpstreamHandler.requests), 1)
        forwarded = _UpstreamHandler.requests[0]
        self.assertEqual(forwarded["method"], "POST")
        self.assertEqual(forwarded["path"], "/v1/embeddings?x=1")
        self.assertEqual(forwarded["body"], b'{"input":"demo"}')
        self.assertEqual(forwarded["headers"]["x-trace"], "trace-id")
        self.assertEqual(forwarded["headers"]["authorization"], "Bearer proxy-secret")
        self.assertNotIn("proxy-authorization", forwarded["headers"])
        self.assertNotIn("x-private-hop", forwarded["headers"])

    def test_malformed_content_length_returns_400_without_forwarding(self):
        sock = __import__("socket").create_connection(("127.0.0.1", self.proxy.server_address[1]), timeout=5)
        try:
            sock.sendall(
                b"POST /v1/embeddings HTTP/1.1\r\n"
                b"Host: localhost\r\n"
                b"Authorization: Bearer proxy-secret\r\n"
                b"Content-Length: invalid\r\n\r\n"
            )
            response = http.client.HTTPResponse(sock)
            response.begin()
            payload = response.read()
        finally:
            sock.close()

        self.assertEqual(response.status, 400)
        self.assertEqual(json.loads(payload), {"error": "invalid content length"})
        self.assertEqual(_UpstreamHandler.requests, [])

    def test_health_checks_the_upstream_runtime(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.proxy.server_address[1], timeout=5)
        try:
            connection.request("GET", "/health", headers={"Authorization": "Bearer proxy-secret"})
            response = connection.getresponse()
            response.read()
        finally:
            connection.close()

        self.assertEqual(response.status, 200)
        self.assertEqual(_UpstreamHandler.requests[-1]["path"], "/health")

    def test_intercepts_retrieval_adapter_endpoint_without_forwarding(self):
        class FakeAdapter:
            def handle(self, path, payload):
                return 200, {"path": path, "ready": payload.get("document_count") == 2}

        self.proxy.shutdown()
        self.proxy.server_close()
        self.proxy_thread.join(timeout=5)
        self.proxy = launcher.AuthenticatedProxy(
            ("127.0.0.1", 0),
            "proxy-secret",
            _LiveProcess(),
            self.upstream.server_address[1],
            retrieval_adapter=FakeAdapter(),
        )
        self.proxy_thread = threading.Thread(target=self.proxy.serve_forever, daemon=True)
        self.proxy_thread.start()

        connection = http.client.HTTPConnection("127.0.0.1", self.proxy.server_address[1], timeout=5)
        try:
            connection.request(
                "POST",
                "/v1/indexes/status",
                body=json.dumps({"document_count": 2}).encode("utf-8"),
                headers={
                    "Authorization": "Bearer proxy-secret",
                    "Content-Type": "application/json",
                },
            )
            response = connection.getresponse()
            payload = json.loads(response.read())
        finally:
            connection.close()

        self.assertEqual(response.status, 200)
        self.assertEqual(payload, {"path": "/v1/indexes/status", "ready": True})
        self.assertEqual(_UpstreamHandler.requests, [])


def registry_fixture(path, token, state="starting", absolute_lease_expires_at=None):
    created = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)
    record = {
        "id": "gpu-p40-retrieval-test",
        "tier": "p40-retrieval",
        "role": "retrieval",
        "state": state,
        "model_profile": "retrieval-profile",
        "model": "/models/retrieval",
        "capabilities": ["embeddings", "faiss_search", "rerank"],
        "context_limit_tokens": 32768,
        "gpu": {"type": "p40", "count": 1},
        "created_at": launcher.format_time(created),
        "startup_deadline": launcher.format_time(created + timedelta(minutes=10)),
        "lease_expires_at": launcher.format_time(created + timedelta(hours=4)),
        "registration_token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
    }
    if absolute_lease_expires_at is not None:
        record["absolute_lease_expires_at"] = launcher.format_time(absolute_lease_expires_at)
    path.write_text(
        json.dumps(
            {
                "schema": launcher.REGISTRY_SCHEMA,
                "updated_at": launcher.format_time(created),
                "records": [record],
                "demands": [],
            }
        ),
        encoding="utf-8",
    )
    os.chmod(path, 0o644)
    return created


class RegistryPublisherTests(unittest.TestCase):
    def test_publish_and_renew_validate_token_and_keep_registry_private(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "gpu-services.json"
            token = "registration-secret"
            created = registry_fixture(path, token)
            wrong = launcher.RegistryPublisher(path, "gpu-p40-retrieval-test", "wrong-token")
            with self.assertRaisesRegex(RuntimeError, "registration denied"):
                wrong.publish("http://node:9000", "endpoint-secret", "123")

            publisher = launcher.RegistryPublisher(path, "gpu-p40-retrieval-test", token)
            publish_time = created + timedelta(minutes=2)
            renew_time = created + timedelta(minutes=3)
            with mock.patch.object(launcher, "utc_now", side_effect=[publish_time, publish_time]):
                publisher.publish("http://node:9000", "endpoint-secret", "123")

            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(Path(str(path) + ".lock").stat().st_mode), 0o600)
            published = json.loads(path.read_text(encoding="utf-8"))["records"][0]
            self.assertEqual(published["state"], "ready")
            self.assertEqual(published["endpoint"], "http://node:9000")
            self.assertEqual(
                published["endpoint_auth"],
                {"type": "bearer", "bearer_token": "endpoint-secret"},
            )
            self.assertEqual(published["slurm_job_id"], "123")
            self.assertEqual(launcher.parse_time(published["heartbeat_at"]), publish_time)
            self.assertEqual(launcher.parse_time(published["lease_expires_at"]), publish_time + timedelta(hours=4))

            with self.assertRaisesRegex(RuntimeError, "registration denied"):
                wrong.renew()
            with mock.patch.object(launcher, "utc_now", side_effect=[renew_time, renew_time]):
                publisher.renew()

            renewed = json.loads(path.read_text(encoding="utf-8"))["records"][0]
            self.assertEqual(launcher.parse_time(renewed["heartbeat_at"]), renew_time)
            self.assertEqual(launcher.parse_time(renewed["lease_expires_at"]), renew_time + timedelta(hours=4))
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_publish_and_renew_cannot_extend_scale_zero_absolute_lease(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "gpu-services.json"
            token = "registration-secret"
            absolute = datetime(2026, 7, 10, 12, 30, tzinfo=timezone.utc)
            created = registry_fixture(path, token, absolute_lease_expires_at=absolute)
            publisher = launcher.RegistryPublisher(path, "gpu-p40-retrieval-test", token)
            publish_time = created + timedelta(minutes=5)
            renew_time = created + timedelta(minutes=25)

            with mock.patch.object(launcher, "utc_now", side_effect=[publish_time, publish_time]):
                publisher.publish("http://node:9000", "endpoint-secret", "123")
            published = json.loads(path.read_text(encoding="utf-8"))["records"][0]
            self.assertEqual(launcher.parse_time(published["lease_expires_at"]), absolute)

            with mock.patch.object(launcher, "utc_now", side_effect=[renew_time, renew_time]):
                publisher.renew()
            renewed = json.loads(path.read_text(encoding="utf-8"))["records"][0]
            self.assertEqual(launcher.parse_time(renewed["lease_expires_at"]), absolute)

    def test_go_zero_time_absolute_lease_does_not_cap_warm_p40(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "gpu-services.json"
            token = "registration-secret"
            created = registry_fixture(
                path,
                token,
                absolute_lease_expires_at=datetime.min.replace(tzinfo=timezone.utc),
            )
            publisher = launcher.RegistryPublisher(path, "gpu-p40-retrieval-test", token)
            publish_time = created + timedelta(minutes=2)

            with mock.patch.object(launcher, "utc_now", side_effect=[publish_time, publish_time]):
                publisher.publish("http://node:9000", "endpoint-secret", "123")

            published = json.loads(path.read_text(encoding="utf-8"))["records"][0]
            self.assertEqual(
                launcher.parse_time(published["lease_expires_at"]),
                publish_time + timedelta(hours=4),
            )

    def test_publish_cannot_revive_unhealthy_reservation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "gpu-services.json"
            token = "registration-secret"
            registry_fixture(path, token, state="unhealthy")
            publisher = launcher.RegistryPublisher(path, "gpu-p40-retrieval-test", token)

            with self.assertRaisesRegex(RuntimeError, "no longer publishable"):
                publisher.publish("http://node:9000", "endpoint-secret", "123")


class LaunchSpecTests(unittest.TestCase):
    def test_main_unlinks_private_spec_before_starting_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            spec_path = Path(temp) / "launch.json"
            spec_path.write_text('{"service_id":"demo"}', encoding="utf-8")
            os.chmod(spec_path, 0o600)

            def assert_removed(spec):
                self.assertEqual(spec, {"service_id": "demo"})
                self.assertFalse(spec_path.exists())
                return 0

            with mock.patch.object(sys, "argv", ["gpu-service", "--spec", str(spec_path)]), mock.patch.object(
                launcher, "run", side_effect=assert_removed
            ):
                self.assertEqual(launcher.main(), 0)

    def test_slurm_wrapper_does_not_pass_private_spec_path_in_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            capture = root / "capture.txt"
            fake_python = fake_bin / "python3"
            fake_python.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' \"${BROKER_GPU_SERVICE_SPEC_PATH-unset}\" \"$@\" > \"${CAPTURE_PATH}\"\n",
                encoding="utf-8",
            )
            os.chmod(fake_python, 0o755)
            env = os.environ.copy()
            env.update(
                {
                    "PATH": str(fake_bin) + os.pathsep + env.get("PATH", ""),
                    "CAPTURE_PATH": str(capture),
                    "BROKER_GPU_SERVICE_SPEC_PATH": "/private/launch.json",
                }
            )

            completed = subprocess.run(
                ["bash", str(SLURM_WRAPPER_PATH)],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            captured = capture.read_text(encoding="utf-8").splitlines()
            self.assertEqual(captured[0], "unset")
            self.assertEqual(captured[-2:], ["--spec", "/private/launch.json"])

    def test_slurm_wrapper_prefers_slurm_submit_dir_over_spooled_script_location(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            capture = root / "capture.txt"
            fake_python = fake_bin / "python3"
            fake_python.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' \"$@\" > \"${CAPTURE_PATH}\"\n",
                encoding="utf-8",
            )
            os.chmod(fake_python, 0o755)

            spooled_wrapper = root / "gpu_service.slurm"
            spooled_wrapper.write_text(SLURM_WRAPPER_PATH.read_text(encoding="utf-8"), encoding="utf-8")
            os.chmod(spooled_wrapper, 0o755)

            env = os.environ.copy()
            env.update(
                {
                    "PATH": str(fake_bin) + os.pathsep + env.get("PATH", ""),
                    "CAPTURE_PATH": str(capture),
                    "BROKER_GPU_SERVICE_SPEC_PATH": "/private/launch.json",
                    "SLURM_SUBMIT_DIR": str(REPO_ROOT),
                }
            )

            completed = subprocess.run(
                ["bash", str(spooled_wrapper)],
                cwd=root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            captured = capture.read_text(encoding="utf-8").splitlines()
            self.assertEqual(captured[0], str(REPO_ROOT / "workers" / "gpu-service" / "main.py"))
            self.assertEqual(captured[-2:], ["--spec", "/private/launch.json"])


if __name__ == "__main__":
    unittest.main()

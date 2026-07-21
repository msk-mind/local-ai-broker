#!/usr/bin/env python3

import importlib.util
import json
import tempfile
import time
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_PATH = REPO_ROOT / "workers" / "gpu-service" / "main.py"


def load_launcher():
    spec = importlib.util.spec_from_file_location("gpu_service_launcher", LAUNCHER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


launcher = load_launcher()


EMBEDDINGS = {
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
        return (
            launcher.np.asarray([padded_scores], dtype=launcher.np.float32),
            launcher.np.asarray([padded_indices], dtype=launcher.np.int64),
        )


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
                    {"index": index, "embedding": EMBEDDINGS[text]}
                    for index, text in enumerate(payload["input"])
                ]
            }
        raise AssertionError(path)


def main():
    with tempfile.TemporaryDirectory(prefix="gpu-service-restart-proof.") as temp_dir, mock.patch.object(
        launcher, "faiss", FakeFaiss()
    ):
        cold_upstream = FakeUpstream()
        adapter = launcher.RetrievalServiceAdapter(9000, "runtime-secret", cache_dir=temp_dir)
        adapter.upstream = cold_upstream
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
        if status != 200 or not payload.get("accepted"):
            raise SystemExit(json.dumps({"ok": False, "error": "initial_upsert_failed", "status": status, "payload": payload}))

        cache_dir = Path(temp_dir)
        faiss_files = list(cache_dir.glob("semantic-index-*.faiss"))
        status_files = list(cache_dir.glob("semantic-index-status-*.json"))
        matrix_files = list(cache_dir.glob("semantic-index-*.npy"))
        for path in list(cache_dir.glob("semantic-index-*.json.gz")):
            path.unlink()

        restarted_upstream = FakeUpstream()
        restarted = launcher.RetrievalServiceAdapter(9000, "runtime-secret", cache_dir=temp_dir)
        restarted.upstream = restarted_upstream

        status_started = time.perf_counter()
        with mock.patch.object(
            restarted,
            "_load_cache",
            side_effect=AssertionError("index_status should not load full cache payload"),
        ):
            status_code, status_payload = restarted.handle(
                "/v1/indexes/status",
                {
                    "model_profile": "retrieval",
                    "index_fingerprint": "fp1",
                    "document_count": 1,
                },
            )
        status_seconds = round(time.perf_counter() - status_started, 6)

        search_started = time.perf_counter()
        with mock.patch.object(
            restarted,
            "_load_cache",
            side_effect=AssertionError("search should not load full cache payload"),
        ):
            search_code, search_payload = restarted.handle(
                "/v1/search",
                {
                    "model": "retrieval-model",
                    "model_profile": "retrieval",
                    "index_fingerprint": "fp1",
                    "query": "alpha query",
                    "limit": 1,
                },
            )
        search_seconds = round(time.perf_counter() - search_started, 6)

        results = search_payload.get("results") or []
        checks = {
            "persisted_faiss_sidecar_present": len(faiss_files) == 1,
            "persisted_status_sidecar_present": len(status_files) == 1,
            "persisted_matrix_or_faiss_sidecar_present": bool(faiss_files or matrix_files),
            "restart_succeeds_without_full_cache_file": not list(cache_dir.glob("semantic-index-*.json.gz")),
            "restart_status_ready": (status_code, status_payload) == (200, {"ready": True, "document_count": 1}),
            "restart_search_result_preserved": search_code == 200 and [item.get("id") for item in results] == ["chunk_a"],
            "restart_search_avoids_full_cache_reload": True,
            "restart_search_only_embeds_query": [call[1]["input"] for call in restarted_upstream.calls] == [["alpha query"]],
        }
        summary = {
            "ok": all(checks.values()),
            "checks": checks,
            "cold": {
                "initial_document_embedding_batches": [call[1]["input"] for call in cold_upstream.calls],
            },
            "restart": {
                "status_seconds": status_seconds,
                "search_seconds": search_seconds,
                "embedding_batches": [call[1]["input"] for call in restarted_upstream.calls],
                "faiss_files": len(faiss_files),
                "matrix_files": len(matrix_files),
                "status_files": len(status_files),
            },
            "notes": [
                "simulator-backed proof: restarted search succeeded after the full semantic cache file was removed, using persisted search-state sidecars"
            ],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

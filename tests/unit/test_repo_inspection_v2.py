import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_DIR = REPO_ROOT / "workers" / "rag-compression"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

import gpu_client
import inspection_cached_result
import inspection_hotpath
import inspection_index
import inspection_pipeline
import inspect_repo_worker
import main as rag_main
import service_control


def git_args_match(args, root, subcommand):
    if len(args) < 4 or list(args[:3]) != ["git", "-C", str(root)]:
        return False
    index = 3
    if len(args) > index and args[index] == "--no-optional-locks":
        index += 1
    return len(args) > index and args[index] == subcommand


def git_args_match_legacy_clean_probe(args, root):
    return (
        (git_args_match(args, root, "diff-index") and "--quiet" in args)
        or (git_args_match(args, root, "diff-files") and "--quiet" in args)
        or (
            git_args_match(args, root, "ls-files")
            and "--others" in args
            and "--exclude-standard" in args
        )
    )


def service(tier, capabilities, gpu_count, *, state="ready", failure_category=""):
    now = datetime.now(timezone.utc)
    return {
        "id": f"svc-{tier}",
        "tier": tier,
        "state": state,
        "endpoint": f"http://{tier}.invalid",
        "endpoint_auth": {"type": "bearer", "bearer_token": "test-token"},
        "model_profile": f"profile-{tier}",
        "model": f"/models/{tier}",
        "capabilities": capabilities,
        "context_limit_tokens": 32_000,
        "gpu": {"type": tier.split("-", 1)[0], "count": gpu_count},
        "slurm_job_id": f"job-{tier}",
        "failure_category": failure_category,
        "heartbeat_at": now.isoformat().replace("+00:00", "Z"),
        "lease_expires_at": (now + timedelta(minutes=5)).isoformat().replace("+00:00", "Z"),
    }


def all_services():
    return [
        service("p40-retrieval", ["embed", "faiss_search", "rerank"], 1),
        service("p40-synthesis", ["chat"], 1),
        service("v100-reasoning", ["chat"], 4),
        service("a100-single", ["chat"], 1),
        service("a100-multigpu", ["chat"], 4),
    ]


class FakeClientFactory:
    def __init__(self, chat_results=None):
        self.chat_results = {key: list(value) for key, value in (chat_results or {}).items()}
        self.chat_calls = []
        self.embed_calls = []
        self.rerank_calls = []
        self.semantic_search_calls = []
        self.ensure_semantic_index_calls = []

    def __call__(self, record):
        return FakeClient(record, self)


class LexicalSharedPublishTests(unittest.TestCase):
    def _sample_chunks(self):
        chunks = inspection_index.ChunkList(
            [
                {
                    "chunk_id": "chunk_001",
                    "path": "service.py",
                    "repository_path": "service.py",
                    "source_namespace": "input_0",
                    "language": "python",
                    "symbol": "retry_job",
                    "line_start": 1,
                    "line_end": 3,
                    "content": "def retry_job(job_id):\n    return job_id\n",
                    "content_hash": "sha256:test-content",
                    "chunk_hash": "sha256:test-content",
                    "token_estimate": 12,
                }
            ]
        )
        return chunks

    def test_ensure_lexical_index_skips_sqlite_probe_when_file_signatures_match(self):
        chunks = self._sample_chunks()
        fingerprint = "sha256:test-fingerprint-next"
        build_config_digest = "sha256:test-build"

        with tempfile.TemporaryDirectory() as tempdir:
            cache_dir = Path(tempdir) / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            working_path = inspection_index._working_index_path(cache_dir)
            working_manifest_path = inspection_index._working_manifest_path(cache_dir)

            inspection_index._rebuild_working_lexical_index(working_path, chunks, "sha256:test-fingerprint-prev")
            manifest = inspection_index._lexical_manifest_for_chunks(chunks)
            only_key = next(iter(manifest))
            previous_record = dict(manifest[only_key])
            previous_record["path"] = "renamed-service.py"
            previous_record["repository_path"] = "renamed-service.py"
            previous_manifest = {
                only_key: previous_record,
                "__meta__": {
                    "fingerprint": "sha256:test-fingerprint-prev",
                    "chunk_count": len(chunks),
                    "build_config_digest": build_config_digest,
                },
            }
            inspection_index._write_lexical_working_manifest(
                working_manifest_path,
                previous_manifest,
                fingerprint="sha256:test-fingerprint-prev",
                chunk_count=len(chunks),
                build_config_digest=build_config_digest,
            )

            with mock.patch.object(
                inspection_index.sqlite3,
                "connect",
                side_effect=AssertionError("same-signature lexical hit should not probe sqlite"),
            ):
                index_path, cache_hit, stats = inspection_index.ensure_lexical_index(
                    chunks,
                    cache_dir,
                    fingerprint,
                    build_config_digest=build_config_digest,
                )

            self.assertEqual(index_path, working_path)
            self.assertTrue(cache_hit)
            self.assertTrue(stats["working_cache_hit"])
            self.assertEqual(stats["working_index_check_ms"], 0.0)
            updated_manifest = inspection_index._load_lexical_working_manifest(working_manifest_path)
            self.assertEqual(
                inspection_index._lexical_manifest_meta(updated_manifest),
                {
                    "fingerprint": fingerprint,
                    "chunk_count": len(chunks),
                    "build_config_digest": build_config_digest,
                },
            )

    def test_git_file_signature_manifest_second_load_uses_process_cache(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "git-file-signature-cache" / "manifest.json"
            inspection_index._write_git_file_signature_manifest(
                path,
                head="head-1",
                status="status-1",
                signatures={"a.py": "git:blob-a"},
            )
            first = inspection_index._load_git_file_signature_manifest(path)
            self.assertEqual(first["head"], "head-1")
            self.assertEqual(first["status"], "status-1")
            self.assertEqual(first["signatures"], {"a.py": "git:blob-a"})

            original_read_text = Path.read_text

            def guarded_read_text(self, *args, **kwargs):
                if self == path:
                    raise AssertionError("second manifest load should use process cache")
                return original_read_text(self, *args, **kwargs)

            with mock.patch.object(Path, "read_text", new=guarded_read_text):
                second = inspection_index._load_git_file_signature_manifest(path)

        self.assertEqual(second["head"], "head-1")
        self.assertEqual(second["status"], "status-1")
        self.assertEqual(second["signatures"], {"a.py": "git:blob-a"})

    def test_write_preferred_file_chunk_working_manifest_clones_shared_targets_from_local(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            local_path = root / "local" / "working.json"
            shared_path = root / "shared" / "working.json"
            shared_latest_path = root / "shared-latest" / "working.json"
            files = {
                "repo\x00service.py": {
                    "signature": "sha256:sig",
                    "cache_key": "chunk-cache",
                    "config_key": "cfg",
                    "empty": False,
                }
            }
            original_path_bytes_equal = inspection_index._path_bytes_equal

            def guarded_path_bytes_equal(path, payload):
                if path in {shared_path, shared_latest_path}:
                    raise AssertionError("shared targets should be cloned from local without shared byte-compare")
                return original_path_bytes_equal(path, payload)

            with mock.patch.object(inspection_index, "_path_bytes_equal", side_effect=guarded_path_bytes_equal):
                inspection_index._write_preferred_file_chunk_working_manifest(
                    local_path,
                    shared_path,
                    shared_latest_path,
                    files,
                    repository_state_fingerprint="sha256:repo",
                    build_config_digest="sha256:build",
                    publish_shared=True,
                )

            self.assertEqual(local_path.read_bytes(), shared_path.read_bytes())
            self.assertEqual(local_path.read_bytes(), shared_latest_path.read_bytes())


class DiscoveryReuseTests(LexicalSharedPublishTests):
    def test_discovered_files_from_previous_manifest_trusts_untouched_entries_when_enabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service_path = root / "service.py"
            untouched_path = root / "untouched.py"
            service_path.write_text("def retry_job(job_id):\n    return job_id\n", encoding="utf-8")
            untouched_path.write_text("def helper():\n    return 1\n", encoding="utf-8")
            discovered = [{"id": "input_0", "type": "repo", "classification": "internal", "path": root}]
            namespaces = {id(discovered[0]): "input_0"}
            previous_manifest = {
                "input_0\0service.py": {"signature": "sig-service"},
                "input_0\0untouched.py": {"signature": "sig-untouched"},
            }
            real_is_file = Path.is_file

            def guarded_is_file(path):
                if Path(path) == untouched_path:
                    raise AssertionError("untouched manifest entry should not be re-statted")
                return real_is_file(path)

            with mock.patch.object(Path, "is_file", autospec=True, side_effect=guarded_is_file):
                files = inspection_index._discovered_files_from_previous_manifest(
                    discovered,
                    namespaces,
                    previous_manifest,
                    ignored=set(),
                    touched_paths_hint=("service.py",),
                    trust_untouched_manifest=True,
                )

        self.assertEqual([rel for _item, _candidate, rel in files], ["service.py", "untouched.py"])

    def test_discovered_files_from_previous_manifest_rechecks_untouched_entries_when_not_trusted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service_path = root / "service.py"
            untouched_path = root / "untouched.py"
            service_path.write_text("def retry_job(job_id):\n    return job_id\n", encoding="utf-8")
            untouched_path.write_text("def helper():\n    return 1\n", encoding="utf-8")
            discovered = [{"id": "input_0", "type": "repo", "classification": "internal", "path": root}]
            namespaces = {id(discovered[0]): "input_0"}
            previous_manifest = {
                "input_0\0service.py": {"signature": "sig-service"},
                "input_0\0untouched.py": {"signature": "sig-untouched"},
            }
            real_is_file = Path.is_file
            untouched_checks = []

            def tracked_is_file(path):
                if Path(path) == untouched_path:
                    untouched_checks.append(str(path))
                return real_is_file(path)

            with mock.patch.object(Path, "is_file", autospec=True, side_effect=tracked_is_file):
                files = inspection_index._discovered_files_from_previous_manifest(
                    discovered,
                    namespaces,
                    previous_manifest,
                    ignored=set(),
                    touched_paths_hint=("service.py",),
                    trust_untouched_manifest=False,
                )

        self.assertEqual([rel for _item, _candidate, rel in files], ["service.py", "untouched.py"])
        self.assertEqual(untouched_checks, [str(untouched_path)])

    def test_write_file_chunk_snapshot_clones_shared_targets_from_local(self):
        chunks = self._sample_chunks()
        chunks._lexical_manifest = inspection_index._lexical_manifest_for_chunks(chunks)
        chunks._index_manifest = {"input_0\x00service.py": "sha256:index"}
        chunks._semantic_document_signatures = {"chunk_001": "sha256:semantic"}
        chunks._chunk_ids = ("chunk_001",)
        chunks._file_key_by_chunk_id = {"chunk_001": "input_0\x00service.py"}
        chunks._chunks_by_file = {"input_0\x00service.py": [dict(chunks[0])]}

        with tempfile.TemporaryDirectory() as tempdir:
            cache_dir = Path(tempdir) / "cache"
            shared_cache_dir = Path(tempdir) / "shared-cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
                original_path_bytes_equal = inspection_index._path_bytes_equal
                shared_snapshot_path = inspection_index._shared_file_chunk_snapshot_path("sha256:repo", "sha256:build", create=True)
                shared_metadata_path = inspection_index._shared_file_chunk_snapshot_metadata_path("sha256:repo", "sha256:build", create=True)
                local_snapshot_path = inspection_index._file_chunk_snapshot_path(cache_dir)
                local_metadata_path = inspection_index._file_chunk_snapshot_metadata_path(cache_dir)

                def guarded_path_bytes_equal(path, payload):
                    if path in {shared_snapshot_path, shared_metadata_path}:
                        raise AssertionError("shared snapshot targets should be cloned from local without shared byte-compare")
                    return original_path_bytes_equal(path, payload)

                with mock.patch.object(inspection_index, "_path_bytes_equal", side_effect=guarded_path_bytes_equal):
                    inspection_index._write_file_chunk_snapshot(
                        cache_dir,
                        repository_state_fingerprint="sha256:repo",
                        build_config_digest="sha256:build",
                        chunks=chunks,
                        publish_shared=True,
                    )

            self.assertEqual(local_snapshot_path.read_bytes(), shared_snapshot_path.read_bytes())
            self.assertEqual(local_metadata_path.read_bytes(), shared_metadata_path.read_bytes())

    def test_load_file_chunk_snapshot_reuses_memory_cache_without_rereading_file(self):
        chunks = self._sample_chunks()
        chunks._lexical_manifest = inspection_index._lexical_manifest_for_chunks(chunks)
        chunks._index_manifest = {"input_0\x00service.py": "sha256:index"}
        chunks._semantic_document_signatures = {"chunk_001": "sha256:semantic"}
        chunks._chunk_ids = ("chunk_001",)
        chunks._file_key_by_chunk_id = {"chunk_001": "input_0\x00service.py"}
        chunks._chunks_by_file = {"input_0\x00service.py": [dict(chunks[0])]}

        with tempfile.TemporaryDirectory() as tempdir:
            cache_dir = Path(tempdir) / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            inspection_index._write_file_chunk_snapshot(
                cache_dir,
                repository_state_fingerprint="sha256:repo",
                build_config_digest="sha256:build",
                chunks=chunks,
                publish_shared=False,
            )

            first = inspection_index._load_file_chunk_snapshot(cache_dir)
            self.assertIsNotNone(first)

            original_read_text = Path.read_text
            snapshot_path = inspection_index._file_chunk_snapshot_path(cache_dir)

            def guarded_read_text(path_obj, *args, **kwargs):
                if Path(path_obj) == snapshot_path:
                    raise AssertionError("snapshot memory cache should avoid rereading snapshot file")
                return original_read_text(path_obj, *args, **kwargs)

            with mock.patch.object(Path, "read_text", new=guarded_read_text):
                second = inspection_index._load_file_chunk_snapshot(cache_dir)

            self.assertEqual(first, second)

    def test_write_preferred_discovery_working_manifest_clones_shared_target_from_local(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            local_path = root / "local" / "discovery.json"
            shared_path = root / "shared" / "discovery.json"
            roots = {
                "/repo": {
                    "git_top": "/repo",
                    "scope_rel": ".",
                    "scope_oid": "git:head",
                    "repository_state_fingerprint": "sha256:repo",
                    "filter_key": "filter",
                    "files": ["service.py"],
                    "dir_signatures": {},
                }
            }
            original_path_bytes_equal = inspection_index._path_bytes_equal

            def guarded_path_bytes_equal(path, payload):
                if path == shared_path:
                    raise AssertionError("shared discovery manifest should be cloned from local without shared byte-compare")
                return original_path_bytes_equal(path, payload)

            with mock.patch.object(inspection_index, "_path_bytes_equal", side_effect=guarded_path_bytes_equal):
                inspection_index._write_preferred_discovery_working_manifest(local_path, shared_path, roots)

            self.assertEqual(local_path.read_bytes(), shared_path.read_bytes())

    def test_git_dirty_manifest_entry_keys_reuses_subset_dirty_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "worker.py"
            source.write_text("def worker():\n    return 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "worker.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            source.write_text("def worker():\n    return 2\n", encoding="utf-8")

            item = {"id": "repo", "type": "repo", "classification": "internal", "path": root}
            discovered_files = [(item, source.resolve(strict=False), "worker.py")]
            namespaces = {id(item): "repo"}
            git_probe_cache = {}
            snapshot = inspection_index._scoped_git_status_snapshot(root, [], git_probe_cache=git_probe_cache)
            self.assertIsNotNone(snapshot)
            subset_paths = ("worker.py",)
            digest, dirty = inspection_index._status_subset_digest_and_dirty(
                snapshot["output"],
                subset_paths,
                parsed_entries=snapshot.get("parsed_entries"),
            )
            snapshot.setdefault("subset_dirty", {})[subset_paths] = (digest, tuple(sorted(dirty)))

            with mock.patch.object(
                inspection_index,
                "_status_subset_dirty",
                side_effect=AssertionError("subset_dirty cache should be reused"),
            ):
                result = inspection_index._git_dirty_manifest_entry_keys(
                    discovered_files,
                    namespaces,
                    git_probe_cache=git_probe_cache,
                )

            self.assertEqual(result, {"repo\000worker.py"})

    def test_cached_scoped_status_preserves_inventory_reuses_cached_verdict(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "worker.py"
            source.write_text("def worker():\n    return 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "worker.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            source.write_text("def worker():\n    return 2\n", encoding="utf-8")

            git_probe_cache = {}
            snapshot = inspection_index._scoped_git_status_snapshot(root, [], git_probe_cache=git_probe_cache)
            self.assertIsNotNone(snapshot)

            first = inspection_index._cached_scoped_status_preserves_inventory(
                ".",
                snapshot,
                ignored=set(inspection_index.DEFAULT_IGNORE_DIRS),
                ignored_paths=set(),
                top=root,
            )
            self.assertTrue(first)

            with mock.patch.object(Path, "resolve", side_effect=AssertionError("cached inventory-preserved verdict should avoid rescanning paths")):
                second = inspection_index._cached_scoped_status_preserves_inventory(
                    ".",
                    snapshot,
                    ignored=set(inspection_index.DEFAULT_IGNORE_DIRS),
                    ignored_paths=set(),
                    top=root,
                )

            self.assertTrue(second)

    def test_publish_shared_lexical_index_skips_republish_when_targets_are_current(self):
        chunks = self._sample_chunks()
        fingerprint = "sha256:test-fingerprint"
        build_config_digest = "sha256:test-build"

        with tempfile.TemporaryDirectory() as tempdir:
            cache_dir = Path(tempdir) / "cache"
            shared_cache_dir = Path(tempdir) / "shared-cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            working_path = inspection_index._working_index_path(cache_dir)
            working_manifest_path = inspection_index._working_manifest_path(cache_dir)

            inspection_index._rebuild_working_lexical_index(working_path, chunks, fingerprint)
            manifest = inspection_index._lexical_manifest_for_chunks(chunks)
            manifest["__meta__"] = {
                "fingerprint": fingerprint,
                "chunk_count": len(chunks),
                "build_config_digest": build_config_digest,
            }
            inspection_index._write_lexical_working_manifest(
                working_manifest_path,
                manifest,
                fingerprint=fingerprint,
                chunk_count=len(chunks),
                build_config_digest=build_config_digest,
            )

            with mock.patch.dict(
                os.environ,
                {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)},
                clear=False,
            ):
                with mock.patch.object(
                    inspection_index.shutil,
                    "copy2",
                    wraps=inspection_index.shutil.copy2,
                ) as copy_mock:
                    inspection_index._publish_shared_lexical_index(
                        cache_dir,
                        fingerprint,
                        build_config_digest=build_config_digest,
                    )
                    first_publish_copy_count = copy_mock.call_count
                    self.assertGreaterEqual(first_publish_copy_count, 2)

                    inspection_index._publish_shared_lexical_index(
                        cache_dir,
                        fingerprint,
                        build_config_digest=build_config_digest,
                    )
                    self.assertEqual(copy_mock.call_count, first_publish_copy_count)


class FakeClient:
    def __init__(self, record, factory):
        self.record = record
        self.factory = factory

    @staticmethod
    def _vector(text):
        lowered = text.lower()
        return [
            1.0,
            float(lowered.count("retry")),
            float(lowered.count("mcp")),
            float(lowered.count("service")),
        ]

    def embed(self, texts):
        self.factory.embed_calls.append((self.record["tier"], len(texts)))
        return [self._vector(text) for text in texts]

    def rerank(self, query, documents):
        self.factory.rerank_calls.append((self.record["tier"], len(documents)))
        terms = {term.lower() for term in query.split()}
        return [
            float(sum(document.lower().count(term) for term in terms)) + (1.0 / (index + 1))
            for index, document in enumerate(documents)
        ]

    def semantic_search(self, query, chunks, fingerprint, limit):
        self.factory.semantic_search_calls.append((self.record["tier"], limit))
        query_terms = {term.lower().strip(".,?!") for term in query.split()}
        scored = []
        for chunk in chunks:
            text = f"{chunk['path']} {chunk.get('symbol', '')} {chunk['content']}".lower()
            score = float(sum(text.count(term) for term in query_terms))
            scored.append((chunk["chunk_id"], score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [
            {"chunk_id": chunk_id, "score": score, "rank": index, "source": "semantic"}
            for index, (chunk_id, score) in enumerate(scored[:limit], start=1)
        ]

    def ensure_semantic_index(self, chunks, fingerprint, *, sync_plan=None):
        self.factory.ensure_semantic_index_calls.append(
            (
                self.record["tier"],
                len(chunks),
                tuple(sorted(str(value) for value in ((sync_plan or {}).get("changed_ids") or ()))),
            )
        )
        return {
            "cache_hit": True,
            "document_count": len(chunks),
            "embedded_documents": 0,
            "reused_documents": 0,
        }

    def chat(self, messages, response_schema):
        tier = self.record["tier"]
        self.factory.chat_calls.append(tier)
        scripted = self.factory.chat_results.get(tier) or []
        if scripted:
            result = scripted.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        return {
            "answer": "The retry flow is implemented by the cited service helper.",
            "findings": [{"summary": "The helper owns retry submission.", "evidence_refs": ["ev_001"]}],
        }


class RepoFixture(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "service.py").write_text(
            "def retry_job(job_id):\n    return submit_job(job_id)\n\ndef submit_job(job_id):\n    return job_id\n",
            encoding="utf-8",
        )
        (self.root / "mcp.go").write_text(
            "package mcp\n\nfunc InspectRepo(query string) string {\n\treturn query\n}\n",
            encoding="utf-8",
        )
        self.discovered = [
            {"id": "input_0", "type": "repo", "classification": "internal", "path": self.root}
        ]

    def tearDown(self):
        self.temp_dir.cleanup()

    def run_pipeline(self, *, mode="auto", services=None, factory=None, constraints=None, task_params=None, discovered=None):
        return inspection_pipeline.run_inspection(
            self.discovered if discovered is None else discovered,
            "Trace the retry_job service call chain",
            mode=mode,
            constraints=constraints,
            task_params=(
                {"index_cache_dir": str(self.root / ".broker" / "inspection-test")}
                if task_params is None
                else task_params
            ),
            services=[] if services is None else services,
            client_factory=factory or FakeClientFactory(),
            output_dir=self.root / "out",
        )["payload"]


class SemanticDiagnosticsFactory(FakeClientFactory):
    def __init__(self, *, cache_hit=False, embedded_documents=0, reused_documents=0, chat_results=None):
        super().__init__(chat_results=chat_results)
        self.cache_hit = cache_hit
        self.embedded_documents = embedded_documents
        self.reused_documents = reused_documents

    def __call__(self, record):
        return SemanticDiagnosticsClient(record, self)


class SemanticDiagnosticsClient(FakeClient):
    def ensure_semantic_index(self, chunks, fingerprint, *, sync_plan=None):
        self.factory.ensure_semantic_index_calls.append(
            (
                self.record["tier"],
                len(chunks),
                tuple(sorted(str(value) for value in ((sync_plan or {}).get("changed_ids") or ()))),
            )
        )
        return {
            "cache_hit": self.factory.cache_hit,
            "document_count": len(chunks),
            "embedded_documents": self.factory.embedded_documents,
            "reused_documents": self.factory.reused_documents,
        }


class RequestContractTests(unittest.TestCase):
    def test_query_and_mode_validation(self):
        with self.assertRaisesRegex(ValueError, "non-empty query"):
            inspection_pipeline.validate_request(" ", "auto")
        with self.assertRaisesRegex(ValueError, "auto, evidence, answer"):
            inspection_pipeline.validate_request("query", "fast")
        self.assertEqual(inspection_pipeline.validate_request(" query ", None), ("query", "auto"))

    def test_prepare_prefetched_state_uses_broker_repository_fingerprint_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "service.py").write_text("def handler():\n    return 'ok'\n", encoding="utf-8")
            discovered = [
                {
                    "id": "input_0",
                    "type": "repo",
                    "classification": "internal",
                    "path": repo_root,
                }
            ]
            with mock.patch.object(
                inspection_hotpath,
                "repository_fingerprint",
                side_effect=AssertionError("repository_fingerprint should not run when broker hint is present"),
            ):
                state = inspection_hotpath.prepare_prefetched_state(
                    discovered,
                    "trace handler",
                    mode="answer",
                    constraints={},
                    task_params={
                        "_broker_repository_state_fingerprint": "git:test-fingerprint",
                        "_broker_repository_state_fingerprint_source": "request_cache",
                    },
                    execution_plan={},
                    output_dir=repo_root / "out",
                )
        self.assertEqual(state["repository_state_fingerprint"], "git:test-fingerprint")
        self.assertEqual(
            state["fingerprint_state"],
            [
                {
                    "kind": "broker_hint",
                    "source": "request_cache",
                    "fingerprint": "git:test-fingerprint",
                }
            ],
        )

    def test_prepare_prefetched_state_uses_input_manifest_content_hash_hint(self):
        discovered = [
            {
                "id": "input_0",
                "type": "repo",
                "classification": "internal",
                "path": Path("/tmp/repo"),
                "content_hash": "git:manifest-fingerprint",
            }
        ]

        with mock.patch.object(
            inspection_hotpath,
            "repository_fingerprint",
            side_effect=AssertionError("repository_fingerprint should not run when input manifest content_hash is present"),
        ):
            state = inspection_hotpath.prepare_prefetched_state(
                discovered,
                "trace repo timeout",
                mode="evidence",
                constraints={},
                task_params={},
                execution_plan={"repo_inspection_cache_path": "/tmp/cache"},
                output_dir="/tmp/out",
            )

        self.assertEqual(state["repository_state_fingerprint"], "git:manifest-fingerprint")
        self.assertEqual(
            state["fingerprint_state"],
            [
                {
                    "kind": "input_manifest",
                    "source": "input_manifest",
                    "fingerprint": "git:manifest-fingerprint",
                }
            ],
        )

    def test_prepare_prefetched_state_keeps_hint_fingerprint_when_snapshot_cache_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            repo_root = temp_root / "repo"
            cache_dir = temp_root / "cache"
            output_dir = temp_root / "out"
            repo_root.mkdir()
            output_dir.mkdir()
            source = repo_root / "service.py"
            source.write_text("def handler():\n    return 'ok'\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q", str(repo_root)], check=True)
            subprocess.run(["git", "-C", str(repo_root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(repo_root), "config", "user.name", "Test"], check=True)
            subprocess.run(["git", "-C", str(repo_root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(repo_root), "commit", "-qm", "initial"], check=True)
            discovered = [
                {
                    "id": "input_0",
                    "type": "repo",
                    "classification": "internal",
                    "path": repo_root,
                    "uri": repo_root.as_uri(),
                }
            ]

            actual_fingerprint, actual_state = inspection_index.repository_fingerprint(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir, output_dir},
            )
            inspection_index.build_syntax_chunks(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir, output_dir},
                repository_state_fingerprint=actual_fingerprint,
                repository_fingerprint_state=actual_state,
            )

            with mock.patch.object(
                inspection_hotpath,
                "repository_fingerprint",
                side_effect=AssertionError("hinted request should not recompute local repository fingerprint"),
            ):
                state = inspection_hotpath.prepare_prefetched_state(
                    discovered,
                    "trace handler",
                    mode="evidence",
                    constraints={},
                    task_params={
                        "_broker_repository_state_fingerprint": "git:request-cache-fingerprint",
                        "_broker_repository_state_fingerprint_source": "request_cache",
                    },
                    execution_plan={"repo_inspection_cache_path": str(cache_dir)},
                    output_dir=str(output_dir),
                )

        self.assertEqual(state["repository_state_fingerprint"], "git:request-cache-fingerprint")
        self.assertEqual(
            state["fingerprint_state"],
            [
                {
                    "kind": "broker_hint",
                    "source": "request_cache",
                    "fingerprint": "git:request-cache-fingerprint",
                }
            ],
        )
        self.assertIsNone(state["cached_chunk_snapshot_metadata"])
        self.assertEqual(state["prefetch_state_source"], "fresh")

    def test_broker_fingerprint_hint_reuses_existing_manifest_without_rebuilding_all_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "service.py").write_text(
                "def retry_job(job_id):\n    return submit_job(job_id)\n\ndef submit_job(job_id):\n    return job_id\n",
                encoding="utf-8",
            )
            (root / "mcp.go").write_text(
                "package mcp\n\nfunc InspectRepo(query string) string {\n\treturn query\n}\n",
                encoding="utf-8",
            )
            discovered = [
                {"id": "input_0", "type": "repo", "classification": "internal", "path": root}
            ]
            cache_dir = root / ".broker" / "inspection-test"
            output_dir = root / "out"

            first_factory = SemanticDiagnosticsFactory()
            first = inspection_pipeline.run_inspection(
                discovered,
                "Trace the retry_job service call chain",
                mode="evidence",
                task_params={"index_cache_dir": str(cache_dir)},
                services=all_services(),
                client_factory=first_factory,
                output_dir=output_dir,
            )["payload"]

            second_factory = SemanticDiagnosticsFactory()
            with mock.patch.object(
                inspection_pipeline,
                "repository_fingerprint",
                side_effect=AssertionError("broker fingerprint hint should avoid local repository fingerprint recomputation"),
            ):
                second = inspection_pipeline.run_inspection(
                    discovered,
                    "Trace the submit_job service call chain",
                    mode="evidence",
                    task_params={
                        "index_cache_dir": str(cache_dir),
                        "_broker_repository_state_fingerprint": "git:broker-cache-fingerprint",
                        "_broker_repository_state_fingerprint_source": "request_cache",
                    },
                    services=all_services(),
                    client_factory=second_factory,
                    output_dir=output_dir,
                )["payload"]

        self.assertTrue(str((first.get("provenance") or {}).get("repository_fingerprint") or ""))
        self.assertEqual(second["retrieval"]["fingerprint_sources"], ["broker_hint"])
        self.assertEqual(second["retrieval"]["setup_timings_ms"]["repository_fingerprint_ms"], 0.0)
        self.assertEqual(second["retrieval"]["chunk_cache_total_files"], 2)
        self.assertEqual(second["retrieval"]["chunk_cache_reused_files"], 2)
        self.assertEqual(second["retrieval"]["chunk_cache_rebuilt_files"], 0)



    def test_prepare_prefetched_state_skips_input_manifest_content_hash_hint_when_custom_exclusions_present(self):
        discovered = [
            {
                "id": "input_0",
                "type": "repo",
                "classification": "internal",
                "path": Path("/tmp/repo"),
                "content_hash": "git:manifest-fingerprint",
            }
        ]

        with mock.patch.object(
            inspection_hotpath,
            "repository_fingerprint",
            return_value=("git:actual", [{"kind": "git", "head": "abc"}]),
        ) as fingerprint_mock:
            state = inspection_hotpath.prepare_prefetched_state(
                discovered,
                "trace repo timeout",
                mode="evidence",
                constraints={},
                task_params={"excluded_dir_names": ["vendor"]},
                execution_plan={"repo_inspection_cache_path": "/tmp/cache"},
                output_dir="/tmp/out",
            )

        fingerprint_mock.assert_called_once()
        self.assertEqual(state["repository_state_fingerprint"], "git:actual")


    def test_inspect_repo_worker_discover_inputs_preserves_content_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir) / "repo"
            repo_root.mkdir()
            discovered = inspect_repo_worker.discover_inputs(
                [
                    {
                        "type": "repo",
                        "uri": repo_root.as_uri(),
                        "classification": "internal",
                        "content_hash": "git:manifest-fingerprint",
                    }
                ]
            )

        self.assertEqual(len(discovered), 1)
        self.assertEqual(discovered[0]["content_hash"], "git:manifest-fingerprint")

    def test_prepare_prefetched_state_reuses_snapshot_build_context_in_process_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir) / "repo"
            repo_root.mkdir()
            discovered = [
                {
                    "id": "input_0",
                    "type": "repo",
                    "classification": "internal",
                    "path": repo_root,
                    "content_hash": "git:manifest-fingerprint",
                }
            ]

            inspection_hotpath._SNAPSHOT_BUILD_CONTEXT_CACHE.clear()
            first = inspection_hotpath.prepare_prefetched_state(
                discovered,
                "trace repo timeout",
                mode="evidence",
                constraints={},
                task_params={},
                execution_plan={"repo_inspection_cache_path": str(Path(temp_dir) / "cache")},
                output_dir=str(Path(temp_dir) / "out"),
            )

            with mock.patch.object(
                inspection_hotpath,
                "_chunk_build_config_digest",
                side_effect=AssertionError("second identical prefetch should reuse cached snapshot build context"),
            ):
                second = inspection_hotpath.prepare_prefetched_state(
                    discovered,
                    "trace repo timeout",
                    mode="evidence",
                    constraints={},
                    task_params={},
                    execution_plan={"repo_inspection_cache_path": str(Path(temp_dir) / "cache")},
                    output_dir=str(Path(temp_dir) / "out"),
                )

        self.assertEqual(first["build_config_digest"], second["build_config_digest"])

    def test_build_config_digest_ignores_transient_cache_and_output_roots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            repo_root = temp_root / "repo"
            repo_root.mkdir()
            discovered = [
                {
                    "id": "input_0",
                    "type": "repo",
                    "uri": "file:///stable/repo",
                    "classification": "internal",
                    "path": repo_root,
                    "content_hash": "git:manifest-fingerprint",
                }
            ]

            first_discovered, first_namespaces, _first_effective_excluded_paths, first_digest = inspection_hotpath._snapshot_build_context(
                discovered,
                set(),
                cache_dir=temp_root / "cache-a",
                excluded_paths={temp_root / "cache-a", temp_root / "out-a"},
                transient_excluded_paths={temp_root / "out-a"},
            )
            second_discovered, second_namespaces, _second_effective_excluded_paths, second_digest = inspection_hotpath._snapshot_build_context(
                discovered,
                set(),
                cache_dir=temp_root / "cache-b",
                excluded_paths={temp_root / "cache-b", temp_root / "out-b"},
                transient_excluded_paths={temp_root / "out-b"},
            )

        self.assertEqual(first_namespaces[id(first_discovered[0])], second_namespaces[id(second_discovered[0])])
        self.assertEqual(first_digest, second_digest)

    def test_build_config_digest_uses_stable_uri_instead_of_absolute_repo_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            first_root = temp_root / "repo-one"
            second_root = temp_root / "repo-two"
            first_root.mkdir()
            second_root.mkdir()
            first_discovered = [
                {
                    "id": "repo",
                    "type": "repo",
                    "uri": "file:///stable/repo",
                    "classification": "internal",
                    "path": first_root,
                    "content_hash": "git:manifest-fingerprint",
                }
            ]
            second_discovered = [
                {
                    "id": "repo",
                    "type": "repo",
                    "uri": "file:///stable/repo",
                    "classification": "internal",
                    "path": second_root,
                    "content_hash": "git:manifest-fingerprint",
                }
            ]

            _first_discovered, _first_namespaces, _first_effective_excluded_paths, first_digest = inspection_hotpath._snapshot_build_context(
                first_discovered,
                set(),
                cache_dir=temp_root / "cache-a",
                excluded_paths={temp_root / "cache-a"},
            )
            _second_discovered, _second_namespaces, _second_effective_excluded_paths, second_digest = inspection_hotpath._snapshot_build_context(
                second_discovered,
                set(),
                cache_dir=temp_root / "cache-b",
                excluded_paths={temp_root / "cache-b"},
            )

        self.assertEqual(first_digest, second_digest)

    def test_prepare_prefetched_state_reuses_prefetch_result_in_process_cache_for_identical_hinted_request(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir) / "repo"
            repo_root.mkdir()
            discovered = [
                {
                    "id": "input_0",
                    "type": "repo",
                    "classification": "internal",
                    "path": repo_root,
                    "content_hash": "git:manifest-fingerprint",
                }
            ]
            cache_dir = Path(temp_dir) / "cache"
            output_dir = Path(temp_dir) / "out"

            inspection_hotpath._PREFETCH_STATE_CACHE.clear()
            first = inspection_hotpath.prepare_prefetched_state(
                discovered,
                "trace repo timeout",
                mode="evidence",
                constraints={},
                task_params={},
                execution_plan={"repo_inspection_cache_path": str(cache_dir)},
                output_dir=str(output_dir),
            )

            with mock.patch.object(
                inspection_hotpath,
                "_snapshot_build_context",
                side_effect=AssertionError("second identical hinted prefetch should return before snapshot build context"),
            ), mock.patch.object(
                inspection_hotpath,
                "load_cached_chunk_snapshot_metadata",
                side_effect=AssertionError("second identical hinted prefetch should reuse cached prefetched state"),
            ), mock.patch.object(
                inspection_hotpath,
                "load_query_stage_cache",
                side_effect=AssertionError("second identical hinted prefetch should reuse cached prefetched state"),
            ):
                second = inspection_hotpath.prepare_prefetched_state(
                    discovered,
                    "trace repo timeout",
                    mode="evidence",
                    constraints={},
                    task_params={},
                    execution_plan={"repo_inspection_cache_path": str(cache_dir)},
                    output_dir=str(output_dir),
                )

        self.assertEqual(first["repository_state_fingerprint"], second["repository_state_fingerprint"])
        self.assertEqual(first["build_config_digest"], second["build_config_digest"])
        self.assertFalse(bool(first.get("prefetch_state_cache_hit")))
        self.assertEqual(first.get("prefetch_state_source"), "fresh")
        self.assertTrue(bool(second.get("prefetch_state_cache_hit")))
        self.assertEqual(second.get("prefetch_state_source"), "early_process_cache")

    def test_prepare_prefetched_state_reuses_prefetch_result_across_different_output_dirs_for_identical_hinted_request(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir) / "repo"
            repo_root.mkdir()
            discovered = [
                {
                    "id": "input_0",
                    "type": "repo",
                    "classification": "internal",
                    "path": repo_root,
                    "content_hash": "git:manifest-fingerprint",
                }
            ]
            cache_dir = Path(temp_dir) / "cache"
            first_output_dir = Path(temp_dir) / "out-a"
            second_output_dir = Path(temp_dir) / "out-b"

            inspection_hotpath._PREFETCH_STATE_CACHE.clear()
            first = inspection_hotpath.prepare_prefetched_state(
                discovered,
                "trace repo timeout",
                mode="evidence",
                constraints={},
                task_params={},
                execution_plan={"repo_inspection_cache_path": str(cache_dir)},
                output_dir=str(first_output_dir),
            )

            with mock.patch.object(
                inspection_hotpath,
                "_snapshot_build_context",
                side_effect=AssertionError("second identical hinted prefetch with different output dir should return before snapshot build context"),
            ), mock.patch.object(
                inspection_hotpath,
                "load_cached_chunk_snapshot_metadata",
                side_effect=AssertionError("second identical hinted prefetch with different output dir should reuse cached prefetched state"),
            ), mock.patch.object(
                inspection_hotpath,
                "load_query_stage_cache",
                side_effect=AssertionError("second identical hinted prefetch with different output dir should reuse cached prefetched state"),
            ):
                second = inspection_hotpath.prepare_prefetched_state(
                    discovered,
                    "trace repo timeout",
                    mode="evidence",
                    constraints={},
                    task_params={},
                    execution_plan={"repo_inspection_cache_path": str(cache_dir)},
                    output_dir=str(second_output_dir),
                )

        self.assertEqual(first["repository_state_fingerprint"], second["repository_state_fingerprint"])
        self.assertEqual(first["build_config_digest"], second["build_config_digest"])
        self.assertFalse(bool(first.get("prefetch_state_cache_hit")))
        self.assertEqual(first.get("prefetch_state_source"), "fresh")
        self.assertTrue(bool(second.get("prefetch_state_cache_hit")))
        self.assertEqual(second.get("prefetch_state_source"), "early_process_cache")

    def test_hotpath_private_cache_dir_skips_repeated_chmod_for_same_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            inspection_hotpath._PRIVATE_CACHE_DIR_READY.clear()
            inspection_hotpath._private_cache_dir(cache_dir)

            chmod_calls = 0
            path_type = type(cache_dir)
            original_chmod = path_type.chmod

            def counted_chmod(self, mode, *args, **kwargs):
                nonlocal chmod_calls
                if self == cache_dir:
                    chmod_calls += 1
                return original_chmod(self, mode, *args, **kwargs)

            with mock.patch.object(path_type, "chmod", counted_chmod):
                inspection_hotpath._private_cache_dir(cache_dir)

        self.assertEqual(chmod_calls, 0)

    def test_lexical_working_index_is_current_does_not_create_cache_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            self.assertFalse(cache_dir.exists())

            current = inspection_index.lexical_working_index_is_current(
                cache_dir,
                "sha256:test",
                0,
            )

        self.assertFalse(current)
        self.assertFalse(cache_dir.exists())

    def test_index_private_cache_dir_skips_repeated_chmod_for_same_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            inspection_index._PRIVATE_CACHE_DIR_READY.clear()
            inspection_index._private_cache_dir(cache_dir)

            chmod_calls = 0
            path_type = type(cache_dir)
            original_chmod = path_type.chmod

            def counted_chmod(self, mode, *args, **kwargs):
                nonlocal chmod_calls
                if self == cache_dir:
                    chmod_calls += 1
                return original_chmod(self, mode, *args, **kwargs)

            with mock.patch.object(path_type, "chmod", counted_chmod):
                inspection_index._private_cache_dir(cache_dir)

        self.assertEqual(chmod_calls, 0)

    def test_hotpath_load_cached_chunk_snapshot_metadata_reuses_in_process_memory_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            cache_dir = temp_root / "cache"
            (root / "worker.py").write_text("def worker():\n    return 1\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            repository_fp = "sha256:hotpath-local-snapshot-memory"

            inspection_index.build_syntax_chunks(
                discovered,
                cache_dir=cache_dir,
                repository_state_fingerprint=repository_fp,
                return_diagnostics=True,
            )
            inspection_hotpath._SNAPSHOT_METADATA_CACHE.clear()
            first = inspection_hotpath.load_cached_chunk_snapshot_metadata(
                discovered,
                cache_dir=cache_dir,
                repository_state_fingerprint=repository_fp,
            )

            metadata_path = inspection_hotpath._file_chunk_snapshot_metadata_path(cache_dir)
            original_read_text = Path.read_text

            def guarded_read_text(path_self, *args, **kwargs):
                if path_self == metadata_path:
                    raise AssertionError("second hotpath metadata load should reuse in-process cache")
                return original_read_text(path_self, *args, **kwargs)

            with mock.patch.object(Path, "read_text", autospec=True, side_effect=guarded_read_text):
                second = inspection_hotpath.load_cached_chunk_snapshot_metadata(
                    discovered,
                    cache_dir=cache_dir,
                    repository_state_fingerprint=repository_fp,
                )

        self.assertEqual(first, second)

    def test_hotpath_load_cached_chunk_snapshot_metadata_reuses_shared_in_process_memory_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            shared_cache_dir = temp_root / "shared-cache"
            first_cache_dir = temp_root / "cache-one"
            second_cache_dir = temp_root / "cache-two"
            (root / "worker.py").write_text("def worker():\n    return 1\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            repository_fp = "sha256:hotpath-shared-snapshot-memory"

            with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
                inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=first_cache_dir,
                    repository_state_fingerprint=repository_fp,
                    return_diagnostics=True,
                )
                inspection_hotpath._SNAPSHOT_METADATA_CACHE.clear()
                first = inspection_hotpath.load_cached_chunk_snapshot_metadata(
                    discovered,
                    cache_dir=second_cache_dir,
                    repository_state_fingerprint=repository_fp,
                )

                shared_metadata_path = inspection_hotpath._shared_file_chunk_snapshot_metadata_path(
                    repository_fp,
                    str(first.get("build_config_digest") or ""),
                    create=False,
                )
                original_read_text = Path.read_text

                def guarded_read_text(path_self, *args, **kwargs):
                    if path_self == shared_metadata_path:
                        raise AssertionError("second hotpath shared metadata load should reuse in-process cache")
                    return original_read_text(path_self, *args, **kwargs)

                with mock.patch.object(Path, "read_text", autospec=True, side_effect=guarded_read_text):
                    second = inspection_hotpath.load_cached_chunk_snapshot_metadata(
                        discovered,
                        cache_dir=second_cache_dir,
                        repository_state_fingerprint=repository_fp,
                    )

        self.assertEqual(first, second)

    def test_hotpath_load_cached_chunk_snapshot_metadata_local_hit_skips_working_manifest_read(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            cache_dir = temp_root / "cache"
            (root / "worker.py").write_text("def worker():\n    return 1\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            repository_fp = "sha256:hotpath-local-snapshot"

            inspection_index.build_syntax_chunks(
                discovered,
                cache_dir=cache_dir,
                repository_state_fingerprint=repository_fp,
                return_diagnostics=True,
            )

            with mock.patch.object(
                inspection_hotpath,
                "_load_file_chunk_working_manifest",
                side_effect=AssertionError("matching hotpath local snapshot metadata should not read the working manifest"),
            ):
                metadata = inspection_hotpath.load_cached_chunk_snapshot_metadata(
                    discovered,
                    cache_dir=cache_dir,
                    repository_state_fingerprint=repository_fp,
                )

        self.assertIsNotNone(metadata)
        self.assertEqual(int(metadata["total_files"]), 1)

    def test_hotpath_load_cached_chunk_snapshot_metadata_shared_hit_skips_working_manifest_read(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            shared_cache_dir = temp_root / "shared-cache"
            first_cache_dir = temp_root / "cache-one"
            second_cache_dir = temp_root / "cache-two"
            (root / "worker.py").write_text("def worker():\n    return 1\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            repository_fp = "sha256:hotpath-shared-snapshot"

            with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
                inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=first_cache_dir,
                    repository_state_fingerprint=repository_fp,
                    return_diagnostics=True,
                )

                with mock.patch.object(
                    inspection_hotpath,
                    "_load_file_chunk_working_manifest",
                    side_effect=AssertionError("shared hotpath snapshot metadata hit should not read the working manifest"),
                ):
                    metadata = inspection_hotpath.load_cached_chunk_snapshot_metadata(
                        discovered,
                        cache_dir=second_cache_dir,
                        repository_state_fingerprint=repository_fp,
                    )

        self.assertIsNotNone(metadata)
        self.assertEqual(int(metadata["total_files"]), 1)

    def test_hotpath_load_query_stage_cache_reuses_in_process_memory_cache_on_second_hit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_key = "sha256:test-hotpath-query-stage-memory"
            local_cache_dir = root / ".broker" / "hotpath-inspection-local"
            local_path = inspection_hotpath._query_stage_cache_path(local_cache_dir, cache_key)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text(
                json.dumps(
                    {
                        "schema": inspection_hotpath.QUERY_STAGE_CACHE_SCHEMA,
                        "retrieval_signature": {"tier": "cpu-lexical-fallback"},
                        "retrieval_quality": "gpu",
                        "rerank_quality": "gpu",
                        "ranked": [{"chunk_id": "chunk-1", "rrf_score": 1.0, "rank": 1}],
                        "selected": [{"chunk_id": "chunk-1", "rank": 1}],
                        "evidence": [{"id": "ev_001", "chunk_id": "chunk-1", "path": "worker.py", "content": "def worker(): pass"}],
                        "evidence_budget_trimmed": False,
                        "answer": "",
                        "findings": [],
                        "warnings": [],
                        "provenance": {},
                        "runtime_attempts": [],
                        "synthesis_quality": "not_requested",
                    },
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )

            inspection_hotpath._QUERY_STAGE_MEMORY_CACHE.clear()
            first = inspection_hotpath.load_query_stage_cache(local_cache_dir, cache_key)

            original_read_text = Path.read_text

            def guarded_read_text(path_self, *args, **kwargs):
                if path_self == local_path:
                    raise AssertionError("second hotpath query-stage cache hit should not reread the cache file")
                return original_read_text(path_self, *args, **kwargs)

            with mock.patch.object(Path, "read_text", autospec=True, side_effect=guarded_read_text):
                second = inspection_hotpath.load_query_stage_cache(local_cache_dir, cache_key)

        self.assertIsNotNone(first)
        self.assertEqual(first, second)

    def test_hotpath_load_query_stage_cache_does_not_touch_mtime_on_hit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_key = "sha256:test-hotpath-query-stage-no-touch"
            local_cache_dir = root / ".broker" / "hotpath-inspection-local"
            local_path = inspection_hotpath._query_stage_cache_path(local_cache_dir, cache_key)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text(
                json.dumps(
                    {
                        "schema": inspection_hotpath.QUERY_STAGE_CACHE_SCHEMA,
                        "retrieval_signature": {"tier": "cpu-lexical-fallback"},
                        "retrieval_quality": "gpu",
                        "rerank_quality": "gpu",
                        "ranked": [{"chunk_id": "chunk-1", "rrf_score": 1.0, "rank": 1}],
                        "selected": [{"chunk_id": "chunk-1", "rank": 1}],
                        "evidence": [{"id": "ev_001", "chunk_id": "chunk-1", "path": "worker.py", "content": "def worker(): pass"}],
                        "evidence_budget_trimmed": False,
                        "answer": "",
                        "findings": [],
                        "warnings": [],
                        "provenance": {},
                        "runtime_attempts": [],
                        "synthesis_quality": "not_requested",
                    },
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            inspection_hotpath._QUERY_STAGE_MEMORY_CACHE.clear()
            first = inspection_hotpath.load_query_stage_cache(local_cache_dir, cache_key)
            self.assertIsNotNone(first)

            with mock.patch.object(inspection_hotpath.os, "utime", side_effect=AssertionError("query-stage cache hit should not touch mtime")):
                second = inspection_hotpath.load_query_stage_cache(local_cache_dir, cache_key)

        self.assertEqual(first, second)

    def test_hotpath_load_query_stage_cache_reuses_shared_in_process_memory_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            shared_cache_dir = root / ".broker" / "hotpath-shared-query-stage"
            cache_key = "sha256:test-hotpath-shared-query-stage-memory"
            shared_payload = {
                "schema": inspection_hotpath.QUERY_STAGE_CACHE_SCHEMA,
                "retrieval_signature": {"tier": "cpu-lexical-fallback"},
                "retrieval_quality": "gpu",
                "rerank_quality": "gpu",
                "ranked": [{"chunk_id": "chunk-1", "rrf_score": 1.0, "rank": 1}],
                "selected": [{"chunk_id": "chunk-1", "rank": 1}],
                "evidence": [{"id": "ev_001", "chunk_id": "chunk-1", "path": "worker.py", "content": "def worker(): pass"}],
                "evidence_budget_trimmed": False,
                "answer": "",
                "findings": [],
                "warnings": [],
                "provenance": {},
                "runtime_attempts": [],
                "synthesis_quality": "not_requested",
            }
            local_cache_dir = root / ".broker" / "hotpath-inspection-local-two"

            with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
                shared_path = inspection_hotpath._shared_query_stage_cache_path(cache_key)
                shared_path.parent.mkdir(parents=True, exist_ok=True)
                shared_path.write_text(json.dumps(shared_payload, separators=(",", ":")), encoding="utf-8")

                inspection_hotpath._QUERY_STAGE_MEMORY_CACHE.clear()
                first = inspection_hotpath.load_query_stage_cache(local_cache_dir, cache_key)

                original_read_text = Path.read_text

                def guarded_read_text(path_self, *args, **kwargs):
                    if path_self == shared_path:
                        raise AssertionError("second hotpath shared query-stage cache hit should not reread the shared cache file")
                    return original_read_text(path_self, *args, **kwargs)

                with mock.patch.object(Path, "read_text", autospec=True, side_effect=guarded_read_text):
                    second = inspection_hotpath.load_query_stage_cache(local_cache_dir, cache_key)

        self.assertIsNotNone(first)
        self.assertEqual(first, second)

    def test_prepare_prefetched_state_uses_query_stage_alias_without_loading_snapshot_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            shared_cache_dir = temp_root / "shared-cache"
            local_cache_dir = temp_root / "local-cache"
            output_dir = temp_root / "out"
            output_dir.mkdir()
            (root / "worker.py").write_text("def retry_job(job_id):\n    return job_id\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            query = "trace retry_job"
            budgets = inspection_pipeline.normalize_token_budgets({})

            with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
                repository_fp, _ = inspection_index.repository_fingerprint(
                    discovered,
                    excluded_paths=[local_cache_dir, output_dir],
                    cache_dir=local_cache_dir,
                )
                _discovered, _namespaces, _effective_excluded_paths, build_config_digest = inspection_hotpath._snapshot_build_context(
                    discovered,
                    set(),
                    cache_dir=local_cache_dir,
                    excluded_paths=inspection_hotpath.exclusion_paths_for_execution(
                        {"repo_inspection_cache_path": str(local_cache_dir)},
                        output_dir,
                    ),
                    transient_excluded_paths=inspection_hotpath.transient_excluded_paths_for_execution(output_dir),
                )
                cache_key = inspection_hotpath.query_stage_cache_key(
                    query,
                    "sha256:index-fp",
                    {"tier": "cpu-lexical-fallback", "mode": "lexical_fallback"},
                    budgets,
                )
                inspection_pipeline._write_query_stage_cache(
                    local_cache_dir,
                    query,
                    cache_key,
                    {"tier": "cpu-lexical-fallback", "mode": "lexical_fallback"},
                    [{"chunk_id": "chunk-1", "rrf_score": 1.0, "rank": 1}],
                    [{"chunk_id": "chunk-1", "rank": 1}],
                    [{
                        "id": "ev_001",
                        "chunk_id": "chunk-1",
                        "path": "worker.py",
                        "content": "def retry_job(job_id):\n    return job_id\n",
                        "excerpt": "def retry_job(job_id):\n    return job_id\n",
                        "language": "python",
                        "rank": 1,
                        "symbol": "retry_job",
                        "source_refs": [{
                            "path": "worker.py",
                            "line_start": 1,
                            "line_end": 2,
                            "content_hash": "sha256:test",
                            "input_id": "repo",
                            "source_namespace": "repo",
                        }],
                    }],
                    False,
                    retrieval_quality="cpu",
                    rerank_quality="cpu",
                    repository_state_fingerprint=repository_fp,
                    build_config_digest=build_config_digest,
                    index_fingerprint="sha256:index-fp",
                    total_files=1,
                    chunk_count=1,
                    budgets=budgets,
                )

                with mock.patch.object(
                    inspection_hotpath,
                    "load_cached_chunk_snapshot_metadata",
                    side_effect=AssertionError("query-stage alias hit should avoid snapshot metadata load"),
                ):
                    state = inspection_hotpath.prepare_prefetched_state(
                        discovered,
                        query,
                        mode="evidence",
                        constraints={},
                        task_params={
                            "_broker_repository_state_fingerprint": repository_fp,
                            "_broker_repository_state_fingerprint_source": "request_cache",
                        },
                        execution_plan={"repo_inspection_cache_path": str(local_cache_dir)},
                        output_dir=output_dir,
                    )

        self.assertIsNotNone(state["cached_query_stage"])
        self.assertIsNotNone(state.get("cached_lexical_fallback_run"))
        self.assertEqual(state["fingerprint"], "sha256:index-fp")
        self.assertEqual(int(state["cached_chunk_snapshot_metadata"]["total_files"]), 1)
        self.assertEqual(int(getattr(state["metadata_chunks"], "_chunk_count", 0)), 1)

    def test_prepare_prefetched_state_reuses_cached_prefetched_lexical_fallback_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir) / "repo"
            repo_root.mkdir()
            local_cache_dir = Path(temp_dir) / "local-cache"
            output_dir = Path(temp_dir) / "out"
            output_dir.mkdir()
            (repo_root / "worker.py").write_text("def retry_job(job_id):\n    return job_id\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": repo_root}]
            query = "trace retry_job"
            budgets = inspection_pipeline.normalize_token_budgets({})

            repository_fp, _ = inspection_index.repository_fingerprint(
                discovered,
                excluded_paths=[local_cache_dir, output_dir],
                cache_dir=local_cache_dir,
            )
            _discovered, _namespaces, _effective_excluded_paths, build_config_digest = inspection_hotpath._snapshot_build_context(
                discovered,
                set(),
                cache_dir=local_cache_dir,
                excluded_paths=inspection_hotpath.exclusion_paths_for_execution(
                    {"repo_inspection_cache_path": str(local_cache_dir)},
                    output_dir,
                ),
                transient_excluded_paths=inspection_hotpath.transient_excluded_paths_for_execution(output_dir),
            )
            cache_key = inspection_hotpath.query_stage_cache_key(
                query,
                "sha256:index-fp",
                {"tier": "cpu-lexical-fallback", "mode": "lexical_fallback"},
                budgets,
            )
            inspection_pipeline._write_query_stage_cache(
                local_cache_dir,
                query,
                cache_key,
                {"tier": "cpu-lexical-fallback", "mode": "lexical_fallback"},
                [{"chunk_id": "chunk-1", "rrf_score": 1.0, "rank": 1}],
                [{"chunk_id": "chunk-1", "rank": 1}],
                [{
                    "id": "ev_001",
                    "chunk_id": "chunk-1",
                    "path": "worker.py",
                    "content": "def retry_job(job_id):\n    return job_id\n",
                    "excerpt": "def retry_job(job_id):\n    return job_id\n",
                    "language": "python",
                    "rank": 1,
                    "symbol": "retry_job",
                    "source_refs": [{
                        "path": "worker.py",
                        "line_start": 1,
                        "line_end": 2,
                        "content_hash": "sha256:test",
                        "input_id": "repo",
                        "source_namespace": "repo",
                    }],
                }],
                False,
                retrieval_quality="cpu",
                rerank_quality="cpu",
                repository_state_fingerprint=repository_fp,
                build_config_digest=build_config_digest,
                index_fingerprint="sha256:index-fp",
                total_files=1,
                chunk_count=1,
                budgets=budgets,
            )

            inspection_hotpath._PREFETCH_STATE_CACHE.clear()
            first = inspection_hotpath.prepare_prefetched_state(
                discovered,
                query,
                mode="evidence",
                constraints={},
                task_params={
                    "_broker_repository_state_fingerprint": repository_fp,
                    "_broker_repository_state_fingerprint_source": "request_cache",
                },
                execution_plan={"repo_inspection_cache_path": str(local_cache_dir)},
                output_dir=output_dir,
            )
            self.assertIsNotNone(first.get("cached_lexical_fallback_run"))

            with mock.patch.object(
                inspection_hotpath,
                "cached_lexical_fallback_from_context",
                side_effect=AssertionError("second identical prefetched hit should reuse cached lexical fallback run from _PREFETCH_STATE_CACHE"),
            ):
                second = inspection_hotpath.prepare_prefetched_state(
                    discovered,
                    query,
                    mode="evidence",
                    constraints={},
                    task_params={
                        "_broker_repository_state_fingerprint": repository_fp,
                        "_broker_repository_state_fingerprint_source": "request_cache",
                    },
                    execution_plan={"repo_inspection_cache_path": str(local_cache_dir)},
                    output_dir=output_dir,
                )

        self.assertIsNotNone(second.get("cached_lexical_fallback_run"))

    def test_prepare_prefetched_state_uses_alias_released_lexical_fallback_without_loading_query_stage_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            shared_cache_dir = temp_root / "shared-cache"
            local_cache_dir = temp_root / "local-cache"
            output_dir = temp_root / "out"
            output_dir.mkdir()
            (root / "worker.py").write_text("def retry_job(job_id):\n    return job_id\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            query = "trace retry_job"
            budgets = inspection_pipeline.normalize_token_budgets({})

            with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
                repository_fp, _ = inspection_index.repository_fingerprint(
                    discovered,
                    excluded_paths=[local_cache_dir, output_dir],
                    cache_dir=local_cache_dir,
                )
                _discovered, _namespaces, _effective_excluded_paths, build_config_digest = inspection_hotpath._snapshot_build_context(
                    discovered,
                    set(),
                    cache_dir=local_cache_dir,
                    excluded_paths=inspection_hotpath.exclusion_paths_for_execution(
                        {"repo_inspection_cache_path": str(local_cache_dir)},
                        output_dir,
                    ),
                    transient_excluded_paths=inspection_hotpath.transient_excluded_paths_for_execution(output_dir),
                )
                cache_key = inspection_hotpath.query_stage_cache_key(
                    query,
                    "sha256:index-fp",
                    {"tier": "cpu-lexical-fallback", "mode": "lexical_fallback"},
                    budgets,
                )
                released_payload = {
                    "mode": "evidence",
                    "query": query,
                    "findings": [],
                    "evidence": [{
                        "id": "ev_001",
                        "chunk_id": "chunk-1",
                        "path": "worker.py",
                        "excerpt": "def retry_job(job_id): return job_id",
                        "source_refs": [{"path": "worker.py", "line_start": 1, "line_end": 2}],
                    }],
                    "quality": {"result": "evidence_only", "retrieval": "cpu", "reranking": "cpu", "synthesis": "not_requested", "answer_ready": False},
                    "warnings": ["GPU_RETRIEVAL_UNAVAILABLE_LEXICAL_FALLBACK"],
                    "provenance": {"repository_fingerprint": repository_fp, "index_fingerprint": "sha256:index-fp"},
                }
                released_artifact_payloads = {
                    "evidence_pack": {"query": query, "evidence": [{"id": "ev_001"}]},
                    "runtime_diagnostics": {"attempts": []},
                }
                inspection_pipeline._write_query_stage_cache(
                    local_cache_dir,
                    query,
                    cache_key,
                    {"tier": "cpu-lexical-fallback", "mode": "lexical_fallback"},
                    [{"chunk_id": "chunk-1", "rrf_score": 1.0, "rank": 1}],
                    [{"chunk_id": "chunk-1", "rank": 1}],
                    [{
                        "id": "ev_001",
                        "chunk_id": "chunk-1",
                        "path": "worker.py",
                        "content": "def retry_job(job_id):\n    return job_id\n",
                        "excerpt": "def retry_job(job_id):\n    return job_id\n",
                        "language": "python",
                        "rank": 1,
                        "symbol": "retry_job",
                        "source_refs": [{
                            "path": "worker.py",
                            "line_start": 1,
                            "line_end": 2,
                            "content_hash": "sha256:test",
                            "input_id": "repo",
                            "source_namespace": "repo",
                        }],
                    }],
                    False,
                    retrieval_quality="cpu",
                    rerank_quality="cpu",
                    released_payload=released_payload,
                    released_artifact_payloads=released_artifact_payloads,
                    repository_state_fingerprint=repository_fp,
                    build_config_digest=build_config_digest,
                    index_fingerprint="sha256:index-fp",
                    total_files=1,
                    chunk_count=1,
                    budgets=budgets,
                )

                with mock.patch.object(
                    inspection_hotpath,
                    "load_query_stage_cache",
                    side_effect=AssertionError("released lexical-fallback alias hit should avoid full query-stage cache load"),
                ), mock.patch.object(
                    inspection_hotpath,
                    "load_cached_chunk_snapshot_metadata",
                    side_effect=AssertionError("released lexical-fallback alias hit should avoid snapshot metadata load"),
                ):
                    state = inspection_hotpath.prepare_prefetched_state(
                        discovered,
                        query,
                        mode="evidence",
                        constraints={},
                        task_params={
                            "_broker_repository_state_fingerprint": repository_fp,
                            "_broker_repository_state_fingerprint_source": "request_cache",
                        },
                        execution_plan={"repo_inspection_cache_path": str(local_cache_dir)},
                        output_dir=output_dir,
                    )

        self.assertIsNotNone(state["cached_query_stage"])
        self.assertIsNotNone(state.get("cached_lexical_fallback_run"))
        self.assertEqual(state["fingerprint"], "sha256:index-fp")
        self.assertEqual(
            state["cached_lexical_fallback_run"]["payload"]["warnings"],
            ["GPU_RETRIEVAL_UNAVAILABLE_LEXICAL_FALLBACK"],
        )

    def test_cached_lexical_fallback_from_context_reuses_persisted_released_payload_without_retrimming(self):
        context = {
            "query": "trace worker",
            "mode": "evidence",
            "budgets": {
                "retrieval_token_budget": 16000,
                "evidence_token_budget": 4000,
                "final_pack_token_budget": 2048,
                "synthesis_context_token_budget": 16000,
            },
            "task_params": {},
            "cache_dir": Path("/tmp/cache"),
            "execution_plan": {},
            "output_dir": Path("/tmp/out"),
            "repository_state_fingerprint": "sha256:repo",
            "fingerprint_state": [{"kind": "input_manifest"}],
            "cached_chunk_snapshot_metadata": {"total_files": 1},
            "metadata_chunks": type("MetadataChunkList", (list,), {"_chunk_count": 1})(),
            "fingerprint": "sha256:index",
            "cached_query_stage": {
                "ranked": [{"chunk_id": "chunk-1", "rank": 1}],
                "selected": [{"chunk_id": "chunk-1", "rank": 1}],
                "evidence": [{
                    "id": "ev_001",
                    "chunk_id": "chunk-1",
                    "path": "worker.py",
                    "excerpt": "def worker(): pass",
                    "source_refs": [{"path": "worker.py", "line_start": 1, "line_end": 1}],
                }],
                "retrieval_quality": "cpu",
                "rerank_quality": "cpu",
                "released_payload": {
                    "mode": "evidence",
                    "query": "trace worker",
                    "findings": [],
                    "evidence": [{
                        "id": "ev_001",
                        "chunk_id": "chunk-1",
                        "path": "worker.py",
                        "excerpt": "def worker(): pass",
                        "source_refs": [{"path": "worker.py", "line_start": 1, "line_end": 1}],
                    }],
                    "quality": {"result": "evidence_only", "retrieval": "cpu", "reranking": "cpu", "synthesis": "not_requested", "answer_ready": False},
                    "warnings": ["GPU_RETRIEVAL_UNAVAILABLE_LEXICAL_FALLBACK"],
                    "provenance": {"repository_fingerprint": "sha256:repo", "index_fingerprint": "sha256:index"},
                },
            },
        }

        with mock.patch.object(
            inspection_hotpath,
            "trim_evidence_for_final_pack",
            side_effect=AssertionError("persisted released payload should avoid final-pack retrimming"),
        ):
            cached_run = inspection_hotpath.cached_lexical_fallback_from_context(context)

        self.assertIsNotNone(cached_run)
        self.assertEqual(cached_run["payload"]["warnings"], ["GPU_RETRIEVAL_UNAVAILABLE_LEXICAL_FALLBACK"])
        self.assertEqual(cached_run["payload"]["evidence"][0]["id"], "ev_001")

    def test_cached_lexical_fallback_from_context_reuses_persisted_static_artifact_payloads(self):
        context = {
            "query": "trace worker",
            "mode": "evidence",
            "budgets": {
                "retrieval_token_budget": 16000,
                "evidence_token_budget": 4000,
                "final_pack_token_budget": 2048,
                "synthesis_context_token_budget": 16000,
            },
            "task_params": {"include_full_trace": True},
            "cache_dir": Path("/tmp/cache"),
            "execution_plan": {},
            "output_dir": Path("/tmp/out"),
            "repository_state_fingerprint": "sha256:repo",
            "fingerprint_state": [{"kind": "input_manifest"}],
            "cached_chunk_snapshot_metadata": {"total_files": 1},
            "metadata_chunks": type("MetadataChunkList", (list,), {"_chunk_count": 1})(),
            "fingerprint": "sha256:index",
            "cached_query_stage": {
                "ranked": [{"chunk_id": "chunk-1", "rank": 1}],
                "selected": [{"chunk_id": "chunk-1", "rank": 1}],
                "evidence": [{
                    "id": "ev_001",
                    "chunk_id": "chunk-1",
                    "path": "worker.py",
                    "excerpt": "def worker(): pass",
                    "source_refs": [{"path": "worker.py", "line_start": 1, "line_end": 1}],
                }],
                "retrieval_quality": "cpu",
                "rerank_quality": "cpu",
                "released_payload": {
                    "mode": "evidence",
                    "query": "trace worker",
                    "findings": [],
                    "evidence": [{
                        "id": "ev_001",
                        "chunk_id": "chunk-1",
                        "path": "worker.py",
                        "excerpt": "def worker(): pass",
                        "source_refs": [{"path": "worker.py", "line_start": 1, "line_end": 1}],
                    }],
                    "quality": {"result": "evidence_only", "retrieval": "cpu", "reranking": "cpu", "synthesis": "not_requested", "answer_ready": False},
                    "warnings": ["GPU_RETRIEVAL_UNAVAILABLE_LEXICAL_FALLBACK"],
                    "provenance": {"repository_fingerprint": "sha256:repo", "index_fingerprint": "sha256:index"},
                },
                "released_artifact_payloads": {
                    "evidence_pack": {"query": "persisted query", "evidence": [{"id": "persisted"}]},
                    "chunk_manifest": {"fingerprint": "sha256:persisted", "chunks": [{"id": "persisted"}], "selected_chunk_ids": ["persisted"]},
                },
            },
        }

        cached_run = inspection_hotpath.cached_lexical_fallback_from_context(context)

        self.assertIsNotNone(cached_run)
        artifact_payloads = cached_run["artifact_payloads"]
        self.assertEqual(artifact_payloads["evidence_pack"]["query"], "persisted query")
        self.assertEqual(artifact_payloads["evidence_pack"]["evidence"], [{"id": "persisted"}])
        self.assertEqual(artifact_payloads["chunk_manifest"]["fingerprint"], "sha256:persisted")
        self.assertTrue(artifact_payloads["retrieval_result"]["query_stage_cache_hit"])
        self.assertEqual(artifact_payloads["runtime_diagnostics"], cached_run["payload"]["runtime"])

    def test_prepare_prefetched_state_uses_gpu_query_stage_alias_without_loading_snapshot_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            shared_cache_dir = temp_root / "shared-cache"
            local_cache_dir = temp_root / "local-cache"
            output_dir = temp_root / "out"
            output_dir.mkdir()
            registry_path = temp_root / "gpu-registry.json"
            (root / "worker.py").write_text("def retry_job(job_id):\n    return job_id\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            query = "trace retry_job"
            budgets = inspection_pipeline.normalize_token_budgets({})
            retrieval_signature = {
                "tier": "p40-retrieval",
                "model_profile": "profile-p40-retrieval",
                "model": "/models/p40-retrieval",
                "mode": "gpu",
            }
            registry_path.write_text(
                json.dumps({"schema": "gpu_service_registry_v1", "records": [service("p40-retrieval", ["embed", "faiss_search", "rerank"], 1)]}),
                encoding="utf-8",
            )

            with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
                repository_fp, _ = inspection_index.repository_fingerprint(
                    discovered,
                    excluded_paths=[local_cache_dir, output_dir],
                    cache_dir=local_cache_dir,
                )
                _discovered, _namespaces, _effective_excluded_paths, build_config_digest = inspection_hotpath._snapshot_build_context(
                    discovered,
                    set(),
                    cache_dir=local_cache_dir,
                    excluded_paths=inspection_hotpath.exclusion_paths_for_execution(
                        {"repo_inspection_cache_path": str(local_cache_dir)},
                        output_dir,
                    ),
                    transient_excluded_paths=inspection_hotpath.transient_excluded_paths_for_execution(output_dir),
                )
                cache_key = inspection_hotpath.query_stage_cache_key(
                    query,
                    "sha256:index-fp",
                    retrieval_signature,
                    budgets,
                )
                inspection_pipeline._write_query_stage_cache(
                    local_cache_dir,
                    query,
                    cache_key,
                    retrieval_signature,
                    [{"chunk_id": "chunk-1", "rrf_score": 1.0, "rank": 1}],
                    [{"chunk_id": "chunk-1", "rank": 1}],
                    [{"id": "ev_001", "chunk_id": "chunk-1", "path": "worker.py", "content": "def retry_job(job_id):\n    return job_id\n"}],
                    False,
                    retrieval_quality="gpu",
                    rerank_quality="gpu",
                    answer="retry_job is defined in worker.py",
                    findings=[{"summary": "retry_job is defined in worker.py", "evidence_refs": ["ev_001"]}],
                    warnings=[],
                    provenance={"index_fingerprint": "sha256:index-fp"},
                    runtime_attempts=[{"operation": "semantic_retrieval", "status": "succeeded"}],
                    synthesis_quality="gpu",
                    repository_state_fingerprint=repository_fp,
                    build_config_digest=build_config_digest,
                    index_fingerprint="sha256:index-fp",
                    total_files=1,
                    chunk_count=1,
                    budgets=budgets,
                )

                with mock.patch.object(
                    inspection_hotpath,
                    "load_cached_chunk_snapshot_metadata",
                    side_effect=AssertionError("gpu query-stage alias hit should avoid snapshot metadata load"),
                ):
                    state = inspection_hotpath.prepare_prefetched_state(
                        discovered,
                        query,
                        mode="answer",
                        constraints={},
                        task_params={
                            "_broker_repository_state_fingerprint": repository_fp,
                            "_broker_repository_state_fingerprint_source": "request_cache",
                        },
                        execution_plan={
                            "repo_inspection_cache_path": str(local_cache_dir),
                            "gpu_service_registry_path": str(registry_path),
                        },
                        output_dir=output_dir,
                    )

        self.assertIsNotNone(state["cached_query_stage"])
        self.assertEqual(state["prefetched_retrieval_signature"]["mode"], "gpu")
        self.assertEqual(state["prefetched_retrieval_signature"]["tier"], "p40-retrieval")
        self.assertEqual(state["fingerprint"], "sha256:index-fp")
        self.assertEqual(int(state["cached_chunk_snapshot_metadata"]["total_files"]), 1)
        self.assertEqual(int(getattr(state["metadata_chunks"], "_chunk_count", 0)), 1)

    def test_prepare_prefetched_state_cache_key_tracks_retrieval_signature(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            shared_cache_dir = temp_root / "shared-cache"
            local_cache_dir = temp_root / "local-cache"
            output_dir = temp_root / "out"
            output_dir.mkdir()
            registry_path = temp_root / "gpu-registry.json"
            (root / "worker.py").write_text("def retry_job(job_id):\n    return job_id\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            query = "trace retry_job"
            budgets = inspection_pipeline.normalize_token_budgets({})
            lexical_signature = {"tier": "cpu-lexical-fallback", "mode": "lexical_fallback"}
            gpu_signature = {
                "tier": "p40-retrieval",
                "model_profile": "profile-p40-retrieval",
                "model": "/models/p40-retrieval",
                "mode": "gpu",
            }
            registry_path.write_text(
                json.dumps({"schema": "gpu_service_registry_v1", "records": [service("p40-retrieval", ["embed", "faiss_search", "rerank"], 1)]}),
                encoding="utf-8",
            )

            with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
                inspection_hotpath._PREFETCH_STATE_CACHE.clear()
                repository_fp, _ = inspection_index.repository_fingerprint(
                    discovered,
                    excluded_paths=[local_cache_dir, output_dir],
                    cache_dir=local_cache_dir,
                )
                _discovered, _namespaces, _effective_excluded_paths, build_config_digest = inspection_hotpath._snapshot_build_context(
                    discovered,
                    set(),
                    cache_dir=local_cache_dir,
                    excluded_paths=inspection_hotpath.exclusion_paths_for_execution(
                        {"repo_inspection_cache_path": str(local_cache_dir)},
                        output_dir,
                    ),
                    transient_excluded_paths=inspection_hotpath.transient_excluded_paths_for_execution(output_dir),
                )
                for signature, retrieval_quality, answer in (
                    (lexical_signature, "cpu", ""),
                    (gpu_signature, "gpu", "retry_job is defined in worker.py"),
                ):
                    cache_key = inspection_hotpath.query_stage_cache_key(
                        query,
                        "sha256:index-fp",
                        signature,
                        budgets,
                    )
                    inspection_pipeline._write_query_stage_cache(
                        local_cache_dir,
                        query,
                        cache_key,
                        signature,
                        [{"chunk_id": "chunk-1", "rrf_score": 1.0, "rank": 1}],
                        [{"chunk_id": "chunk-1", "rank": 1}],
                        [{"id": "ev_001", "chunk_id": "chunk-1", "path": "worker.py", "content": "def retry_job(job_id):\n    return job_id\n"}],
                        False,
                        retrieval_quality=retrieval_quality,
                        rerank_quality=retrieval_quality,
                        answer=answer,
                        findings=([{"summary": answer, "evidence_refs": ["ev_001"]}] if answer else []),
                        warnings=[],
                        provenance={"index_fingerprint": "sha256:index-fp"},
                        runtime_attempts=([{"operation": "semantic_retrieval", "status": "succeeded"}] if answer else []),
                        synthesis_quality=("gpu" if answer else "not_requested"),
                        repository_state_fingerprint=repository_fp,
                        build_config_digest=build_config_digest,
                        index_fingerprint="sha256:index-fp",
                        total_files=1,
                        chunk_count=1,
                        budgets=budgets,
                    )

                first = inspection_hotpath.prepare_prefetched_state(
                    discovered,
                    query,
                    mode="evidence",
                    constraints={},
                    task_params={
                        "_broker_repository_state_fingerprint": repository_fp,
                        "_broker_repository_state_fingerprint_source": "request_cache",
                    },
                    execution_plan={"repo_inspection_cache_path": str(local_cache_dir)},
                    output_dir=output_dir,
                )
                second = inspection_hotpath.prepare_prefetched_state(
                    discovered,
                    query,
                    mode="answer",
                    constraints={},
                    task_params={
                        "_broker_repository_state_fingerprint": repository_fp,
                        "_broker_repository_state_fingerprint_source": "request_cache",
                    },
                    execution_plan={
                        "repo_inspection_cache_path": str(local_cache_dir),
                        "gpu_service_registry_path": str(registry_path),
                    },
                    output_dir=output_dir,
                )

        self.assertEqual(first["prefetched_retrieval_signature"]["mode"], "lexical_fallback")
        self.assertEqual(second["prefetched_retrieval_signature"]["mode"], "gpu")
        self.assertEqual(first["cached_query_stage"]["retrieval_quality"], "cpu")
        self.assertEqual(second["cached_query_stage"]["retrieval_quality"], "gpu")

    def test_token_budget_aliases_are_accepted_but_names_are_explicit(self):
        budgets = inspection_pipeline.normalize_token_budgets(
            {
                "retrieved_chunk_budget": 100,
                "final_evidence_pack_budget": 200,
                "remote_model_context_budget": 300,
            }
        )
        self.assertEqual(budgets["retrieval_token_budget"], 100)
        self.assertEqual(budgets["evidence_token_budget"], 200)
        self.assertEqual(budgets["final_pack_token_budget"], 200)
        self.assertEqual(budgets["synthesis_context_token_budget"], 300)


class RetrievalDiagnosticsTests(RepoFixture):
    def test_cache_miss_reuses_selected_gpu_endpoints_without_duplicate_resolution(self):
        factory = SemanticDiagnosticsFactory()
        original_select_endpoint = gpu_client.select_endpoint
        calls = []

        def counting_select_endpoint(*args, **kwargs):
            calls.append((args[1], args[2], kwargs.get("expected_gpu_count")))
            return original_select_endpoint(*args, **kwargs)

        with mock.patch.object(gpu_client, "select_endpoint", side_effect=counting_select_endpoint):
            previous_symbols = inspection_pipeline._GPU_SYMBOLS
            inspection_pipeline._GPU_SYMBOLS = None
            try:
                payload = self.run_pipeline(mode="evidence", services=all_services(), factory=factory)
            finally:
                inspection_pipeline._GPU_SYMBOLS = previous_symbols

        self.assertFalse(payload["retrieval"]["query_stage_cache_hit"])
        self.assertEqual(calls, [
            ("p40-retrieval", "search", 1),
            ("p40-retrieval", "rerank", 1),
        ])
        self.assertTrue(factory.semantic_search_calls)
        self.assertTrue(factory.rerank_calls)

    def test_payload_reports_semantic_index_reuse_counters(self):
        payload = self.run_pipeline(
            mode="evidence",
            services=all_services(),
            factory=SemanticDiagnosticsFactory(cache_hit=False, embedded_documents=1, reused_documents=1),
        )

        self.assertEqual(payload["retrieval"]["semantic_index_cache_hit"], False)
        self.assertEqual(payload["retrieval"]["semantic_index_document_count"], 4)
        self.assertEqual(payload["retrieval"]["semantic_index_embedded_documents"], 1)
        self.assertEqual(payload["retrieval"]["semantic_index_reused_documents"], 1)
        self.assertEqual(payload["retrieval"]["chunk_cache_total_files"], 2)
        self.assertEqual(payload["retrieval"]["chunk_cache_reused_files"], 0)
        self.assertEqual(payload["retrieval"]["chunk_cache_rebuilt_files"], 2)
        self.assertEqual(payload["retrieval"]["lexical_index_working_cache_hit"], False)
        self.assertEqual(payload["retrieval"]["lexical_index_updated_files"], 2)
        self.assertEqual(payload["retrieval"]["lexical_index_removed_files"], 0)
        self.assertEqual(payload["retrieval"]["lexical_index_inserted_chunks"], 4)

    def test_run_inspection_uses_broker_fingerprint_hint_without_prefetch(self):
        task_params = {
            "index_cache_dir": str(self.root / ".broker" / "inspection-test"),
            "_broker_repository_state_fingerprint": "git:test-fingerprint",
            "_broker_repository_state_fingerprint_source": "request_cache",
        }
        with mock.patch.object(
            inspection_pipeline,
            "repository_fingerprint",
            side_effect=AssertionError("direct broker fingerprint hint should skip local repository fingerprinting"),
        ):
            payload = self.run_pipeline(
                mode="evidence",
                services=all_services(),
                factory=SemanticDiagnosticsFactory(),
                task_params=task_params,
            )

        self.assertEqual(payload["retrieval"]["fingerprint_sources"], ["broker_hint"])
        self.assertEqual(payload["retrieval"]["setup_timings_ms"]["repository_fingerprint_ms"], 0.0)

    def test_broker_fingerprint_hint_allows_cached_chunk_reload_without_rebuild(self):
        cache_dir = self.root / ".broker" / "inspection-test"
        first_factory = SemanticDiagnosticsFactory()
        first = inspection_pipeline.run_inspection(
            self.discovered,
            "Trace the retry_job service call chain",
            mode="evidence",
            task_params={"index_cache_dir": str(cache_dir)},
            services=all_services(),
            client_factory=first_factory,
            output_dir=self.root / "out",
        )["payload"]
        repository_fingerprint = str((first.get("provenance") or {}).get("repository_fingerprint") or "")
        self.assertTrue(repository_fingerprint)

        second_factory = SemanticDiagnosticsFactory()
        with mock.patch.object(
            inspection_pipeline,
            "repository_fingerprint",
            side_effect=AssertionError("broker fingerprint hint should skip local repository fingerprinting"),
        ), mock.patch.object(
            inspection_pipeline,
            "build_syntax_chunks",
            side_effect=AssertionError("broker fingerprint hint should allow cached chunk reload without rebuilding"),
        ):
            second = inspection_pipeline.run_inspection(
                self.discovered,
                "Trace the submit_job service call chain",
                mode="evidence",
                task_params={
                    "index_cache_dir": str(cache_dir),
                    "_broker_repository_state_fingerprint": repository_fingerprint,
                    "_broker_repository_state_fingerprint_source": "request_cache",
                },
                services=all_services(),
                client_factory=second_factory,
                output_dir=self.root / "out",
            )["payload"]

        self.assertEqual(second["retrieval"]["fingerprint_sources"], ["broker_hint"])
        self.assertEqual(second["retrieval"]["setup_timings_ms"]["repository_fingerprint_ms"], 0.0)
        self.assertEqual(second["retrieval"]["chunk_cache_total_files"], 2)
        self.assertEqual(second["retrieval"]["chunk_cache_reused_files"], 2)
        self.assertEqual(second["retrieval"]["chunk_cache_rebuilt_files"], 0)

    def test_run_inspection_evidence_mode_writes_query_stage_cache_once(self):
        calls = []
        original_write = inspection_pipeline._write_query_stage_cache

        def recording_write(*args, **kwargs):
            calls.append((args, kwargs))
            return original_write(*args, **kwargs)

        with mock.patch.object(
            inspection_pipeline,
            "_write_query_stage_cache",
            side_effect=recording_write,
        ):
            payload = self.run_pipeline(
                mode="evidence",
                services=all_services(),
                factory=SemanticDiagnosticsFactory(),
            )

        self.assertEqual(payload["quality"]["result"], "evidence_only")
        self.assertEqual(len(calls), 1)

    def test_index_cache_dir_reuses_cached_chunk_reload_across_output_dirs(self):
        cache_dir = self.root / ".broker" / "inspection-test"
        inspection_pipeline.run_inspection(
            self.discovered,
            "Trace the retry_job service call chain",
            mode="evidence",
            task_params={"index_cache_dir": str(cache_dir)},
            services=all_services(),
            client_factory=SemanticDiagnosticsFactory(),
            output_dir=self.root / "out-one",
        )

        with mock.patch.object(
            inspection_pipeline,
            "build_syntax_chunks",
            side_effect=AssertionError("stable index_cache_dir should allow cached chunk reload across output dirs"),
        ):
            second = inspection_pipeline.run_inspection(
                self.discovered,
                "Trace the submit_job service call chain",
                mode="evidence",
                task_params={"index_cache_dir": str(cache_dir)},
                services=all_services(),
                client_factory=SemanticDiagnosticsFactory(),
                output_dir=self.root / "out-two",
            )["payload"]

        self.assertEqual(second["retrieval"]["chunk_cache_total_files"], 2)
        self.assertEqual(second["retrieval"]["chunk_cache_reused_files"], 2)
        self.assertEqual(second["retrieval"]["chunk_cache_rebuilt_files"], 0)

    def test_run_inspection_uses_input_manifest_content_hash_without_prefetch(self):
        discovered = [
            {
                "id": "input_0",
                "type": "repo",
                "classification": "internal",
                "path": self.root,
                "content_hash": "git:manifest-fingerprint",
            }
        ]
        with mock.patch.object(
            inspection_pipeline,
            "repository_fingerprint",
            side_effect=AssertionError("direct manifest content_hash should skip local repository fingerprinting"),
        ):
            payload = self.run_pipeline(
                mode="evidence",
                services=all_services(),
                factory=SemanticDiagnosticsFactory(),
                discovered=discovered,
            )

        self.assertEqual(payload["retrieval"]["fingerprint_sources"], ["input_manifest"])
        self.assertEqual(payload["retrieval"]["setup_timings_ms"]["repository_fingerprint_ms"], 0.0)

    def test_second_pipeline_run_reuses_cached_file_chunks(self):
        first = self.run_pipeline(mode="evidence", services=all_services(), factory=SemanticDiagnosticsFactory())
        second = self.run_pipeline(mode="evidence", services=all_services(), factory=SemanticDiagnosticsFactory())

        self.assertEqual(first["retrieval"]["chunk_cache_reused_files"], 0)
        self.assertEqual(first["retrieval"]["chunk_cache_rebuilt_files"], 2)
        self.assertEqual(second["retrieval"]["chunk_cache_total_files"], 2)
        self.assertEqual(second["retrieval"]["chunk_cache_reused_files"], 2)
        self.assertEqual(second["retrieval"]["chunk_cache_rebuilt_files"], 0)
        if second["retrieval"]["query_stage_cache_hit"]:
            self.assertEqual(second["retrieval"]["lexical_index_cache_hit"], False)
            self.assertEqual(second["retrieval"]["lexical_index_working_cache_hit"], False)
            self.assertEqual(second["retrieval"]["lexical_index_updated_files"], 0)
            self.assertEqual(second["retrieval"]["lexical_index_removed_files"], 0)
            self.assertEqual(second["retrieval"]["lexical_index_inserted_chunks"], 0)
        else:
            self.assertEqual(second["retrieval"]["lexical_index_cache_hit"], True)
            self.assertEqual(second["retrieval"]["lexical_index_working_cache_hit"], True)
            self.assertEqual(second["retrieval"]["lexical_index_updated_files"], 0)
            self.assertEqual(second["retrieval"]["lexical_index_removed_files"], 0)
            self.assertEqual(second["retrieval"]["lexical_index_inserted_chunks"], 0)

    def test_output_dir_under_repo_is_excluded_from_second_run_discovery(self):
        first = inspection_pipeline.run_inspection(
            self.discovered,
            "Trace the retry_job service call chain",
            mode="evidence",
            task_params={"index_cache_dir": str(self.root / ".broker" / "inspection-test")},
            services=all_services(),
            client_factory=SemanticDiagnosticsFactory(),
            output_dir=self.root / "out",
        )["payload"]
        second = inspection_pipeline.run_inspection(
            self.discovered,
            "Trace the submit_job service call chain",
            mode="evidence",
            task_params={"index_cache_dir": str(self.root / ".broker" / "inspection-test")},
            services=all_services(),
            client_factory=SemanticDiagnosticsFactory(),
            output_dir=self.root / "out",
        )["payload"]

        self.assertEqual(first["retrieval"]["chunk_cache_total_files"], 2)
        self.assertEqual(second["retrieval"]["chunk_cache_total_files"], 2)
        self.assertEqual(second["retrieval"]["chunk_cache_reused_files"], 2)
        self.assertEqual(second["retrieval"]["chunk_cache_rebuilt_files"], 0)
        self.assertFalse(second["retrieval"]["query_stage_cache_hit"])

    def test_new_query_reuses_lexical_index_without_full_snapshot_reload(self):
        cache_dir = self.root / ".broker" / "inspection-test"
        first_factory = SemanticDiagnosticsFactory()
        inspection_pipeline.run_inspection(
            self.discovered,
            "Trace the retry_job service call chain",
            mode="evidence",
            task_params={"index_cache_dir": str(cache_dir)},
            services=all_services(),
            client_factory=first_factory,
            output_dir=self.root / "out",
        )

        second_factory = SemanticDiagnosticsFactory()
        with mock.patch.object(
            inspection_pipeline,
            "load_cached_chunk_snapshot",
            side_effect=AssertionError("warm unchanged repo should reuse lexical index metadata before full snapshot reload"),
        ):
            second = inspection_pipeline.run_inspection(
                self.discovered,
                "Trace the submit_job service call chain",
                mode="evidence",
                task_params={"index_cache_dir": str(cache_dir)},
                services=all_services(),
                client_factory=second_factory,
                output_dir=self.root / "out",
            )["payload"]

        self.assertFalse(second["retrieval"]["query_stage_cache_hit"])
        self.assertEqual(second["retrieval"]["chunk_cache_total_files"], 2)
        self.assertEqual(second["retrieval"]["chunk_cache_reused_files"], 2)
        self.assertEqual(second["retrieval"]["chunk_cache_rebuilt_files"], 0)
        self.assertEqual(second["retrieval"]["chunk_snapshot_local_load_ms"], 0.0)
        self.assertTrue(second["retrieval"]["lexical_index_cache_hit"])
        self.assertTrue(second_factory.ensure_semantic_index_calls)
        self.assertTrue(second_factory.semantic_search_calls)

    def test_second_identical_query_reuses_query_stage_cache(self):
        first_factory = SemanticDiagnosticsFactory()
        first = self.run_pipeline(mode="evidence", services=all_services(), factory=first_factory)
        second_factory = SemanticDiagnosticsFactory()
        with (
            mock.patch.object(
                inspection_pipeline,
                "build_syntax_chunks",
                side_effect=AssertionError("query-stage cache hit should skip full chunk snapshot loading"),
            ),
            mock.patch.object(
                inspection_pipeline,
                "lexical_search",
                side_effect=AssertionError("query-stage cache hit should skip lexical search"),
            ),
            mock.patch.object(
                inspection_pipeline,
                "ensure_lexical_index",
                side_effect=AssertionError("query-stage cache hit should skip lexical index setup"),
            ),
        ):
            second = self.run_pipeline(mode="evidence", services=all_services(), factory=second_factory)

        self.assertFalse(first["retrieval"]["query_stage_cache_hit"])
        self.assertTrue(second["retrieval"]["query_stage_cache_hit"])
        self.assertTrue(first_factory.ensure_semantic_index_calls)
        self.assertTrue(first_factory.semantic_search_calls)
        self.assertTrue(first_factory.rerank_calls)
        self.assertEqual(second_factory.ensure_semantic_index_calls, [])
        self.assertEqual(second_factory.semantic_search_calls, [])
        self.assertEqual(second_factory.rerank_calls, [])
        self.assertEqual(second["retrieval"]["lexical_candidates"], 0)

    def test_query_stage_cache_reuses_same_content_after_unrelated_sibling_git_change(self):
        target = self.root / "target"
        sibling = self.root / "other"
        target.mkdir()
        sibling.mkdir()
        (target / "service.py").write_text(
            "def retry_job(job_id):\n    return submit_job(job_id)\n\ndef submit_job(job_id):\n    return job_id\n",
            encoding="utf-8",
        )
        (target / "mcp.go").write_text(
            "package mcp\n\nfunc InspectRepo(query string) string {\n\treturn query\n}\n",
            encoding="utf-8",
        )
        (sibling / "noise.py").write_text("value = 1\n", encoding="utf-8")
        for path in (self.root / "service.py", self.root / "mcp.go"):
            path.unlink()
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        subprocess.run(["git", "-C", str(self.root), "config", "user.email", "test@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(self.root), "config", "user.name", "Test"], check=True)
        subprocess.run(["git", "-C", str(self.root), "add", "."], check=True)
        subprocess.run(["git", "-C", str(self.root), "commit", "-qm", "initial"], check=True)
        discovered = [{"id": "input_0", "type": "repo", "classification": "internal", "path": target}]

        first_factory = SemanticDiagnosticsFactory()
        first = self.run_pipeline(
            mode="evidence",
            services=all_services(),
            factory=first_factory,
            discovered=discovered,
        )

        (sibling / "noise.py").write_text("value = 2\n", encoding="utf-8")
        second_factory = SemanticDiagnosticsFactory()
        second = self.run_pipeline(
            mode="evidence",
            services=all_services(),
            factory=second_factory,
            discovered=discovered,
        )

        self.assertFalse(first["retrieval"]["query_stage_cache_hit"])
        self.assertTrue(second["retrieval"]["query_stage_cache_hit"])
        self.assertEqual(first["retrieval"]["fingerprint"], second["retrieval"]["fingerprint"])
        self.assertEqual(second_factory.ensure_semantic_index_calls, [])
        self.assertEqual(second_factory.semantic_search_calls, [])
        self.assertEqual(second_factory.rerank_calls, [])

    def test_stale_metadata_fingerprint_is_recomputed_after_chunk_rebuild(self):
        cache_dir = self.root / ".broker" / "inspection-test"
        chunks = inspection_index.build_syntax_chunks(self.discovered, cache_dir=cache_dir)
        stale_metadata = {
            "index_manifest": {"repo\x00service.py": "sha256:stale"},
            "semantic_document_signatures": {"chunk-stale": "sha256:stale"},
            "chunk_ids": ("chunk-stale",),
            "chunk_count": 1,
            "total_files": 2,
            "build_config_digest": "sha256:build",
        }
        cache_keys = []

        def record_cache_probe(cache_root, cache_key):
            cache_keys.append(str(cache_key))
            return None

        with (
            mock.patch.object(
                inspection_pipeline,
                "load_cached_chunk_snapshot_metadata",
                return_value=stale_metadata,
            ),
            mock.patch.object(inspection_pipeline, "load_chunks_from_lexical_index", return_value=None),
            mock.patch.object(inspection_pipeline, "load_cached_chunk_snapshot", return_value=None),
            mock.patch.object(
                inspection_pipeline,
                "build_syntax_chunks",
                return_value=(chunks, {"total_files": 2, "reused_files": 1, "rebuilt_files": 1, "snapshot_cache_hit": False}),
            ),
            mock.patch.object(
                inspection_pipeline,
                "inspection_index_fingerprint",
                side_effect=["sha256:stale-fingerprint", "sha256:fresh-fingerprint"],
            ),
            mock.patch.object(inspection_pipeline, "_load_query_stage_cache", side_effect=record_cache_probe),
        ):
            payload = self.run_pipeline(mode="evidence", services=all_services(), factory=SemanticDiagnosticsFactory())

        self.assertFalse(payload["retrieval"]["query_stage_cache_hit"])
        self.assertEqual(payload["retrieval"]["fingerprint"], "sha256:fresh-fingerprint")
        self.assertEqual(len(cache_keys), 2)
        self.assertNotEqual(cache_keys[0], cache_keys[1])

    def test_pipeline_prefers_lexical_manifest_chunk_reload_before_sqlite_scan(self):
        cache_dir = self.root / ".broker" / "inspection-test"
        repository_state_fingerprint, _ = inspection_index.repository_fingerprint(
            self.discovered,
            cache_dir=cache_dir,
        )
        chunks = inspection_index.build_syntax_chunks(
            self.discovered,
            cache_dir=cache_dir,
            repository_state_fingerprint=repository_state_fingerprint,
        )
        metadata = inspection_index.load_cached_chunk_snapshot_metadata(
            self.discovered,
            cache_dir=cache_dir,
            repository_state_fingerprint=repository_state_fingerprint,
        )
        fingerprint = inspection_index.inspection_index_fingerprint(repository_state_fingerprint, chunks)
        inspection_index.ensure_lexical_index(
            chunks,
            cache_dir,
            fingerprint,
            build_config_digest=str((metadata or {}).get("build_config_digest") or ""),
        )

        with mock.patch.object(
            inspection_pipeline,
            "load_chunks_from_lexical_index",
            side_effect=AssertionError("warm pipeline should reload chunk metadata from lexical manifest before sqlite scan"),
        ):
            payload = inspection_pipeline.run_inspection(
                self.discovered,
                "Trace the retry_job service call chain",
                mode="evidence",
                constraints={},
                task_params={},
                services=all_services(),
                client_factory=SemanticDiagnosticsFactory(),
                output_dir=cache_dir,
            )["payload"]

        self.assertIn("retrieval", payload)
        self.assertEqual(payload["retrieval"]["fingerprint"], fingerprint)

    def test_shared_query_stage_cache_reuses_identical_query_across_local_cache_roots(self):
        shared_cache_dir = self.root / ".broker" / "shared-query-stage"
        first_cache_dir = self.root / ".broker" / "inspection-one"
        second_cache_dir = self.root / ".broker" / "inspection-two"
        first_factory = SemanticDiagnosticsFactory()
        with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
            first = inspection_pipeline.run_inspection(
                self.discovered,
                "Trace the retry_job service call chain",
                mode="evidence",
                constraints={},
                task_params={},
                services=all_services(),
                client_factory=first_factory,
                output_dir=first_cache_dir,
            )["payload"]
            second_factory = SemanticDiagnosticsFactory()
            with (
                mock.patch.object(
                    inspection_pipeline,
                    "lexical_search",
                    side_effect=AssertionError("shared query-stage cache hit should skip lexical search"),
                ),
                mock.patch.object(
                    inspection_pipeline,
                    "ensure_lexical_index",
                    side_effect=AssertionError("shared query-stage cache hit should skip lexical index setup"),
                ),
            ):
                second = inspection_pipeline.run_inspection(
                    self.discovered,
                    "Trace the retry_job service call chain",
                    mode="evidence",
                    constraints={},
                    task_params={},
                    services=all_services(),
                    client_factory=second_factory,
                    output_dir=second_cache_dir,
                )["payload"]

        self.assertFalse(first["retrieval"]["query_stage_cache_hit"])
        self.assertTrue(second["retrieval"]["query_stage_cache_hit"])
        self.assertEqual(second_factory.ensure_semantic_index_calls, [])
        self.assertEqual(second_factory.semantic_search_calls, [])
        self.assertEqual(second_factory.rerank_calls, [])

    def test_load_query_stage_cache_reuses_shared_copy_without_local_write_through(self):
        shared_cache_dir = self.root / ".broker" / "shared-query-stage"
        cache_key = "sha256:test-query-stage"
        shared_payload = {
            "schema": inspection_pipeline.QUERY_STAGE_CACHE_SCHEMA,
            "retrieval_signature": {"tier": "p40-retrieval"},
            "retrieval_quality": "gpu",
            "rerank_quality": "gpu",
            "ranked": [{"chunk_id": "chunk-1", "score": 1.0, "rank": 1}],
            "selected": [{"chunk_id": "chunk-1", "rank": 1, "evidence_id": "ev_001"}],
            "evidence": [{"id": "ev_001", "chunk_id": "chunk-1", "path": "worker.py", "content": "def worker(): pass"}],
            "evidence_budget_trimmed": False,
            "answer": "",
            "findings": [],
            "warnings": [],
            "provenance": {},
            "runtime_attempts": [],
            "synthesis_quality": "not_requested",
        }
        local_cache_dir = self.root / ".broker" / "inspection-local"

        with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
            shared_path = inspection_pipeline._shared_query_stage_cache_path(cache_key)
            shared_path.parent.mkdir(parents=True, exist_ok=True)
            shared_path.write_text(json.dumps(shared_payload, separators=(",", ":")), encoding="utf-8")
            local_path = inspection_pipeline._query_stage_cache_path(local_cache_dir, cache_key)

            with mock.patch.object(
                inspection_pipeline.json,
                "dump",
                side_effect=AssertionError("shared query-stage restore should not reserialize the shared cache file"),
            ):
                payload = inspection_pipeline._load_query_stage_cache(local_cache_dir, cache_key)

        self.assertIsNotNone(payload)
        self.assertFalse(local_path.exists())

    def test_load_query_stage_cache_reuses_in_process_memory_cache_after_shared_hit(self):
        shared_cache_dir = self.root / ".broker" / "shared-query-stage"
        cache_key = "sha256:test-query-stage-shared-memory"
        shared_payload = {
            "schema": inspection_pipeline.QUERY_STAGE_CACHE_SCHEMA,
            "retrieval_signature": {"tier": "p40-retrieval"},
            "retrieval_quality": "gpu",
            "rerank_quality": "gpu",
            "ranked": [{"chunk_id": "chunk-1", "score": 1.0, "rank": 1}],
            "selected": [{"chunk_id": "chunk-1", "rank": 1, "evidence_id": "ev_001"}],
            "evidence": [{"id": "ev_001", "chunk_id": "chunk-1", "path": "worker.py", "content": "def worker(): pass"}],
            "evidence_budget_trimmed": False,
            "answer": "",
            "findings": [],
            "warnings": [],
            "provenance": {},
            "runtime_attempts": [],
            "synthesis_quality": "not_requested",
        }
        local_cache_dir = self.root / ".broker" / "inspection-local"

        with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
            shared_path = inspection_pipeline._shared_query_stage_cache_path(cache_key)
            shared_path.parent.mkdir(parents=True, exist_ok=True)
            shared_path.write_text(json.dumps(shared_payload, separators=(",", ":")), encoding="utf-8")
            local_path = inspection_pipeline._query_stage_cache_path(local_cache_dir, cache_key)
            inspection_pipeline._QUERY_STAGE_MEMORY_CACHE.clear()
            first = inspection_pipeline._load_query_stage_cache(local_cache_dir, cache_key)

            original_read_text = Path.read_text

            def guarded_read_text(path_self, *args, **kwargs):
                if path_self == shared_path or path_self == local_path:
                    raise AssertionError("second shared query-stage hit should reuse in-process memory cache")
                return original_read_text(path_self, *args, **kwargs)

            with mock.patch.object(Path, "read_text", autospec=True, side_effect=guarded_read_text):
                second = inspection_pipeline._load_query_stage_cache(local_cache_dir, cache_key)

        self.assertEqual(first, second)

    def test_load_query_stage_cache_reuses_in_process_memory_cache_on_second_hit(self):
        cache_key = "sha256:test-query-stage-memory"
        local_cache_dir = self.root / ".broker" / "inspection-local"
        local_path = inspection_pipeline._query_stage_cache_path(local_cache_dir, cache_key)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(
            json.dumps(
                {
                    "schema": inspection_pipeline.QUERY_STAGE_CACHE_SCHEMA,
                    "retrieval_signature": {"tier": "p40-retrieval"},
                    "retrieval_quality": "gpu",
                    "rerank_quality": "gpu",
                    "ranked": [{"chunk_id": "chunk-1", "rrf_score": 1.0, "rank": 1}],
                    "selected": [{"chunk_id": "chunk-1", "rank": 1}],
                    "evidence": [{"id": "ev_001", "chunk_id": "chunk-1", "path": "worker.py", "content": "def worker(): pass"}],
                    "evidence_budget_trimmed": False,
                    "answer": "",
                    "findings": [],
                    "warnings": [],
                    "provenance": {},
                    "runtime_attempts": [],
                    "synthesis_quality": "not_requested",
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )

        inspection_pipeline._QUERY_STAGE_MEMORY_CACHE.clear()
        first = inspection_pipeline._load_query_stage_cache(local_cache_dir, cache_key)

        original_read_text = Path.read_text

        def guarded_read_text(path_self, *args, **kwargs):
            if path_self == local_path:
                raise AssertionError("second in-process query-stage cache hit should not reread the cache file")
            return original_read_text(path_self, *args, **kwargs)

        with mock.patch.object(Path, "read_text", autospec=True, side_effect=guarded_read_text):
            second = inspection_pipeline._load_query_stage_cache(local_cache_dir, cache_key)

        self.assertIsNotNone(first)
        self.assertEqual(first, second)

    def test_load_query_stage_cache_does_not_touch_mtime_on_hit(self):
        cache_key = "sha256:test-query-stage-no-touch"
        local_cache_dir = self.root / ".broker" / "inspection-local-no-touch"
        local_path = inspection_pipeline._query_stage_cache_path(local_cache_dir, cache_key)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(
            json.dumps(
                {
                    "schema": inspection_pipeline.QUERY_STAGE_CACHE_SCHEMA,
                    "retrieval_signature": {"tier": "p40-retrieval"},
                    "retrieval_quality": "gpu",
                    "rerank_quality": "gpu",
                    "ranked": [{"chunk_id": "chunk-1", "rrf_score": 1.0, "rank": 1}],
                    "selected": [{"chunk_id": "chunk-1", "rank": 1}],
                    "evidence": [{"id": "ev_001", "chunk_id": "chunk-1", "path": "worker.py", "content": "def worker(): pass"}],
                    "evidence_budget_trimmed": False,
                    "answer": "",
                    "findings": [],
                    "warnings": [],
                    "provenance": {},
                    "runtime_attempts": [],
                    "synthesis_quality": "not_requested",
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )

        inspection_pipeline._QUERY_STAGE_MEMORY_CACHE.clear()
        first = inspection_pipeline._load_query_stage_cache(local_cache_dir, cache_key)
        self.assertIsNotNone(first)

        with mock.patch.object(inspection_pipeline.os, "utime", side_effect=AssertionError("query-stage cache hit should not touch mtime")):
            second = inspection_pipeline._load_query_stage_cache(local_cache_dir, cache_key)

        self.assertEqual(first, second)

    def test_write_query_stage_cache_publishes_shared_copy_and_alias_without_local_write_through(self):
        shared_cache_dir = self.root / ".broker" / "shared-query-stage"
        local_cache_dir = self.root / ".broker" / "inspection-local"
        cache_key = "sha256:test-query-stage-write"
        query = "trace worker"
        retrieval_signature = {"tier": "p40-retrieval"}
        repository_state_fingerprint = "sha256:test-repo-fp"
        build_config_digest = "sha256:test-build-config"
        budgets = {
            "retrieval_token_budget": 16000,
            "evidence_token_budget": 4000,
            "final_pack_token_budget": 2048,
            "synthesis_context_token_budget": 16000,
        }

        with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
            shared_path = inspection_pipeline._shared_query_stage_cache_path(cache_key)
            local_path = inspection_pipeline._query_stage_cache_path(local_cache_dir, cache_key)
            alias_key = inspection_pipeline._query_stage_cache_alias_key(
                query,
                repository_state_fingerprint,
                build_config_digest,
                retrieval_signature,
                budgets,
            )
            shared_alias_path = inspection_pipeline._shared_query_stage_cache_alias_path(alias_key)
            local_alias_path = inspection_pipeline._query_stage_cache_alias_path(local_cache_dir, alias_key)
            wrote = inspection_pipeline._write_query_stage_cache(
                local_cache_dir,
                query,
                cache_key,
                retrieval_signature,
                [{"chunk_id": "chunk-1", "score": 1.0, "rank": 1}],
                [{"chunk_id": "chunk-1", "rank": 1, "evidence_id": "ev_001"}],
                [{"id": "ev_001", "chunk_id": "chunk-1", "path": "worker.py", "content": "def worker(): pass"}],
                False,
                repository_state_fingerprint=repository_state_fingerprint,
                build_config_digest=build_config_digest,
                budgets=budgets,
            )

        self.assertTrue(wrote)
        self.assertFalse(local_path.exists())
        self.assertTrue(shared_path.exists())
        self.assertFalse(local_alias_path.exists())
        self.assertTrue(shared_alias_path.exists())

    def test_write_query_stage_cache_does_not_prune_on_every_write(self):
        local_cache_dir = self.root / ".broker" / "inspection-local"
        cache_key = "sha256:test-query-stage-prune"

        original_counter = inspection_pipeline._QUERY_STAGE_PRUNE_COUNTER
        try:
            inspection_pipeline._QUERY_STAGE_PRUNE_COUNTER = 0
            with mock.patch.object(
                inspection_pipeline,
                "_prune_query_stage_cache",
                side_effect=AssertionError("query-stage cache should not prune on every write"),
            ):
                wrote = inspection_pipeline._write_query_stage_cache(
                    local_cache_dir,
                    "trace worker",
                    cache_key,
                    {"tier": "p40-retrieval"},
                    [{"chunk_id": "chunk-1", "score": 1.0, "rank": 1}],
                    [{"chunk_id": "chunk-1", "rank": 1, "evidence_id": "ev_001"}],
                    [{"id": "ev_001", "chunk_id": "chunk-1", "path": "worker.py", "content": "def worker(): pass"}],
                    False,
                )
        finally:
            inspection_pipeline._QUERY_STAGE_PRUNE_COUNTER = original_counter

        self.assertTrue(wrote)

    def test_write_query_stage_cache_skips_rewriting_identical_local_and_shared_payloads(self):
        shared_cache_dir = self.root / ".broker" / "shared-query-stage"
        local_cache_dir = self.root / ".broker" / "inspection-local"
        cache_key = "sha256:test-query-stage-idempotent"

        kwargs = dict(
            cache_dir=local_cache_dir,
            query="trace worker",
            cache_key=cache_key,
            retrieval_signature={"tier": "p40-retrieval"},
            ranked=[{"chunk_id": "chunk-1", "rrf_score": 1.0, "rank": 1}],
            selected=[{"chunk_id": "chunk-1", "rank": 1}],
            evidence=[{"id": "ev_001", "chunk_id": "chunk-1", "path": "worker.py", "content": "def worker(): pass"}],
            evidence_budget_trimmed=False,
            repository_state_fingerprint="sha256:repo",
            build_config_digest="sha256:build",
            index_fingerprint="sha256:index",
            total_files=1,
            chunk_count=1,
            budgets={
                "retrieval_token_budget": 1,
                "evidence_token_budget": 1,
                "final_pack_token_budget": 1,
                "synthesis_context_token_budget": 1,
            },
        )

        with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
            first = inspection_pipeline._write_query_stage_cache(**kwargs)
            self.assertTrue(first)
            with mock.patch.object(
                inspection_pipeline,
                "_atomic_private_bytes",
                side_effect=AssertionError("identical query-stage cache write should not rewrite bytes"),
            ):
                second = inspection_pipeline._write_query_stage_cache(**kwargs)

        self.assertTrue(second)

    def test_write_query_stage_cache_reuses_in_process_memory_before_disk_equality_check(self):
        shared_cache_dir = self.root / ".broker" / "shared-query-stage"
        local_cache_dir = self.root / ".broker" / "inspection-local"
        cache_key = "sha256:test-query-stage-memory"

        kwargs = dict(
            cache_dir=local_cache_dir,
            query="trace worker",
            cache_key=cache_key,
            retrieval_signature={"tier": "p40-retrieval"},
            ranked=[{"chunk_id": "chunk-1", "rrf_score": 1.0, "rank": 1}],
            selected=[{"chunk_id": "chunk-1", "rank": 1}],
            evidence=[{"id": "ev_001", "chunk_id": "chunk-1", "path": "worker.py", "content": "def worker(): pass"}],
            evidence_budget_trimmed=False,
            repository_state_fingerprint="sha256:repo",
            build_config_digest="sha256:build",
            index_fingerprint="sha256:index",
            total_files=1,
            chunk_count=1,
            budgets={
                "retrieval_token_budget": 1,
                "evidence_token_budget": 1,
                "final_pack_token_budget": 1,
                "synthesis_context_token_budget": 1,
            },
        )

        with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
            first = inspection_pipeline._write_query_stage_cache(**kwargs)
            self.assertTrue(first)
            with mock.patch.object(
                inspection_pipeline,
                "_path_bytes_equal",
                side_effect=AssertionError("identical in-process query-stage write should be resolved from memory cache"),
            ):
                second = inspection_pipeline._write_query_stage_cache(**kwargs)

        self.assertTrue(second)

    def test_write_query_stage_cache_writes_only_shared_payload_and_alias_when_shared_cache_enabled(self):
        shared_cache_dir = self.root / ".broker" / "shared-query-stage"
        local_cache_dir = self.root / ".broker" / "inspection-local"
        cache_key = "sha256:test-query-stage-promote"

        kwargs = dict(
            cache_dir=local_cache_dir,
            query="trace worker",
            cache_key=cache_key,
            retrieval_signature={"tier": "p40-retrieval"},
            ranked=[{"chunk_id": "chunk-1", "rrf_score": 1.0, "rank": 1}],
            selected=[{"chunk_id": "chunk-1", "rank": 1}],
            evidence=[{"id": "ev_001", "chunk_id": "chunk-1", "path": "worker.py", "content": "def worker(): pass"}],
            evidence_budget_trimmed=False,
            repository_state_fingerprint="sha256:repo",
            build_config_digest="sha256:build",
            budgets={
                "retrieval_token_budget": 1,
                "evidence_token_budget": 1,
                "final_pack_token_budget": 1,
                "synthesis_context_token_budget": 1,
            },
        )

        atomic_calls = []
        original_atomic = inspection_pipeline._atomic_private_bytes

        def recording_atomic(path, payload):
            atomic_calls.append(Path(path))
            return original_atomic(path, payload)

        with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
            shared_payload_path = inspection_pipeline._shared_query_stage_cache_path(cache_key)
            with mock.patch.object(inspection_pipeline, "_atomic_private_bytes", side_effect=recording_atomic):
                wrote = inspection_pipeline._write_query_stage_cache(**kwargs)
            alias_key = inspection_pipeline._query_stage_cache_alias_key(
                kwargs["query"],
                kwargs["repository_state_fingerprint"],
                kwargs["build_config_digest"],
                kwargs["retrieval_signature"],
                kwargs["budgets"],
            )
            shared_alias_path = inspection_pipeline._shared_query_stage_cache_alias_path(alias_key)

        self.assertTrue(wrote)
        self.assertEqual(len(atomic_calls), 2)
        self.assertTrue(any(path == shared_payload_path for path in atomic_calls))
        self.assertTrue(any(path == shared_alias_path for path in atomic_calls))
        self.assertFalse(inspection_pipeline._query_stage_cache_path(local_cache_dir, cache_key).exists())
        self.assertFalse(inspection_pipeline._query_stage_cache_alias_path(local_cache_dir, alias_key).exists())

    def test_write_query_stage_cache_persists_only_needed_released_artifact_payloads(self):
        local_cache_dir = self.root / ".broker" / "inspection-local"
        cache_key = "sha256:test-query-stage-artifact-filter"

        wrote = inspection_pipeline._write_query_stage_cache(
            local_cache_dir,
            "trace worker",
            cache_key,
            {"tier": "cpu-lexical-fallback"},
            [{"chunk_id": "chunk-1", "rrf_score": 1.0, "rank": 1}],
            [{"chunk_id": "chunk-1", "rank": 1}],
            [{"id": "ev_001", "chunk_id": "chunk-1", "path": "worker.py", "content": "def worker(): pass"}],
            False,
            released_payload={
                "quality": {"result": "evidence_only"},
                "evidence": [{"id": "ev_001"}],
                "warnings": [],
                "provenance": {},
            },
            released_artifact_payloads={
                "evidence_pack": {"keep": True},
                "chunk_manifest": {"keep": True},
                "runtime_diagnostics": {"drop": True},
                "retrieval_result": {"drop": True},
            },
            repository_state_fingerprint="sha256:repo",
            build_config_digest="sha256:build",
            budgets={
                "retrieval_token_budget": 1,
                "evidence_token_budget": 1,
                "final_pack_token_budget": 1,
                "synthesis_context_token_budget": 1,
            },
        )

        self.assertTrue(wrote)
        cached = inspection_pipeline._load_query_stage_cache(local_cache_dir, cache_key)
        self.assertEqual(
            sorted((cached or {}).get("released_artifact_payloads", {}).keys()),
            ["chunk_manifest", "evidence_pack"],
        )

    def test_load_query_stage_cache_alias_reuses_shared_copy_without_local_write_through(self):
        shared_cache_dir = self.root / ".broker" / "shared-query-stage"
        local_cache_dir = self.root / ".broker" / "inspection-local"
        retrieval_signature = {"tier": "p40-retrieval"}
        budgets = {
            "retrieval_token_budget": 16000,
            "evidence_token_budget": 4000,
            "final_pack_token_budget": 2048,
            "synthesis_context_token_budget": 16000,
        }
        alias_key = inspection_hotpath.query_stage_cache_alias_key(
            "trace worker",
            "sha256:test-repo-fp",
            "sha256:test-build-config",
            retrieval_signature,
            budgets,
        )

        with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
            shared_alias_path = inspection_pipeline._shared_query_stage_cache_alias_path(alias_key)
            shared_alias_path.parent.mkdir(parents=True, exist_ok=True)
            shared_alias_path.write_text(
                json.dumps(
                    {
                        "schema": inspection_pipeline.QUERY_STAGE_CACHE_ALIAS_SCHEMA,
                        "cache_key": "sha256:test-cache-key",
                        "index_fingerprint": "sha256:test-index",
                        "total_files": 1,
                        "chunk_count": 1,
                    },
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            local_alias_path = inspection_pipeline._query_stage_cache_alias_path(local_cache_dir, alias_key)
            payload = inspection_hotpath.load_query_stage_cache_alias(local_cache_dir, alias_key)

        self.assertIsNotNone(payload)
        self.assertFalse(local_alias_path.exists())

    def test_load_query_stage_cache_alias_reuses_in_process_memory_cache_on_second_hit(self):
        shared_cache_dir = self.root / ".broker" / "shared-query-stage"
        local_cache_dir = self.root / ".broker" / "inspection-local"
        retrieval_signature = {"tier": "p40-retrieval"}
        budgets = {
            "retrieval_token_budget": 16000,
            "evidence_token_budget": 4000,
            "final_pack_token_budget": 2048,
            "synthesis_context_token_budget": 16000,
        }
        alias_key = inspection_hotpath.query_stage_cache_alias_key(
            "trace worker",
            "sha256:test-repo-fp-2",
            "sha256:test-build-config-2",
            retrieval_signature,
            budgets,
        )

        with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
            shared_alias_path = inspection_pipeline._shared_query_stage_cache_alias_path(alias_key)
            shared_alias_path.parent.mkdir(parents=True, exist_ok=True)
            shared_alias_path.write_text(
                json.dumps(
                    {
                        "schema": inspection_pipeline.QUERY_STAGE_CACHE_ALIAS_SCHEMA,
                        "cache_key": "sha256:test-cache-key-2",
                        "index_fingerprint": "sha256:test-index-2",
                        "total_files": 1,
                        "chunk_count": 1,
                    },
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            local_alias_path = inspection_pipeline._query_stage_cache_alias_path(local_cache_dir, alias_key)
            inspection_hotpath._QUERY_STAGE_ALIAS_MEMORY_CACHE.clear()
            first = inspection_hotpath.load_query_stage_cache_alias(local_cache_dir, alias_key)

            original_read_text = Path.read_text

            def guarded_read_text(path_self, *args, **kwargs):
                if path_self == local_alias_path or path_self == shared_alias_path:
                    raise AssertionError("second alias hit should reuse in-process memory cache")
                return original_read_text(path_self, *args, **kwargs)

            with mock.patch.object(Path, "read_text", autospec=True, side_effect=guarded_read_text):
                second = inspection_hotpath.load_query_stage_cache_alias(local_cache_dir, alias_key)

        self.assertEqual(first, second)

    def test_query_stage_cache_invalidates_when_repository_changes(self):
        self.run_pipeline(mode="evidence", services=all_services(), factory=SemanticDiagnosticsFactory())
        (self.root / "service.py").write_text(
            "def retry_job(job_id):\n    value = submit_job(job_id)\n    return value\n\ndef submit_job(job_id):\n    return job_id\n",
            encoding="utf-8",
        )

        factory = SemanticDiagnosticsFactory()
        payload = self.run_pipeline(mode="evidence", services=all_services(), factory=factory)

        self.assertFalse(payload["retrieval"]["query_stage_cache_hit"])
        self.assertTrue(factory.ensure_semantic_index_calls)
        self.assertTrue(factory.semantic_search_calls)
        self.assertTrue(factory.rerank_calls)

    def test_second_identical_cpu_only_query_reuses_query_stage_cache(self):
        first = self.run_pipeline(mode="evidence", services=[], factory=SemanticDiagnosticsFactory())
        second_factory = SemanticDiagnosticsFactory()
        with (
            mock.patch.object(
                inspection_pipeline,
                "build_syntax_chunks",
                side_effect=AssertionError("query-stage cache hit should skip full chunk snapshot loading"),
            ),
            mock.patch.object(
                inspection_pipeline,
                "lexical_search",
                side_effect=AssertionError("query-stage cache hit should skip lexical search"),
            ),
            mock.patch.object(
                inspection_pipeline,
                "ensure_lexical_index",
                side_effect=AssertionError("query-stage cache hit should skip lexical index setup"),
            ),
        ):
            second = self.run_pipeline(mode="evidence", services=[], factory=second_factory)

        self.assertFalse(first["retrieval"]["query_stage_cache_hit"])
        self.assertTrue(second["retrieval"]["query_stage_cache_hit"])
        self.assertEqual(second["quality"]["retrieval"], "lexical_degraded")
        self.assertEqual(second["quality"]["reranking"], "unavailable")
        self.assertEqual(second["retrieval"]["lexical_candidates"], 0)
        self.assertEqual(second["retrieval"]["semantic_candidates"], 0)
        self.assertEqual(second["runtime"]["attempts"], [])
        self.assertEqual(second_factory.ensure_semantic_index_calls, [])
        self.assertEqual(second_factory.semantic_search_calls, [])
        self.assertEqual(second_factory.rerank_calls, [])

    def test_second_identical_answer_query_reuses_cached_answer_without_synthesis(self):
        first_factory = SemanticDiagnosticsFactory()
        first = self.run_pipeline(mode="answer", services=all_services(), factory=first_factory)
        second_factory = SemanticDiagnosticsFactory()
        with (
            mock.patch.object(
                inspection_pipeline,
                "build_syntax_chunks",
                side_effect=AssertionError("query-stage cache hit should skip full chunk snapshot loading"),
            ),
            mock.patch.object(
                inspection_pipeline,
                "lexical_search",
                side_effect=AssertionError("query-stage cache hit should skip lexical search"),
            ),
            mock.patch.object(
                inspection_pipeline,
                "ensure_lexical_index",
                side_effect=AssertionError("query-stage cache hit should skip lexical index setup"),
            ),
        ):
            second = self.run_pipeline(mode="answer", services=all_services(), factory=second_factory)

        self.assertEqual(first["quality"]["result"], "answer_ready")
        self.assertEqual(second["quality"]["result"], "answer_ready")
        self.assertTrue(second["retrieval"]["query_stage_cache_hit"])
        self.assertEqual(second["answer"], first["answer"])
        self.assertEqual(second["findings"], first["findings"])
        self.assertEqual(second["runtime"]["attempts"], first["runtime"]["attempts"])
        self.assertEqual(second_factory.ensure_semantic_index_calls, [])
        self.assertEqual(second_factory.semantic_search_calls, [])
        self.assertEqual(second_factory.rerank_calls, [])
        self.assertEqual(second_factory.chat_calls, [])

    def test_cached_lexical_fallback_helper_returns_repeated_cpu_only_result(self):
        first = self.run_pipeline(mode="evidence", services=[], factory=SemanticDiagnosticsFactory())

        cached = inspection_cached_result.try_cached_lexical_fallback_run(
            self.discovered,
            "Trace the retry_job service call chain",
            mode="evidence",
            constraints={},
            task_params={"index_cache_dir": str(self.root / ".broker" / "inspection-test")},
            execution_plan={},
            output_dir=self.root / "out",
        )

        self.assertIsNotNone(cached)
        payload = cached["payload"]
        self.assertFalse(first["retrieval"]["query_stage_cache_hit"])
        self.assertTrue(payload["retrieval"]["query_stage_cache_hit"])
        self.assertEqual(payload["quality"]["result"], "evidence_only")
        self.assertEqual(payload["quality"]["retrieval"], "lexical_degraded")
        self.assertEqual(payload["quality"]["reranking"], "unavailable")
        self.assertEqual(payload["runtime"]["attempts"], [])
        self.assertIn("local_cache_path", payload["runtime"])
        self.assertIn("local_cache_origin", payload["runtime"])


class IndexTests(unittest.TestCase):
    def test_file_chunk_cache_write_seeds_memory_cache_for_immediate_reuse(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            cache_key = "sha256:test-bundle"
            chunks = [{
                "chunk_id": "chunk_1",
                "path": "worker.py",
                "repository_path": "worker.py",
                "source_namespace": "repo",
                "language": "python",
                "symbol": "retry_job",
                "line_start": 1,
                "line_end": 2,
                "content": "def retry_job(job_id):\n    return job_id\n",
                "content_hash": "sha256:test",
                "chunk_hash": "sha256:test",
                "token_estimate": 12,
                "input_id": "repo",
                "input_type": "repo",
                "classification": "internal",
            }]
            inspection_index._FILE_CHUNK_CACHE_PAYLOAD_MEMORY_CACHE.clear()
            inspection_index._write_file_chunk_cache(
                cache_dir,
                cache_key,
                chunks,
                publish_shared=False,
            )
            with mock.patch.object(
                Path,
                "read_text",
                side_effect=AssertionError("memory-seeded bundle reuse should avoid rereading the cache file"),
            ):
                loaded = inspection_index._load_file_chunk_cache_bundle(cache_dir, cache_key)

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["chunks"][0]["chunk_id"], "chunk_1")

    def test_state_signature_bundle_memory_cache_reuses_bundle_without_disk_lookup(self):
        cache_key = "sha256:test-bundle"
        config_key = "config-key"
        state_signature = "sha256:state"
        chunks = [{
            "chunk_id": "chunk_1",
            "path": "worker.py",
            "repository_path": "worker.py",
            "source_namespace": "repo",
            "language": "python",
            "symbol": "retry_job",
            "line_start": 1,
            "line_end": 2,
            "content": "def retry_job(job_id):\n    return job_id\n",
            "content_hash": "sha256:test",
            "chunk_hash": "sha256:test",
            "token_estimate": 12,
            "input_id": "repo",
            "input_type": "repo",
            "classification": "internal",
        }]
        lexical_record, index_signature = inspection_index._manifest_records_for_file_chunks(chunks)
        inspection_index._FILE_CHUNK_STATE_BUNDLE_MEMORY_CACHE.clear()
        manifest_cache = inspection_index._shared_file_chunk_state_manifest_cache()
        inspection_index._update_shared_file_chunk_state_manifest_cached(
            config_key,
            state_signature,
            cache_key=cache_key,
            empty=False,
            bundle={
                "chunks": chunks,
                "lexical_record": lexical_record,
                "index_signature": index_signature,
                "semantic_document_signatures": {"chunk_1": inspection_index.semantic_chunk_signature(chunks[0])},
            },
            manifest_cache=manifest_cache,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            with mock.patch.object(
                inspection_index,
                "_load_file_chunk_cache",
                side_effect=AssertionError("state-signature memory bundle reuse should avoid disk bundle lookup"),
            ):
                loaded_key, loaded_chunks = inspection_index._load_file_chunk_cache_by_state_signature(
                    cache_dir,
                    config_key,
                    state_signature,
                    manifest_cache=manifest_cache,
                )

        self.assertEqual(loaded_key, cache_key)
        self.assertEqual(loaded_chunks[0]["chunk_id"], "chunk_1")

    def test_path_bytes_equal_reuses_small_payload_memory_cache_without_reread(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.json"
            payload = b'{"files":{"a":{"signature":"sha256:x"}}}'
            inspection_index._PATH_SMALL_PAYLOAD_MEMORY_CACHE.clear()
            inspection_index._atomic_private_bytes(path, payload)

            with mock.patch.object(
                Path,
                "read_bytes",
                side_effect=AssertionError("cached small payload equality should avoid rereading file bytes"),
            ):
                self.assertTrue(inspection_index._path_bytes_equal(path, payload))

    def test_chunks_include_language_symbol_lines_and_content_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "worker.py").write_text(
                "class Worker:\n    def run(self):\n        return 1\n\ndef helper():\n    return 2\n",
                encoding="utf-8",
            )
            chunks = inspection_index.build_syntax_chunks(
                [{"id": "i", "type": "repo", "classification": "internal", "path": root}]
            )

        self.assertTrue(chunks)
        self.assertEqual({chunk["language"] for chunk in chunks}, {"python"})
        self.assertIn("Worker", {chunk["symbol"] for chunk in chunks})
        self.assertIn("Worker.run", {chunk["symbol"] for chunk in chunks})
        self.assertTrue(all(chunk["line_start"] <= chunk["line_end"] for chunk in chunks))
        self.assertTrue(all(chunk["content_hash"].startswith("sha256:") for chunk in chunks))

    def test_minified_source_line_is_split_below_upload_ceiling(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "minified.js").write_text("const payload='" + ("x" * 600_000) + "';\n", encoding="utf-8")
            chunks = inspection_index.build_syntax_chunks(
                [{"id": "i", "type": "repo", "classification": "internal", "path": root}]
            )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(
            all(len(chunk["content"].encode("utf-8")) <= inspection_index.MAX_CHUNK_BYTES for chunk in chunks)
        )

    def test_multiple_repository_inputs_have_distinct_chunk_and_source_namespaces(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            for repository in (first, second):
                (repository / "same.py").write_text("def shared_symbol():\n    return 1\n", encoding="utf-8")
            discovered = [
                {"id": "repo_a", "type": "repo", "classification": "internal", "path": first},
                {"id": "repo_b", "type": "repo", "classification": "internal", "path": second},
            ]
            chunks = inspection_index.build_syntax_chunks(discovered)
            fingerprint = inspection_index.inspection_index_fingerprint("sha256:state", chunks)
            index_path, _, _ = inspection_index.ensure_lexical_index(chunks, root / "cache", fingerprint)
            ranked = inspection_index.lexical_search(index_path, "shared_symbol", chunks)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(len({chunk["chunk_id"] for chunk in chunks}), 2)
        self.assertEqual({chunk["path"] for chunk in chunks}, {"repo_a/same.py", "repo_b/same.py"})
        self.assertEqual({chunk["input_id"] for chunk in chunks}, {"repo_a", "repo_b"})
        self.assertEqual(len(ranked), 2)

    def test_discovery_skips_symlinked_files_outside_repository(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            root = base / "repo"
            root.mkdir()
            outside = base / "outside.py"
            outside.write_text("def secret(): return 'outside'\n", encoding="utf-8")
            (root / "linked.py").symlink_to(outside)
            (root / "safe.py").write_text("def safe(): return True\n", encoding="utf-8")
            chunks = inspection_index.build_syntax_chunks(
                [{"id": "i", "type": "repo", "classification": "internal", "path": root}]
            )
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]
            fingerprint_before, _ = inspection_index.repository_fingerprint(discovered)
            outside.write_text("def secret(): return 'changed outside'\n", encoding="utf-8")
            fingerprint_after, _ = inspection_index.repository_fingerprint(discovered)

        self.assertEqual({chunk["path"] for chunk in chunks}, {"safe.py"})
        self.assertEqual(fingerprint_before, fingerprint_after)

    def test_load_cached_chunk_snapshot_returns_full_snapshot_when_fingerprint_matches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            (root / "worker.py").write_text("def worker():\n    return 1\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            repository_fp, _ = inspection_index.repository_fingerprint(
                discovered,
                excluded_paths=[cache_dir],
                cache_dir=cache_dir,
            )
            first_chunks, _ = inspection_index.build_syntax_chunks(
                discovered,
                cache_dir=cache_dir,
                excluded_paths=[cache_dir],
                repository_state_fingerprint=repository_fp,
                return_diagnostics=True,
            )
            _, namespaces, effective_excluded_paths, build_config_digest = inspection_index._snapshot_build_context(
                discovered,
                None,
                cache_dir=cache_dir,
                excluded_paths=[cache_dir],
            )
            self.assertTrue(namespaces)
            self.assertTrue(effective_excluded_paths)

            second_chunks = inspection_index.load_cached_chunk_snapshot(
                cache_dir,
                repository_state_fingerprint=repository_fp,
                build_config_digest=build_config_digest,
            )

        self.assertIsNotNone(second_chunks)
        self.assertEqual(first_chunks, second_chunks)
        self.assertTrue(all("content" in chunk for chunk in second_chunks))

    def test_load_cached_chunk_snapshot_restores_shared_snapshot_across_local_cache_roots_without_local_write_through(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            shared_cache_dir = temp_root / "shared-cache"
            first_cache_dir = temp_root / "cache-one"
            second_cache_dir = temp_root / "cache-two"
            (root / "worker.py").write_text("def worker():\n    return 1\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            repository_fp = "sha256:shared-snapshot"

            with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
                first_chunks, _ = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=first_cache_dir,
                    repository_state_fingerprint=repository_fp,
                    return_diagnostics=True,
                )
                _, _, _, build_config_digest = inspection_index._snapshot_build_context(
                    discovered,
                    None,
                    cache_dir=second_cache_dir,
                )
                with mock.patch.object(
                    inspection_index,
                    "_load_file_chunk_cache",
                    side_effect=AssertionError("shared full snapshot should avoid per-file cache stitching"),
                ):
                    second_chunks = inspection_index.load_cached_chunk_snapshot(
                        second_cache_dir,
                        repository_state_fingerprint=repository_fp,
                        build_config_digest=build_config_digest,
                    )

                self.assertEqual(first_chunks, second_chunks)
                self.assertFalse(inspection_index._file_chunk_snapshot_path(second_cache_dir).exists())
                self.assertFalse(inspection_index._file_chunk_snapshot_metadata_path(second_cache_dir).exists())

    def test_build_syntax_chunks_shared_snapshot_primary_allows_local_write_through(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            shared_cache_dir = temp_root / "shared-cache"
            cache_dir = temp_root / "cache"
            (root / "worker.py").write_text("def worker():\n    return 1\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            repository_fp = "sha256:shared-primary"

            with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
                first_chunks, first_stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    repository_state_fingerprint=repository_fp,
                    return_diagnostics=True,
                )
                self.assertTrue(inspection_index._file_chunk_snapshot_path(cache_dir).exists())
                self.assertTrue(inspection_index._file_chunk_snapshot_metadata_path(cache_dir).exists())
                second_chunks, second_stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    repository_state_fingerprint=repository_fp,
                    return_diagnostics=True,
                )

        self.assertEqual(first_chunks, second_chunks)
        self.assertEqual(first_stats["rebuilt_files"], 1)
        self.assertTrue(second_stats["snapshot_cache_hit"])

    def test_load_cached_chunk_snapshot_metadata_restores_shared_metadata_without_local_write_through(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            shared_cache_dir = temp_root / "shared-cache"
            first_cache_dir = temp_root / "cache-one"
            second_cache_dir = temp_root / "cache-two"
            (root / "worker.py").write_text("def worker():\n    return 1\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            repository_fp = "sha256:shared-snapshot"

            with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
                inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=first_cache_dir,
                    repository_state_fingerprint=repository_fp,
                    return_diagnostics=True,
                )
                local_metadata_path = inspection_index._file_chunk_snapshot_metadata_path(second_cache_dir)
                metadata = inspection_index.load_cached_chunk_snapshot_metadata(
                    discovered,
                    cache_dir=second_cache_dir,
                    repository_state_fingerprint=repository_fp,
                )
                self.assertFalse(local_metadata_path.exists())

        self.assertIsNotNone(metadata)
        self.assertEqual(int(metadata["total_files"]), 1)

    def test_load_cached_chunk_snapshot_metadata_local_hit_skips_working_manifest_read(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            cache_dir = temp_root / "cache"
            (root / "worker.py").write_text("def worker():\n    return 1\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            repository_fp = "sha256:local-snapshot"

            inspection_index.build_syntax_chunks(
                discovered,
                cache_dir=cache_dir,
                repository_state_fingerprint=repository_fp,
                return_diagnostics=True,
            )

            with mock.patch.object(
                inspection_index,
                "_load_file_chunk_working_manifest",
                side_effect=AssertionError("matching local snapshot metadata should not read the working manifest"),
            ):
                metadata = inspection_index.load_cached_chunk_snapshot_metadata(
                    discovered,
                    cache_dir=cache_dir,
                    repository_state_fingerprint=repository_fp,
                )

        self.assertIsNotNone(metadata)
        self.assertEqual(int(metadata["total_files"]), 1)

    def test_load_cached_chunk_snapshot_metadata_reuses_in_process_memory_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            cache_dir = temp_root / "cache"
            (root / "worker.py").write_text("def worker():\n    return 1\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            repository_fp = "sha256:local-snapshot-memory"

            inspection_index.build_syntax_chunks(
                discovered,
                cache_dir=cache_dir,
                repository_state_fingerprint=repository_fp,
                return_diagnostics=True,
            )
            inspection_index._SNAPSHOT_METADATA_CACHE.clear()
            first = inspection_index.load_cached_chunk_snapshot_metadata(
                discovered,
                cache_dir=cache_dir,
                repository_state_fingerprint=repository_fp,
            )

            metadata_path = inspection_index._file_chunk_snapshot_metadata_path(cache_dir)
            original_read_text = Path.read_text

            def guarded_read_text(path_self, *args, **kwargs):
                if path_self == metadata_path:
                    raise AssertionError("second metadata load should reuse in-process cache")
                return original_read_text(path_self, *args, **kwargs)

            with mock.patch.object(Path, "read_text", autospec=True, side_effect=guarded_read_text):
                second = inspection_index.load_cached_chunk_snapshot_metadata(
                    discovered,
                    cache_dir=cache_dir,
                    repository_state_fingerprint=repository_fp,
                )

        self.assertEqual(first, second)

    def test_build_syntax_chunks_reused_run_skips_manifest_and_snapshot_rewrite_without_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            cache_dir = temp_root / "cache"
            (root / "worker.py").write_text("def worker():\n    return 1\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            first_chunks, first_stats = inspection_index.build_syntax_chunks(
                discovered,
                cache_dir=cache_dir,
                excluded_paths=[cache_dir],
                return_diagnostics=True,
            )
            manifest_path = inspection_index._file_chunk_manifest_path(cache_dir)
            snapshot_path = inspection_index._file_chunk_snapshot_path(cache_dir)
            snapshot_metadata_path = inspection_index._file_chunk_snapshot_metadata_path(cache_dir)
            first_manifest_stat = manifest_path.stat()
            first_snapshot_stat = snapshot_path.stat()
            first_snapshot_meta_stat = snapshot_metadata_path.stat()

            time.sleep(0.01)

            second_chunks, second_stats = inspection_index.build_syntax_chunks(
                discovered,
                cache_dir=cache_dir,
                excluded_paths=[cache_dir],
                return_diagnostics=True,
            )
            second_manifest_stat = manifest_path.stat()
            second_snapshot_stat = snapshot_path.stat()
            second_snapshot_meta_stat = snapshot_metadata_path.stat()

        self.assertEqual(first_chunks, second_chunks)
        self.assertEqual(first_stats["rebuilt_files"], 1)
        self.assertEqual(second_stats["rebuilt_files"], 0)
        self.assertEqual(second_stats["reused_files"], 1)
        self.assertEqual(first_manifest_stat.st_mtime_ns, second_manifest_stat.st_mtime_ns)
        self.assertEqual(first_snapshot_stat.st_mtime_ns, second_snapshot_stat.st_mtime_ns)
        self.assertEqual(first_snapshot_meta_stat.st_mtime_ns, second_snapshot_meta_stat.st_mtime_ns)

    def test_discover_source_files_reused_run_skips_discovery_manifest_rewrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            cache_dir = temp_root / "cache"
            (root / "worker.py").write_text("def worker():\n    return 1\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            inspection_index.discover_source_files(
                discovered,
                cache_dir=cache_dir,
                excluded_paths=[cache_dir],
            )
            manifest_path = inspection_index._discovery_manifest_path(cache_dir)
            first_manifest_stat = manifest_path.stat()

            time.sleep(0.01)

            inspection_index.discover_source_files(
                discovered,
                cache_dir=cache_dir,
                excluded_paths=[cache_dir],
            )
            second_manifest_stat = manifest_path.stat()

        self.assertEqual(first_manifest_stat.st_mtime_ns, second_manifest_stat.st_mtime_ns)

    def test_discover_source_files_reused_run_skips_fingerprint_discovery_manifest_rewrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            cache_dir = temp_root / "cache"
            (root / "worker.py").write_text("def worker():\n    return 1\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            repository_fp = "sha256:test-fingerprint"

            inspection_index.discover_source_files(
                discovered,
                cache_dir=cache_dir,
                excluded_paths=[cache_dir],
                repository_state_fingerprint=repository_fp,
            )
            manifest_path = inspection_index._discovery_fingerprint_manifest_path(cache_dir, repository_fp)
            first_manifest_stat = manifest_path.stat()

            time.sleep(0.01)

            inspection_index.discover_source_files(
                discovered,
                cache_dir=cache_dir,
                excluded_paths=[cache_dir],
                repository_state_fingerprint=repository_fp,
            )
            second_manifest_stat = manifest_path.stat()

        self.assertEqual(first_manifest_stat.st_mtime_ns, second_manifest_stat.st_mtime_ns)

    def test_run_inspection_restores_shared_snapshot_before_rebuilding_chunks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            shared_cache_dir = temp_root / "shared-cache"
            first_cache_dir = temp_root / "cache-one"
            second_cache_dir = temp_root / "cache-two"
            first_output_dir = temp_root / "out-one"
            second_output_dir = temp_root / "out-two"
            first_output_dir.mkdir()
            second_output_dir.mkdir()
            (root / "worker.py").write_text("def retry_job(job_id):\n    return job_id\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
                first_prefetched = inspection_hotpath.prepare_prefetched_state(
                    discovered,
                    "trace retry_job",
                    mode="evidence",
                    constraints={},
                    task_params={},
                    execution_plan={"repo_inspection_cache_path": str(first_cache_dir)},
                    output_dir=first_output_dir,
                )
                inspection_pipeline.run_inspection(
                    discovered,
                    "trace retry_job",
                    mode="evidence",
                    constraints={},
                    task_params={},
                    execution_plan={"repo_inspection_cache_path": str(first_cache_dir)},
                    output_dir=first_output_dir,
                    services=[],
                    client_factory=SemanticDiagnosticsFactory(),
                    prefetched_state=first_prefetched,
                )

                second_prefetched = inspection_hotpath.prepare_prefetched_state(
                    discovered,
                    "trace another call path",
                    mode="evidence",
                    constraints={},
                    task_params={},
                    execution_plan={"repo_inspection_cache_path": str(second_cache_dir)},
                    output_dir=second_output_dir,
                )
                with mock.patch.object(
                    inspection_pipeline,
                    "build_syntax_chunks",
                    side_effect=AssertionError("shared full snapshot should avoid rebuilding chunks on fresh local cache"),
                ):
                    payload = inspection_pipeline.run_inspection(
                        discovered,
                        "trace another call path",
                        mode="evidence",
                        constraints={},
                        task_params={},
                        execution_plan={"repo_inspection_cache_path": str(second_cache_dir)},
                        output_dir=second_output_dir,
                        services=[],
                        client_factory=SemanticDiagnosticsFactory(),
                        prefetched_state=second_prefetched,
                    )["payload"]

        self.assertEqual(payload["retrieval"]["chunk_cache_reused_files"], 1)
        self.assertEqual(payload["retrieval"]["chunk_cache_rebuilt_files"], 0)

    def test_lexical_helper_persists_across_in_memory_cache_clear(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            (root / "worker.py").write_text("def worker():\n    return 1\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            repository_fp, _ = inspection_index.repository_fingerprint(
                discovered,
                excluded_paths=[cache_dir],
                cache_dir=cache_dir,
            )
            chunks, _ = inspection_index.build_syntax_chunks(
                discovered,
                cache_dir=cache_dir,
                excluded_paths=[cache_dir],
                repository_state_fingerprint=repository_fp,
                return_diagnostics=True,
            )
            fingerprint = inspection_index.inspection_index_fingerprint(repository_fp, chunks)
            index_path, _, _ = inspection_index.ensure_lexical_index(chunks, cache_dir, fingerprint)
            cache_key = inspection_index.lexical_cache_key(index_path, chunks)

            inspection_index._LEXICAL_HELPER_CACHE.clear()
            first = inspection_index.lexical_path_catalog(index_path, chunks, cache_key=cache_key)

            inspection_index._LEXICAL_HELPER_CACHE.clear()
            with mock.patch.object(
                inspection_index,
                "_build_lexical_helper",
                side_effect=AssertionError("persisted lexical helper should avoid rebuilding from chunks"),
            ), mock.patch.object(
                inspection_index,
                "_build_lexical_helper_from_index",
                side_effect=AssertionError("persisted lexical helper should avoid rebuilding from sqlite"),
            ):
                second = inspection_index.lexical_path_catalog(index_path, chunks, cache_key=cache_key)

        self.assertEqual(first, second)

    def test_run_inspection_falls_back_to_full_snapshot_when_sqlite_chunk_reload_unavailable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            output_dir = root / "out"
            output_dir.mkdir()
            (root / "worker.py").write_text("def retry_job(job_id):\n    return job_id\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            repository_fp, _ = inspection_index.repository_fingerprint(
                discovered,
                excluded_paths=[cache_dir],
                cache_dir=cache_dir,
            )
            inspection_index.build_syntax_chunks(
                discovered,
                cache_dir=cache_dir,
                excluded_paths=[cache_dir],
                repository_state_fingerprint=repository_fp,
                return_diagnostics=True,
            )
            prefetched = inspection_hotpath.prepare_prefetched_state(
                discovered,
                "trace retry_job",
                mode="evidence",
                constraints={},
                task_params={},
                execution_plan={"repo_inspection_cache_path": str(cache_dir)},
                output_dir=output_dir,
            )

            original_load_snapshot = inspection_pipeline.load_cached_chunk_snapshot
            with mock.patch.object(
                inspection_pipeline,
                "load_cached_chunk_snapshot",
                wraps=original_load_snapshot,
            ) as load_snapshot:
                payload = inspection_pipeline.run_inspection(
                    discovered,
                    "trace retry_job",
                    mode="evidence",
                    constraints={},
                    task_params={},
                    execution_plan={"repo_inspection_cache_path": str(cache_dir)},
                    output_dir=output_dir,
                    services=[],
                    client_factory=SemanticDiagnosticsFactory(),
                    prefetched_state=prefetched,
                )["payload"]

        self.assertEqual(payload["retrieval"]["setup_timings_ms"]["repository_fingerprint_ms"], 0.0)
        self.assertEqual(payload["retrieval"]["chunk_cache_reused_files"], 1)
        self.assertEqual(payload["retrieval"]["chunk_cache_rebuilt_files"], 0)
        self.assertGreaterEqual(load_snapshot.call_count, 1)

    def test_run_inspection_skips_duplicate_query_stage_cache_probe_after_prefetched_miss(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            output_dir = root / "out"
            output_dir.mkdir()
            (root / "worker.py").write_text("def retry_job(job_id):\n    return job_id\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            repository_fp, _ = inspection_index.repository_fingerprint(
                discovered,
                excluded_paths=[cache_dir],
                cache_dir=cache_dir,
            )
            inspection_index.build_syntax_chunks(
                discovered,
                cache_dir=cache_dir,
                excluded_paths=[cache_dir],
                repository_state_fingerprint=repository_fp,
                return_diagnostics=True,
            )
            prefetched = inspection_hotpath.prepare_prefetched_state(
                discovered,
                "trace retry_job",
                mode="evidence",
                constraints={},
                task_params={},
                execution_plan={"repo_inspection_cache_path": str(cache_dir)},
                output_dir=output_dir,
            )
            self.assertIsNone(prefetched["cached_query_stage"])
            self.assertTrue(prefetched["prefetched_query_stage_cache_probed"])

            with mock.patch.object(
                inspection_pipeline,
                "_load_query_stage_cache",
                side_effect=AssertionError("run_inspection should not reprobe the same prefetched query-stage cache miss"),
            ):
                payload = inspection_pipeline.run_inspection(
                    discovered,
                    "trace retry_job",
                    mode="evidence",
                    constraints={},
                    task_params={},
                    execution_plan={"repo_inspection_cache_path": str(cache_dir)},
                    output_dir=output_dir,
                    services=[],
                    client_factory=SemanticDiagnosticsFactory(),
                    prefetched_state=prefetched,
                )["payload"]

        self.assertFalse(payload["retrieval"]["query_stage_cache_hit"])

    def test_run_inspection_reuses_prefetched_query_stage_cache_hit_without_reprobe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            output_dir = root / "out"
            output_dir.mkdir()
            (root / "worker.py").write_text("def retry_job(job_id):\n    return job_id\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            repository_fp, _ = inspection_index.repository_fingerprint(
                discovered,
                excluded_paths=[cache_dir, output_dir],
                cache_dir=cache_dir,
            )
            inspection_index.build_syntax_chunks(
                discovered,
                cache_dir=cache_dir,
                excluded_paths=[cache_dir, output_dir],
                repository_state_fingerprint=repository_fp,
                return_diagnostics=True,
            )
            query = "trace retry_job"
            budgets = inspection_pipeline.normalize_token_budgets({})
            _discovered, _namespaces, _effective_excluded_paths, build_config_digest = inspection_hotpath._snapshot_build_context(
                discovered,
                set(),
                cache_dir=cache_dir,
                excluded_paths=inspection_hotpath.exclusion_paths_for_execution(
                    {"repo_inspection_cache_path": str(cache_dir)},
                    output_dir,
                ),
                transient_excluded_paths=inspection_hotpath.transient_excluded_paths_for_execution(output_dir),
            )
            cache_key = inspection_hotpath.query_stage_cache_key(
                query,
                "sha256:index-fp",
                {"tier": "cpu-lexical-fallback", "mode": "lexical_fallback"},
                budgets,
            )
            inspection_pipeline._write_query_stage_cache(
                cache_dir,
                query,
                cache_key,
                {"tier": "cpu-lexical-fallback", "mode": "lexical_fallback"},
                [{"chunk_id": "chunk-1", "rrf_score": 1.0, "rank": 1}],
                [{"chunk_id": "chunk-1", "rank": 1}],
                [{
                    "id": "ev_001",
                    "chunk_id": "chunk-1",
                    "path": "worker.py",
                    "content": "def retry_job(job_id):\n    return job_id\n",
                    "excerpt": "def retry_job(job_id):\n    return job_id\n",
                    "language": "python",
                    "rank": 1,
                    "symbol": "retry_job",
                    "source_refs": [{
                        "path": "worker.py",
                        "line_start": 1,
                        "line_end": 2,
                        "content_hash": "sha256:test",
                        "input_id": "repo",
                        "source_namespace": "repo",
                    }],
                }],
                False,
                retrieval_quality="cpu",
                rerank_quality="cpu",
                repository_state_fingerprint=repository_fp,
                build_config_digest=build_config_digest,
                index_fingerprint="sha256:index-fp",
                total_files=1,
                chunk_count=1,
                budgets=budgets,
            )

            prefetched = inspection_hotpath.prepare_prefetched_state(
                discovered,
                query,
                mode="evidence",
                constraints={},
                task_params={
                    "_broker_repository_state_fingerprint": repository_fp,
                    "_broker_repository_state_fingerprint_source": "request_cache",
                },
                execution_plan={"repo_inspection_cache_path": str(cache_dir)},
                output_dir=output_dir,
            )
            self.assertIsNotNone(prefetched["cached_query_stage"])

            with mock.patch.object(
                inspection_pipeline,
                "_load_query_stage_cache",
                side_effect=AssertionError("run_inspection should reuse prefetched query-stage cache hit without reprobe"),
            ):
                payload = inspection_pipeline.run_inspection(
                    discovered,
                    query,
                    mode="evidence",
                    constraints={},
                    task_params={
                        "_broker_repository_state_fingerprint": repository_fp,
                        "_broker_repository_state_fingerprint_source": "request_cache",
                    },
                    execution_plan={"repo_inspection_cache_path": str(cache_dir)},
                    output_dir=output_dir,
                    services=[],
                    client_factory=SemanticDiagnosticsFactory(),
                    prefetched_state=prefetched,
                )["payload"]

        self.assertTrue(payload["retrieval"]["query_stage_cache_hit"])
        self.assertEqual(payload["retrieval"]["lexical_candidates"], 0)

    def test_prepare_prefetched_state_reuses_exact_query_stage_cache_for_cpu_lexical_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            output_dir = root / "out"
            output_dir.mkdir()
            (root / "worker.py").write_text("def retry_job(job_id):\n    return job_id\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            query = "trace retry_job"
            budgets = inspection_pipeline.normalize_token_budgets({})
            repository_fp = "sha256:repo-fp"
            build_config_digest = "build-config"
            fingerprint = "sha256:index-fp"
            cached_query_stage = {
                "ranked": [{"chunk_id": "chunk-1", "rank": 1, "rrf_score": 1.0}],
                "selected": [{"chunk_id": "chunk-1", "rank": 1}],
                "evidence": [{
                    "id": "ev_001",
                    "chunk_id": "chunk-1",
                    "path": "worker.py",
                    "excerpt": "def retry_job(job_id):\n    return job_id\n",
                    "source_refs": [{
                        "path": "worker.py",
                        "line_start": 1,
                        "line_end": 2,
                        "content_hash": "sha256:test",
                        "input_id": "repo",
                        "source_namespace": "repo",
                    }],
                }],
                "retrieval_quality": "cpu",
                "rerank_quality": "cpu",
                "released_payload": {
                    "mode": "evidence",
                    "query": query,
                    "findings": [],
                    "evidence": [{
                        "id": "ev_001",
                        "chunk_id": "chunk-1",
                        "path": "worker.py",
                        "excerpt": "def retry_job(job_id):\n    return job_id\n",
                        "source_refs": [{"path": "worker.py", "line_start": 1, "line_end": 2}],
                    }],
                    "quality": {
                        "result": "evidence_only",
                        "retrieval": "cpu",
                        "reranking": "cpu",
                        "synthesis": "not_requested",
                        "answer_ready": False,
                    },
                    "warnings": ["GPU_RETRIEVAL_UNAVAILABLE_LEXICAL_FALLBACK"],
                    "provenance": {
                        "repository_fingerprint": repository_fp,
                        "index_fingerprint": fingerprint,
                    },
                },
            }
            snapshot_metadata = {
                "repository_state_fingerprint": repository_fp,
                "build_config_digest": build_config_digest,
                "index_manifest": {},
                "semantic_document_signatures": {},
                "chunk_ids": ["chunk-1"],
                "chunk_count": 1,
                "total_files": 1,
            }
            cache_key = inspection_hotpath.query_stage_cache_key(
                query,
                fingerprint,
                {"tier": "cpu-lexical-fallback", "mode": "lexical_fallback"},
                budgets,
            )

            with mock.patch.object(
                inspection_hotpath,
                "load_query_stage_cache_alias",
                return_value=None,
            ), mock.patch.object(
                inspection_hotpath,
                "repository_fingerprint",
                return_value=(repository_fp, [{"kind": "input_manifest"}]),
            ), mock.patch.object(
                inspection_hotpath,
                "_snapshot_build_context",
                return_value=(discovered, {}, set(), build_config_digest),
            ), mock.patch.object(
                inspection_hotpath,
                "load_cached_chunk_snapshot_metadata",
                return_value=snapshot_metadata,
            ), mock.patch.object(
                inspection_hotpath,
                "load_query_stage_cache",
                return_value=cached_query_stage,
            ):
                state = inspection_hotpath.prepare_prefetched_state(
                    discovered,
                    query,
                    mode="evidence",
                    constraints={},
                    task_params={
                        "_broker_repository_state_fingerprint": repository_fp,
                        "_broker_repository_state_fingerprint_source": "request_cache",
                    },
                    execution_plan={"repo_inspection_cache_path": str(cache_dir)},
                    output_dir=output_dir,
                )

        self.assertIsNotNone(state["cached_query_stage"])
        self.assertIsNotNone(state.get("cached_lexical_fallback_run"))
        self.assertFalse(state.get("prefetched_query_stage_requires_verification"))
        self.assertTrue(state["cached_lexical_fallback_run"]["payload"]["retrieval"]["query_stage_cache_hit"])

    def test_discover_source_files_uses_rg_fast_path_with_hidden_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            visible = root / "main.ts"
            visible.write_text("export const value = 1;\n", encoding="utf-8")
            hidden = root / ".env"
            hidden.write_text("PORT=3000\n", encoding="utf-8")
            ignored_dir = root / "node_modules"
            ignored_dir.mkdir()
            ignored_file = ignored_dir / "skip.js"
            ignored_file.write_text("module.exports = 1;\n", encoding="utf-8")

            def fake_run(args, check, stdout, stderr, timeout, text):
                if git_args_match(args, root, "rev-parse"):
                    raise subprocess.CalledProcessError(128, args)
                self.assertIn("--files", args)
                self.assertIn("--hidden", args)
                self.assertIn("--no-ignore", args)
                output = "\n".join([str(visible), str(hidden), str(ignored_file)]) + "\n"
                return subprocess.CompletedProcess(args, 0, stdout=output)

            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            with mock.patch.object(inspection_index.subprocess, "run", side_effect=fake_run):
                files = inspection_index.discover_source_files(discovered)

        self.assertEqual([(path.name, rel) for _item, path, rel in files], [("main.ts", "main.ts"), (".env", ".env")])

    def test_discover_source_files_prefers_git_ls_files_for_git_repo(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            visible = root / "main.ts"
            visible.write_text("export const value = 1;\n", encoding="utf-8")
            hidden = root / ".env"
            hidden.write_text("PORT=3000\n", encoding="utf-8")

            def fake_run(args, check, stdout, stderr, timeout, text):
                if git_args_match(args, root, "rev-parse"):
                    return subprocess.CompletedProcess(args, 0, stdout=(str(root) + "\n") if text else (str(root) + "\n").encode("utf-8"))
                if git_args_match(args, root, "ls-files"):
                    output = b"main.ts\0.env\0node_modules/skip.js\0"
                    return subprocess.CompletedProcess(args, 0, stdout=output)
                raise AssertionError(args)

            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            with mock.patch.object(inspection_index.subprocess, "run", side_effect=fake_run):
                files = inspection_index.discover_source_files(discovered)

        self.assertEqual([(path.name, rel) for _item, path, rel in files], [("main.ts", "main.ts"), (".env", ".env")])

    def test_discover_source_files_git_probe_cache_tracks_source_files_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "main.ts").write_text("export const value = 1;\n", encoding="utf-8")
            (root / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            (root / "archive.zip").write_bytes(b"PK\x03\x04")
            git_probe_cache = {}

            def fake_run(args, check, stdout, stderr, timeout, text):
                if git_args_match(args, root, "rev-parse"):
                    return subprocess.CompletedProcess(args, 0, stdout=(str(root) + "\n") if text else (str(root) + "\n").encode("utf-8"))
                if git_args_match(args, root, "ls-files"):
                    output = b"100644 deadbeef 0\tmain.ts\000100644 cafefood 0\tlogo.png\000100644 facefeed 0\tarchive.zip\0"
                    return subprocess.CompletedProcess(args, 0, stdout=output)
                raise AssertionError(args)

            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            with mock.patch.object(inspection_index.subprocess, "run", side_effect=fake_run):
                files = inspection_index.discover_source_files(discovered, git_probe_cache=git_probe_cache)

        self.assertEqual([(path.name, rel) for _item, path, rel in files], [("main.ts", "main.ts")])
        self.assertEqual(
            git_probe_cache.get("tracked_blob_signatures", {}).get(str(root.resolve(strict=False))),
            {"main.ts": "git:deadbeef"},
        )

    def test_file_chunk_cache_reuses_unchanged_files_and_rebuilds_only_delta(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            alpha = root / "alpha.py"
            beta = root / "beta.py"
            alpha.write_text("def alpha():\n    return 1\n", encoding="utf-8")
            beta.write_text("def beta():\n    return 2\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            with mock.patch.object(inspection_index, "symbol_markers", wraps=inspection_index.symbol_markers) as markers:
                first_chunks, first_stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    return_diagnostics=True,
                )
                self.assertEqual(
                    first_stats,
                    {"total_files": 2, "reused_files": 0, "rebuilt_files": 2, "snapshot_cache_hit": False},
                )
                self.assertEqual(markers.call_count, 2)

            beta.write_text("def beta():\n    return 3\n", encoding="utf-8")
            with mock.patch.object(inspection_index, "symbol_markers", wraps=inspection_index.symbol_markers) as markers:
                second_chunks, second_stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    return_diagnostics=True,
                )
                self.assertEqual(
                    second_stats,
                    {"total_files": 2, "reused_files": 1, "rebuilt_files": 1, "snapshot_cache_hit": False},
                )
                self.assertEqual(markers.call_count, 1)

        self.assertEqual(len(first_chunks), len(second_chunks))
        alpha_first = [chunk for chunk in first_chunks if chunk["repository_path"] == "alpha.py"]
        alpha_second = [chunk for chunk in second_chunks if chunk["repository_path"] == "alpha.py"]
        beta_first = [chunk for chunk in first_chunks if chunk["repository_path"] == "beta.py"]
        beta_second = [chunk for chunk in second_chunks if chunk["repository_path"] == "beta.py"]
        self.assertEqual(alpha_first, alpha_second)
        self.assertNotEqual(beta_first, beta_second)

    def test_cold_chunk_build_reuses_symbol_markers_for_duplicate_content_in_one_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            content = "def shared():\n    return 1\n"
            (root / "alpha.py").write_text(content, encoding="utf-8")
            (root / "beta.py").write_text(content, encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            inspection_index._SYMBOL_MARKER_MEMORY_CACHE.clear()

            original_symbol_markers = inspection_index.symbol_markers
            symbol_marker_calls = 0

            def counting_symbol_markers(text, language):
                nonlocal symbol_marker_calls
                symbol_marker_calls += 1
                return original_symbol_markers(text, language)

            with mock.patch.object(inspection_index, "symbol_markers", side_effect=counting_symbol_markers):
                chunks, stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    return_diagnostics=True,
                )

        self.assertTrue(chunks)
        self.assertEqual(stats, {"total_files": 2, "reused_files": 0, "rebuilt_files": 2, "snapshot_cache_hit": False})
        self.assertEqual(symbol_marker_calls, 1)

    def test_regex_symbol_markers_preserve_line_numbers(self):
        text = "\n".join(
            [
                "export function alpha() {}",
                "",
                "const beta = () => 1",
                "",
                "class Gamma {}",
            ]
        )

        markers = inspection_index.symbol_markers(text, "javascript")

        self.assertEqual(markers, [(1, "alpha"), (3, "beta"), (5, "Gamma")])

    def test_python_symbol_markers_preserve_class_methods(self):
        text = "\n".join(
            [
                "class Alpha:",
                "    def method(self):",
                "        return 1",
                "",
                "def top():",
                "    return 2",
            ]
        )

        markers = inspection_index.symbol_markers(text, "python")

        self.assertEqual(markers, [(1, "Alpha"), (2, "Alpha.method"), (5, "top")])

    def test_file_chunk_cache_hot_run_does_not_reopen_clean_git_tracked_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "worker.py"
            source.write_text("def worker():\n    return 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "worker.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            inspection_index.build_syntax_chunks(discovered, cache_dir=cache_dir, return_diagnostics=True)

            original_read_text = Path.read_text

            def guarded_read_text(path_self, *args, **kwargs):
                if path_self == source:
                    raise AssertionError("clean tracked file should not be reopened on hot run")
                return original_read_text(path_self, *args, **kwargs)

            with mock.patch.object(Path, "read_text", autospec=True, side_effect=guarded_read_text):
                chunks, stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    return_diagnostics=True,
                )

        self.assertTrue(chunks)
        self.assertEqual(stats, {"total_files": 1, "reused_files": 1, "rebuilt_files": 0, "snapshot_cache_hit": False})

    def test_discovery_hot_run_reuses_git_inventory_without_repo_wide_ls_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "worker.py"
            source.write_text("def worker():\n    return 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "worker.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            inspection_index.build_syntax_chunks(discovered, cache_dir=cache_dir, return_diagnostics=True)

            original_run = inspection_index.subprocess.run

            def guarded_run(args, *pargs, **kwargs):
                if (
                    len(args) >= 6
                    and git_args_match(args, root, "ls-files")
                    and "--cached" in args
                    and "--others" in args
                ):
                    raise AssertionError("hot discovery should not rerun repo-wide git ls-files")
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=guarded_run):
                chunks, stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    return_diagnostics=True,
                )

        self.assertTrue(chunks)
        self.assertEqual(stats, {"total_files": 1, "reused_files": 1, "rebuilt_files": 0, "snapshot_cache_hit": False})

    def test_discovery_after_fingerprint_reuses_cached_status_without_untracked_ls_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "worker.py"
            source.write_text("def worker():\n    return 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "worker.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            git_probe_cache = {}

            inspection_index.repository_fingerprint(
                discovered,
                cache_dir=cache_dir,
                git_probe_cache=git_probe_cache,
            )

            original_run = inspection_index.subprocess.run

            def guarded_run(args, *pargs, **kwargs):
                if (
                    len(args) >= 6
                    and git_args_match(args, root, "ls-files")
                    and "--others" in args
                    and "--exclude-standard" in args
                ):
                    raise AssertionError("discovery should reuse cached status instead of rerunning untracked ls-files after fingerprint")
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=guarded_run):
                files = inspection_index.discover_source_files(
                    discovered,
                    cache_dir=cache_dir,
                    excluded_paths={cache_dir},
                    git_probe_cache=git_probe_cache,
                )

        self.assertEqual([rel for _item, _path, rel in files], ["worker.py"])

    def test_discovery_inventory_delta_reuses_cached_status_snapshot_without_git_diff_or_untracked_probe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "worker.py"
            source.write_text("def worker():\n    return 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "worker.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            source.write_text("def worker():\n    value = 2\n    return value\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            git_probe_cache = {}

            inspection_index.repository_fingerprint(
                discovered,
                cache_dir=cache_dir,
                git_probe_cache=git_probe_cache,
            )

            original_run = inspection_index.subprocess.run

            def guarded_run(args, *pargs, **kwargs):
                if git_args_match(args, root, "diff") and "--name-only" in args:
                    raise AssertionError("prefer_inventory_delta should reuse cached scoped status snapshot before git diff probe")
                if (
                    git_args_match(args, root, "ls-files")
                    and "--others" in args
                    and "--exclude-standard" in args
                ):
                    raise AssertionError("prefer_inventory_delta should reuse cached scoped status snapshot before untracked probe")
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=guarded_run):
                files = inspection_index.discover_source_files(
                    discovered,
                    cache_dir=cache_dir,
                    excluded_paths={cache_dir},
                    git_probe_cache=git_probe_cache,
                    prefer_inventory_delta=True,
                )

        self.assertEqual([rel for _item, _path, rel in files], ["worker.py"])

    def test_discovery_hot_run_does_not_restat_untouched_cached_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "worker.py"
            source.write_text("def worker():\n    return 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "worker.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            inspection_index.discover_source_files(discovered, cache_dir=cache_dir, excluded_paths={cache_dir})

            original_is_file = Path.is_file

            def guarded_is_file(path_self):
                if path_self.resolve(strict=False) == source.resolve(strict=False):
                    raise AssertionError("untouched cached file should not be re-statted during hot discovery")
                return original_is_file(path_self)

            with mock.patch.object(Path, "is_file", autospec=True, side_effect=guarded_is_file):
                files = inspection_index.discover_source_files(discovered, cache_dir=cache_dir, excluded_paths={cache_dir})

        self.assertEqual([rel for _item, _path, rel in files], ["worker.py"])

    def test_discovery_hot_clean_run_uses_status_snapshot_without_extra_clean_probe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "worker.py"
            source.write_text("def worker():\n    return 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "worker.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            inspection_index.discover_source_files(discovered, cache_dir=cache_dir, excluded_paths={cache_dir})

            original_run = inspection_index.subprocess.run

            def guarded_run(args, *pargs, **kwargs):
                if git_args_match(args, root, "diff-index"):
                    raise AssertionError("clean hot discovery should not run diff-index clean probe")
                if git_args_match(args, root, "diff-files"):
                    raise AssertionError("clean hot discovery should not run diff-files clean probe")
                if git_args_match(args, root, "ls-files") and "--others" in args:
                    raise AssertionError("clean hot discovery should not run ls-files --others clean probe")
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(
                inspection_index.subprocess,
                "run",
                side_effect=guarded_run,
            ), mock.patch.object(
                Path,
                "is_file",
                autospec=True,
                side_effect=AssertionError("clean hot discovery should not restat unchanged cached files"),
            ):
                files = inspection_index.discover_source_files(discovered, cache_dir=cache_dir, excluded_paths={cache_dir})

        self.assertEqual([rel for _item, _path, rel in files], ["worker.py"])

    def test_load_discovery_working_manifest_reuses_in_process_cache_on_second_hit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            manifest_path = temp_root / "discovery-working-manifest.json"
            manifest_payload = {
                "schema": inspection_index.DISCOVERY_WORKING_MANIFEST_SCHEMA,
                "roots": {
                    "/tmp/repo": {
                        "git_top": "/tmp/repo",
                        "scope_rel": ".",
                        "scope_oid": "git:scope",
                        "filter_key": "sha256:filter",
                        "files": ["worker.py"],
                        "dir_signatures": {},
                    }
                },
            }
            manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
            inspection_index._DISCOVERY_WORKING_MANIFEST_CACHE.clear()

            first = inspection_index._load_discovery_working_manifest(manifest_path)
            with mock.patch.object(Path, "read_text", side_effect=AssertionError("expected discovery manifest memory cache hit")):
                second = inspection_index._load_discovery_working_manifest(manifest_path)

        self.assertEqual(second, first)

    def test_discovery_manifest_filter_key_invalidates_clean_shortcut_for_new_exclusions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            excluded_dir = root / "generated"
            excluded_dir.mkdir()
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "worker.py"
            ignored_source = excluded_dir / "ignored.py"
            source.write_text("def worker():\n    return 1\n", encoding="utf-8")
            ignored_source.write_text("def ignored():\n    return 2\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "worker.py", "generated/ignored.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            first = inspection_index.discover_source_files(discovered, cache_dir=cache_dir, excluded_paths={cache_dir})
            self.assertEqual([rel for _item, _path, rel in first], ["generated/ignored.py", "worker.py"])

            second = inspection_index.discover_source_files(
                discovered,
                cache_dir=cache_dir,
                excluded_dir_names={"generated"},
                excluded_paths={cache_dir},
            )

        self.assertEqual([rel for _item, _path, rel in second], ["worker.py"])

    def test_non_git_discovery_manifest_reuses_file_list_when_directory_signatures_match(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            alpha = root / "alpha.py"
            beta = root / "beta.py"
            alpha.write_text("def alpha():\n    return 1\n", encoding="utf-8")
            beta.write_text("def beta():\n    return 2\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            first = inspection_index.discover_source_files(discovered, cache_dir=cache_dir, excluded_paths={cache_dir})

            with mock.patch.object(
                inspection_index,
                "_iter_source_candidates",
                side_effect=AssertionError("non-git warm discovery should reuse prior file list when directory signatures match"),
            ):
                second = inspection_index.discover_source_files(
                    discovered,
                    cache_dir=cache_dir,
                    excluded_paths={cache_dir},
                )

        self.assertEqual([item[2] for item in first], [item[2] for item in second])

    def test_non_git_discovery_manifest_invalidates_when_directory_signatures_change(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            alpha = root / "alpha.py"
            beta = root / "beta.py"
            alpha.write_text("def alpha():\n    return 1\n", encoding="utf-8")
            beta.write_text("def beta():\n    return 2\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            inspection_index.discover_source_files(discovered, cache_dir=cache_dir, excluded_paths={cache_dir})
            gamma = root / "gamma.py"
            gamma.write_text("def gamma():\n    return 3\n", encoding="utf-8")

            calls = 0
            original_iter = inspection_index._iter_source_candidates

            def counting_iter(*args, **kwargs):
                nonlocal calls
                calls += 1
                yield from original_iter(*args, **kwargs)

            with mock.patch.object(inspection_index, "_iter_source_candidates", side_effect=counting_iter):
                files = inspection_index.discover_source_files(
                    discovered,
                    cache_dir=cache_dir,
                    excluded_paths={cache_dir},
                )

        self.assertGreaterEqual(calls, 1)
        self.assertEqual(sorted(rel for _item, _path, rel in files), ["alpha.py", "beta.py", "gamma.py"])

    def test_discovery_reuses_shared_manifest_across_local_cache_roots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            shared_cache_dir = temp_root / "shared-cache"
            first_cache_dir = temp_root / "cache-one"
            second_cache_dir = temp_root / "cache-two"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "worker.py"
            source.write_text("def worker():\n    return 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "worker.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
                inspection_index.discover_source_files(discovered, cache_dir=first_cache_dir, excluded_paths={first_cache_dir})

                original_is_file = Path.is_file

                def guarded_is_file(path_self):
                    if path_self.resolve(strict=False) == source.resolve(strict=False):
                        raise AssertionError("shared discovery manifest should avoid re-statting unchanged file")
                    return original_is_file(path_self)

                with mock.patch.object(Path, "is_file", autospec=True, side_effect=guarded_is_file):
                    files = inspection_index.discover_source_files(
                        discovered,
                        cache_dir=second_cache_dir,
                        excluded_paths={second_cache_dir},
                    )

        self.assertEqual([rel for _item, _path, rel in files], ["worker.py"])
        self.assertFalse(inspection_index._discovery_manifest_path(second_cache_dir).exists())

    def test_discovery_reuses_fingerprint_manifest_without_git_status_probe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            shared_cache_dir = temp_root / "shared-cache"
            first_cache_dir = temp_root / "cache-one"
            second_cache_dir = temp_root / "cache-two"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "worker.py"
            source.write_text("def worker():\n    return 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "worker.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
                repository_fingerprint, _ = inspection_index.repository_fingerprint(
                    discovered,
                    cache_dir=first_cache_dir,
                )
                inspection_index.discover_source_files(
                    discovered,
                    cache_dir=first_cache_dir,
                    excluded_paths={first_cache_dir},
                    repository_state_fingerprint=repository_fingerprint,
                )

                original_run = inspection_index.subprocess.run

                def guarded_run(args, *pargs, **kwargs):
                    if git_args_match(args, root, "diff-index"):
                        raise AssertionError("fingerprint discovery reuse should not probe diff-index")
                    if git_args_match(args, root, "diff-files"):
                        raise AssertionError("fingerprint discovery reuse should not probe diff-files")
                    if git_args_match(args, root, "ls-files") and "--others" in args:
                        raise AssertionError("fingerprint discovery reuse should not probe untracked files")
                    return original_run(args, *pargs, **kwargs)

                files = None
                with mock.patch.object(inspection_index.subprocess, "run", side_effect=guarded_run):
                    files = inspection_index.discover_source_files(
                        discovered,
                        cache_dir=second_cache_dir,
                        excluded_paths={second_cache_dir},
                        repository_state_fingerprint=repository_fingerprint,
                    )

        self.assertEqual([rel for _item, _path, rel in files], ["worker.py"])
        self.assertFalse(inspection_index._discovery_manifest_path(second_cache_dir).exists())

    def test_discovery_reuses_covering_cached_status_snapshot_from_multi_scope_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            target = root / "target"
            sibling = root / "sibling"
            target.mkdir()
            sibling.mkdir()
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            (target / "worker.py").write_text("def worker():\n    return 1\n", encoding="utf-8")
            (sibling / "helper.py").write_text("def helper():\n    return 2\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [
                {"id": "target", "type": "repo", "classification": "internal", "path": target},
                {"id": "sibling", "type": "repo", "classification": "internal", "path": sibling},
            ]
            git_probe_cache = {}

            inspection_index.repository_fingerprint(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
                git_probe_cache=git_probe_cache,
            )
            inspection_index.discover_source_files(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
                git_probe_cache=git_probe_cache,
            )

            original_run = inspection_index.subprocess.run

            def guarded_run(args, *pargs, **kwargs):
                if git_args_match(args, root, "status"):
                    raise AssertionError(
                        "warm scoped discovery should reuse covering cached status snapshot from repository fingerprint"
                    )
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=guarded_run):
                files = inspection_index.discover_source_files(
                    discovered,
                    cache_dir=cache_dir,
                    excluded_paths={cache_dir},
                    git_probe_cache=git_probe_cache,
                )

        self.assertEqual(sorted(rel for _item, _path, rel in files), ["helper.py", "worker.py"])

    def test_git_discovery_hot_run_uses_status_snapshot_instead_of_separate_clean_probe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "worker.py"
            source.write_text("def worker():\n    return 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "worker.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            inspection_index.discover_source_files(discovered, cache_dir=cache_dir, excluded_paths={cache_dir})

            original_run = inspection_index.subprocess.run

            def guarded_run(args, *pargs, **kwargs):
                if git_args_match(args, root, "diff-index"):
                    raise AssertionError("warm git discovery should not run diff-index clean probe")
                if git_args_match(args, root, "diff-files"):
                    raise AssertionError("warm git discovery should not run diff-files clean probe")
                if git_args_match(args, root, "ls-files") and "--others" in args:
                    raise AssertionError("warm git discovery should not run ls-files --others clean probe")
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=guarded_run):
                files = inspection_index.discover_source_files(
                    discovered,
                    cache_dir=cache_dir,
                    excluded_paths={cache_dir},
                )

        self.assertEqual([rel for _item, _path, rel in files], ["worker.py"])

    def test_git_file_signature_hot_run_reuses_cached_tracked_blob_map(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "worker.py"
            source.write_text("def worker():\n    return 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "worker.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            inspection_index.build_syntax_chunks(discovered, cache_dir=cache_dir, return_diagnostics=True)

            original_run = inspection_index.subprocess.run

            def guarded_run(args, *pargs, **kwargs):
                if (
                    len(args) >= 6
                    and git_args_match(args, root, "ls-files")
                    and "-s" in args
                    and "-z" in args
                ):
                    raise AssertionError("clean hot chunk build should not rerun tracked blob ls-files")
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=guarded_run):
                chunks, stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    return_diagnostics=True,
                )

        self.assertTrue(chunks)
        self.assertEqual(stats, {"total_files": 1, "reused_files": 1, "rebuilt_files": 0, "snapshot_cache_hit": False})

    def test_clean_hot_run_without_repository_fingerprint_reuses_previous_manifest_signatures(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "worker.py"
            source.write_text("def worker():\n    return 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "worker.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            first_fp, _ = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)
            inspection_index.build_syntax_chunks(
                discovered,
                cache_dir=cache_dir,
                repository_state_fingerprint=first_fp,
                return_diagnostics=True,
            )

            original_run = inspection_index.subprocess.run

            def guarded_run(args, *pargs, **kwargs):
                if (
                    len(args) >= 6
                    and git_args_match(args, root, "ls-files")
                    and "-s" in args
                    and "-z" in args
                ):
                    raise AssertionError("clean hot run without repository fingerprint should not rerun tracked blob ls-files")
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=guarded_run):
                chunks, stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    return_diagnostics=True,
                )

        self.assertTrue(chunks)
        self.assertEqual(stats, {"total_files": 1, "reused_files": 1, "rebuilt_files": 0, "snapshot_cache_hit": False})

    def test_cold_chunk_build_reuses_tracked_blob_signatures_from_discovery_probe_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "worker.py"
            source.write_text("def worker():\n    return 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "worker.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            git_probe_cache = {}

            inspection_index.repository_fingerprint(
                discovered,
                cache_dir=cache_dir,
                git_probe_cache=git_probe_cache,
            )

            original_run = inspection_index.subprocess.run
            tracked_signature_calls = 0

            def guarded_run(args, *pargs, **kwargs):
                nonlocal tracked_signature_calls
                if (
                    len(args) >= 6
                    and git_args_match(args, root, "ls-files")
                    and "-s" in args
                    and "-z" in args
                ):
                    tracked_signature_calls += 1
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=guarded_run):
                chunks, stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    git_probe_cache=git_probe_cache,
                    return_diagnostics=True,
                )

        self.assertTrue(chunks)
        self.assertEqual(stats, {"total_files": 1, "reused_files": 0, "rebuilt_files": 1, "snapshot_cache_hit": False})
        self.assertEqual(tracked_signature_calls, 1)

    def test_cold_run_reuses_repo_tree_between_fingerprint_and_git_file_signatures(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            (root / "worker.py").write_text("def worker():\n    return 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            git_probe_cache = {}

            original_run = inspection_index.subprocess.run
            tree_calls = 0

            def counting_run(args, *pargs, **kwargs):
                nonlocal tree_calls
                if git_args_match(args, root, "rev-parse") and len(args) >= 6 and args[-1] == "HEAD^{tree}":
                    tree_calls += 1
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=counting_run):
                fingerprint, _ = inspection_index.repository_fingerprint(
                    discovered,
                    cache_dir=cache_dir,
                    git_probe_cache=git_probe_cache,
                )
                chunks, stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    repository_state_fingerprint=fingerprint,
                    git_probe_cache=git_probe_cache,
                    return_diagnostics=True,
                )

        self.assertTrue(chunks)
        self.assertEqual(stats, {"total_files": 1, "reused_files": 0, "rebuilt_files": 1, "snapshot_cache_hit": False})
        self.assertEqual(tree_calls, 1)

    def test_cold_build_without_probe_cache_skips_tracked_blob_signature_ls_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            (root / "worker.py").write_text("def worker():\n    return 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            fingerprint, _ = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

            original_run = inspection_index.subprocess.run

            def guarded_run(args, *pargs, **kwargs):
                if (
                    len(args) >= 6
                    and git_args_match(args, root, "ls-files")
                    and "-s" in args
                    and "-z" in args
                    and "--others" not in args
                ):
                    raise AssertionError("true cold build without probe cache should skip tracked blob ls-files")
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=guarded_run):
                chunks, stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    repository_state_fingerprint=fingerprint,
                    return_diagnostics=True,
                )

        self.assertTrue(chunks)
        self.assertEqual(stats, {"total_files": 1, "reused_files": 0, "rebuilt_files": 1, "snapshot_cache_hit": False})

    def test_cold_build_after_fingerprint_skips_tracked_blob_signature_ls_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            (root / "worker.py").write_text("def worker():\n    return 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            git_probe_cache = {}
            fingerprint, _ = inspection_index.repository_fingerprint(
                discovered,
                cache_dir=cache_dir,
                git_probe_cache=git_probe_cache,
            )

            original_run = inspection_index.subprocess.run
            tracked_ls_files_calls = 0

            def guarded_run(args, *pargs, **kwargs):
                nonlocal tracked_ls_files_calls
                if (
                    len(args) >= 6
                    and git_args_match(args, root, "ls-files")
                    and "-s" in args
                    and "-z" in args
                    and "--others" not in args
                ):
                    tracked_ls_files_calls += 1
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=guarded_run):
                chunks, stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    repository_state_fingerprint=fingerprint,
                    git_probe_cache=git_probe_cache,
                    return_diagnostics=True,
                )

        self.assertTrue(chunks)
        self.assertEqual(stats, {"total_files": 1, "reused_files": 0, "rebuilt_files": 1, "snapshot_cache_hit": False})
        self.assertEqual(tracked_ls_files_calls, 0)

    def test_true_cold_build_after_fingerprint_skips_direct_local_bundle_probe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            (root / "worker.py").write_text("def worker():\n    return 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            fingerprint, _ = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

            with mock.patch.object(
                inspection_index,
                "_load_file_chunk_cache_bundle",
                side_effect=AssertionError("true cold fingerprinted build should skip direct local bundle probe"),
            ):
                chunks, stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    repository_state_fingerprint=fingerprint,
                    return_diagnostics=True,
                )

        self.assertTrue(chunks)
        self.assertEqual(stats, {"total_files": 1, "reused_files": 0, "rebuilt_files": 1, "snapshot_cache_hit": False})

    def test_git_file_signatures_skips_tree_probe_when_probe_cache_already_covers_clean_subset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            (root / "worker.py").write_text("def worker():\n    return 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            git_probe_cache = {}

            inspection_index.repository_fingerprint(
                discovered,
                cache_dir=cache_dir,
                git_probe_cache=git_probe_cache,
            )

            original_run = inspection_index.subprocess.run
            tree_calls = 0

            def counting_run(args, *pargs, **kwargs):
                nonlocal tree_calls
                if git_args_match(args, root, "rev-parse") and len(args) >= 6 and args[-1] == "HEAD^{tree}":
                    tree_calls += 1
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=counting_run):
                chunks, stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    git_probe_cache=git_probe_cache,
                    return_diagnostics=True,
                )

        self.assertTrue(chunks)
        self.assertEqual(stats, {"total_files": 1, "reused_files": 0, "rebuilt_files": 1, "snapshot_cache_hit": False})
        self.assertEqual(tree_calls, 0)

    def test_cold_whole_repo_discovery_reuses_head_tree_probe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            (root / "worker.py").write_text("def worker():\n    return 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            git_probe_cache = {}

            original_run = inspection_index.subprocess.run
            tree_calls = 0

            def guarded_run(args, *pargs, **kwargs):
                if git_args_match(args, root, "rev-parse") and len(args) >= 6 and args[-1] == "HEAD^{tree}":
                    nonlocal tree_calls
                    tree_calls += 1
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=guarded_run):
                fingerprint, _ = inspection_index.repository_fingerprint(
                    discovered,
                    cache_dir=cache_dir,
                    git_probe_cache=git_probe_cache,
                )
                chunks, stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    repository_state_fingerprint=fingerprint,
                    git_probe_cache=git_probe_cache,
                    return_diagnostics=True,
                )

        self.assertTrue(chunks)
        self.assertEqual(stats, {"total_files": 1, "reused_files": 0, "rebuilt_files": 1, "snapshot_cache_hit": False})
        self.assertEqual(tree_calls, 1)

    def test_write_file_chunk_snapshot_skips_rewriting_identical_local_and_shared_payloads(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            cache_dir = temp_root / "cache"
            shared_cache_dir = temp_root / "shared-cache"
            (root / "worker.py").write_text("def worker():\n    return 1\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            repository_fp = "sha256:snapshot-idempotent"

            with mock.patch.dict(
                os.environ,
                {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)},
                clear=False,
            ):
                chunks, _ = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    repository_state_fingerprint=repository_fp,
                    return_diagnostics=True,
                )
                metadata = inspection_index.load_cached_chunk_snapshot_metadata(
                    discovered,
                    cache_dir=cache_dir,
                    repository_state_fingerprint=repository_fp,
                ) or {}
                inspection_index._write_file_chunk_snapshot(
                    cache_dir,
                    repository_state_fingerprint=repository_fp,
                    build_config_digest=str(metadata.get("build_config_digest") or ""),
                    chunks=chunks,
                )
                with mock.patch.object(
                    inspection_index,
                    "_atomic_private_bytes",
                    side_effect=AssertionError("identical chunk snapshot write should not rewrite bytes"),
                ):
                    inspection_index._write_file_chunk_snapshot(
                        cache_dir,
                        repository_state_fingerprint=repository_fp,
                        build_config_digest=str(metadata.get("build_config_digest") or ""),
                        chunks=chunks,
                    )

    def test_git_file_signatures_resolve_git_top_once_per_discovered_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            (root / "pkg").mkdir()
            (root / "pkg" / "worker.py").write_text("def worker():\n    return 1\n", encoding="utf-8")
            (root / "pkg" / "helper.py").write_text("def helper():\n    return 2\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            original_run = inspection_index.subprocess.run
            rev_parse_calls = 0

            def counting_run(args, *pargs, **kwargs):
                nonlocal rev_parse_calls
                if git_args_match(args, root, "rev-parse") and "--show-toplevel" in args:
                    rev_parse_calls += 1
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=counting_run):
                chunks, stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    return_diagnostics=True,
                )

        self.assertTrue(chunks)
        self.assertEqual(stats, {"total_files": 2, "reused_files": 0, "rebuilt_files": 2, "snapshot_cache_hit": False})
        self.assertLessEqual(rev_parse_calls, 2)

    def test_git_file_signatures_reuse_shared_manifest_across_local_cache_roots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            shared_cache_dir = temp_root / "shared-cache"
            first_cache_dir = temp_root / "cache-one"
            second_cache_dir = temp_root / "cache-two"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "worker.py"
            source.write_text("def worker():\n    return 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "worker.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
                inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=first_cache_dir,
                    return_diagnostics=True,
                )

                original_run = inspection_index.subprocess.run

                def guarded_run(args, *pargs, **kwargs):
                    if (
                        git_args_match(args, root, "ls-files")
                        and "-s" in args
                    ):
                        raise AssertionError("shared git file signature manifest should avoid rerunning tracked blob ls-files")
                    return original_run(args, *pargs, **kwargs)

                with mock.patch.object(inspection_index.subprocess, "run", side_effect=guarded_run):
                    chunks, stats = inspection_index.build_syntax_chunks(
                        discovered,
                        cache_dir=second_cache_dir,
                        return_diagnostics=True,
                    )

        self.assertTrue(chunks)
        self.assertEqual(stats, {"total_files": 1, "reused_files": 1, "rebuilt_files": 0, "snapshot_cache_hit": False})

    def test_partial_dirty_hot_run_reuses_cached_clean_blob_signatures_without_ls_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            dirty_file = root / "dirty.py"
            clean_file = root / "clean.py"
            dirty_file.write_text("def dirty():\n    return 1\n", encoding="utf-8")
            clean_file.write_text("def clean():\n    return 2\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            inspection_index.build_syntax_chunks(discovered, cache_dir=cache_dir, return_diagnostics=True)
            dirty_file.write_text("def dirty():\n    value = 3\n    return value\n", encoding="utf-8")

            original_run = inspection_index.subprocess.run

            def guarded_run(args, *pargs, **kwargs):
                if (
                    len(args) >= 6
                    and git_args_match(args, root, "ls-files")
                    and "-s" in args
                    and "-z" in args
                ):
                    raise AssertionError("partial-dirty hot run should reuse cached clean blob signatures without ls-files")
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=guarded_run):
                chunks, stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    return_diagnostics=True,
                )

        self.assertTrue(chunks)
        self.assertEqual(stats["total_files"], 2)
        self.assertEqual(stats["reused_files"], 1)
        self.assertGreaterEqual(stats["rebuilt_files"], 1)

    def test_partial_dirty_discover_source_files_avoids_git_status_for_tracked_content_edit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            dirty_file = root / "dirty.py"
            clean_file = root / "clean.py"
            dirty_file.write_text("def dirty():\n    return 1\n", encoding="utf-8")
            clean_file.write_text("def clean():\n    return 2\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            first = inspection_index.discover_source_files(discovered, cache_dir=cache_dir, excluded_paths={cache_dir})
            dirty_file.write_text("def dirty():\n    value = 3\n    return value\n", encoding="utf-8")

            original_run = inspection_index.subprocess.run

            def guarded_run(args, *pargs, **kwargs):
                if git_args_match(args, root, "status"):
                    raise AssertionError("tracked content-only discovery refresh should avoid git status")
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=guarded_run):
                second = inspection_index.discover_source_files(
                    discovered,
                    cache_dir=cache_dir,
                    excluded_paths={cache_dir},
                    prefer_inventory_delta=True,
                )

        self.assertEqual(
            sorted((path.relative_to(root).as_posix() for _item, path, _rel in first)),
            sorted((path.relative_to(root).as_posix() for _item, path, _rel in second)),
        )

    def test_partial_dirty_medium_repo_build_uses_inventory_delta_automatically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            for index in range(66):
                (root / f"file_{index:02d}.py").write_text(
                    f"def file_{index:02d}():\n    return {index}\n",
                    encoding="utf-8",
                )
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            git_probe_cache = {}

            clean_fp, _ = inspection_index.repository_fingerprint(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
                git_probe_cache=git_probe_cache,
            )
            inspection_index.build_syntax_chunks(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
                repository_state_fingerprint=clean_fp,
                git_probe_cache=git_probe_cache,
                return_diagnostics=True,
            )

            dirty_file = root / "file_00.py"
            dirty_file.write_text("def file_00():\n    value = 100\n    return value\n", encoding="utf-8")
            dirty_fp, _ = inspection_index.repository_fingerprint(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
                git_probe_cache=git_probe_cache,
            )

            original_run = inspection_index.subprocess.run

            def guarded_run(args, *pargs, **kwargs):
                if git_args_match(args, root, "diff") and "--name-only" in args:
                    raise AssertionError("medium partial-dirty build should use inventory delta before git diff probe")
                if (
                    git_args_match(args, root, "ls-files")
                    and "--others" in args
                    and "--exclude-standard" in args
                ):
                    raise AssertionError("medium partial-dirty build should use inventory delta before untracked probe")
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=guarded_run):
                chunks, stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    excluded_paths={cache_dir},
                    repository_state_fingerprint=dirty_fp,
                    git_probe_cache=git_probe_cache,
                    return_diagnostics=True,
                )

        self.assertTrue(chunks)
        self.assertEqual(stats["total_files"], 66)
        self.assertGreaterEqual(stats["reused_files"], 65)
        dirty_chunks = [chunk for chunk in chunks if chunk["path"] == "file_00.py"]
        self.assertTrue(dirty_chunks)
        self.assertTrue(any("value = 100" in chunk["content"] for chunk in dirty_chunks))

    def test_partial_dirty_build_reuses_default_git_probe_cache_without_explicit_cache_argument(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            for index in range(8):
                (root / f"file_{index:02d}.py").write_text(
                    f"def file_{index:02d}():\n    return {index}\n",
                    encoding="utf-8",
                )
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            inspection_index._DEFAULT_GIT_PROBE_CACHE.clear()
            clean_fp, _ = inspection_index.repository_fingerprint(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
            )
            inspection_index.build_syntax_chunks(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
                repository_state_fingerprint=clean_fp,
                return_diagnostics=True,
            )

            dirty_file = root / "file_00.py"
            dirty_file.write_text("def file_00():\n    value = 100\n    return value\n", encoding="utf-8")
            dirty_fp, _ = inspection_index.repository_fingerprint(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
            )

            original_run = inspection_index.subprocess.run

            def guarded_run(args, *pargs, **kwargs):
                if git_args_match(args, root, "diff") and "--name-only" in args:
                    raise AssertionError(
                        "default git probe cache should be reused before partial-dirty inventory diff probe"
                    )
                if (
                    git_args_match(args, root, "ls-files")
                    and "--others" in args
                    and "--exclude-standard" in args
                ):
                    raise AssertionError(
                        "default git probe cache should be reused before partial-dirty untracked probe"
                    )
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=guarded_run):
                chunks, stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    excluded_paths={cache_dir},
                    repository_state_fingerprint=dirty_fp,
                    return_diagnostics=True,
                )

        self.assertTrue(chunks)
        self.assertEqual(stats["total_files"], 8)
        self.assertGreaterEqual(stats["reused_files"], 7)

    def test_build_syntax_chunks_reuses_cached_manifest_files_json_when_only_repository_fingerprint_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            (root / "main.py").write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "main.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            first_chunks, first_stats = inspection_index.build_syntax_chunks(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
                repository_state_fingerprint="sha256:first",
                return_diagnostics=True,
            )

            with mock.patch.object(
                inspection_index,
                "_serialize_file_chunk_manifest_files",
                side_effect=AssertionError(
                    "unchanged manifest files should reuse cached serialized files json"
                ),
            ):
                second_chunks, second_stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    excluded_paths={cache_dir},
                    repository_state_fingerprint="sha256:second",
                    return_diagnostics=True,
                )

        self.assertTrue(first_chunks)
        self.assertTrue(second_chunks)
        self.assertEqual(first_stats["total_files"], 1)
        self.assertEqual(second_stats["reused_files"], 1)
        self.assertEqual(second_stats["rebuilt_files"], 0)

    def test_build_syntax_chunks_reuses_cached_snapshot_metadata_json_when_only_repository_fingerprint_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            (root / "main.py").write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "main.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            inspection_index.build_syntax_chunks(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
                repository_state_fingerprint="sha256:first",
            )

            original_write_metadata = inspection_index._write_file_chunk_snapshot_metadata
            captured = {}

            def recording_write_metadata(path, **kwargs):
                captured["serialized_index_manifest_json"] = str(
                    kwargs.get("serialized_index_manifest_json") or ""
                )
                captured["serialized_semantic_document_signatures_json"] = str(
                    kwargs.get("serialized_semantic_document_signatures_json") or ""
                )
                captured["serialized_chunk_ids_json"] = str(
                    kwargs.get("serialized_chunk_ids_json") or ""
                )
                return original_write_metadata(path, **kwargs)

            with mock.patch.object(
                inspection_index,
                "_write_file_chunk_snapshot_metadata",
                side_effect=recording_write_metadata,
            ):
                chunks, stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    excluded_paths={cache_dir},
                    repository_state_fingerprint="sha256:second",
                    return_diagnostics=True,
                )

        self.assertTrue(chunks)
        self.assertEqual(stats["reused_files"], 1)
        self.assertTrue(captured["serialized_index_manifest_json"])
        self.assertTrue(captured["serialized_semantic_document_signatures_json"])
        self.assertTrue(captured["serialized_chunk_ids_json"])

    def test_partial_dirty_discover_source_files_reuses_cached_inventory_without_head_tree_probe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            dirty_file = root / "dirty.py"
            clean_file = root / "clean.py"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            dirty_file.write_text("def dirty():\n    return 1\n", encoding="utf-8")
            clean_file.write_text("def clean():\n    return 2\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            git_probe_cache = {}

            inspection_index.repository_fingerprint(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
                git_probe_cache=git_probe_cache,
            )
            first = inspection_index.discover_source_files(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
                git_probe_cache=git_probe_cache,
            )
            dirty_file.write_text("def dirty():\n    value = 3\n    return value\n", encoding="utf-8")
            inspection_index.repository_fingerprint(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
                git_probe_cache=git_probe_cache,
            )

            original_run = inspection_index.subprocess.run

            def guarded_run(args, *pargs, **kwargs):
                if git_args_match(args, root, "rev-parse") and "HEAD^{tree}" in args:
                    raise AssertionError("tracked content-only discovery refresh should reuse cached inventory without head tree probe")
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=guarded_run):
                second = inspection_index.discover_source_files(
                    discovered,
                    cache_dir=cache_dir,
                    excluded_paths={cache_dir},
                    git_probe_cache=git_probe_cache,
                    prefer_inventory_delta=True,
                )

        self.assertEqual(
            sorted((path.relative_to(root).as_posix() for _item, path, _rel in first)),
            sorted((path.relative_to(root).as_posix() for _item, path, _rel in second)),
        )

    def test_partial_dirty_build_skips_redundant_repo_status_prefetch_when_probe_cache_is_warm(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            dirty_file = root / "dirty.py"
            clean_file = root / "clean.py"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            dirty_file.write_text("def dirty():\n    return 1\n", encoding="utf-8")
            clean_file.write_text("def clean():\n    return 2\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            git_probe_cache = {}

            first_fp, _ = inspection_index.repository_fingerprint(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
                git_probe_cache=git_probe_cache,
            )
            inspection_index.build_syntax_chunks(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
                repository_state_fingerprint=first_fp,
                git_probe_cache=git_probe_cache,
            )
            dirty_file.write_text("def dirty():\n    value = 3\n    return value\n", encoding="utf-8")
            second_fp, _ = inspection_index.repository_fingerprint(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
                git_probe_cache=git_probe_cache,
            )

            original_scoped_status_snapshot = inspection_index._scoped_git_status_snapshot

            def guarded_scoped_status_snapshot(top, normalized_scope_paths, *args, **kwargs):
                if tuple(normalized_scope_paths or ()) in {tuple(), (".",)}:
                    raise AssertionError("repo status snapshot should be reused from git probe cache")
                return original_scoped_status_snapshot(top, normalized_scope_paths, *args, **kwargs)

            with mock.patch.object(
                inspection_index,
                "_scoped_git_status_snapshot",
                side_effect=guarded_scoped_status_snapshot,
            ):
                chunks, stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    excluded_paths={cache_dir},
                    repository_state_fingerprint=second_fp,
                    git_probe_cache=git_probe_cache,
                    return_diagnostics=True,
                )

        self.assertEqual(stats["rebuilt_files"], 1)
        self.assertEqual(stats["reused_files"], 1)
        self.assertTrue(len(chunks) > 0)

    def test_git_dirty_manifest_entry_keys_uses_dirty_only_subset_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            dirty_file = root / "dirty.py"
            clean_file = root / "clean.py"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            dirty_file.write_text("def dirty():\n    return 1\n", encoding="utf-8")
            clean_file.write_text("def clean():\n    return 2\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            git_probe_cache = {}

            dirty_file.write_text("def dirty():\n    value = 3\n    return value\n", encoding="utf-8")
            discovered_files = inspection_index.discover_source_files(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
                git_probe_cache=git_probe_cache,
                prefer_inventory_delta=True,
            )
            namespaces = {id(discovered[0]): "repo"}

            with mock.patch.object(
                inspection_index,
                "_status_subset_digest_and_dirty",
                side_effect=AssertionError("dirty manifest keys should not build subset digests"),
            ):
                dirty_keys = inspection_index._git_dirty_manifest_entry_keys(
                    discovered_files,
                    namespaces,
                    git_probe_cache=git_probe_cache,
                )

        self.assertTrue(dirty_keys)

    def test_partial_dirty_discover_source_files_reuses_process_cached_record_when_manifest_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            dirty_file = root / "dirty.py"
            clean_file = root / "clean.py"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            dirty_file.write_text("def dirty():\n    return 1\n", encoding="utf-8")
            clean_file.write_text("def clean():\n    return 2\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            git_probe_cache = {}

            inspection_index.repository_fingerprint(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
                git_probe_cache=git_probe_cache,
            )
            first = inspection_index.discover_source_files(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
                git_probe_cache=git_probe_cache,
            )
            inspection_index._discovery_manifest_path(cache_dir).unlink(missing_ok=True)
            dirty_file.write_text("def dirty():\n    value = 3\n    return value\n", encoding="utf-8")
            inspection_index.repository_fingerprint(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
                git_probe_cache=git_probe_cache,
            )

            original_run = inspection_index.subprocess.run

            def guarded_run(args, *pargs, **kwargs):
                if git_args_match(args, root, "ls-files"):
                    raise AssertionError("process-cached discovery record should avoid fresh git ls-files enumeration")
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=guarded_run):
                second = inspection_index.discover_source_files(
                    discovered,
                    cache_dir=cache_dir,
                    excluded_paths={cache_dir},
                    git_probe_cache=git_probe_cache,
                    prefer_inventory_delta=True,
                )

        self.assertEqual(
            sorted((path.relative_to(root).as_posix() for _item, path, _rel in first)),
            sorted((path.relative_to(root).as_posix() for _item, path, _rel in second)),
        )

    def test_discover_source_files_cached_git_record_skips_per_file_language_rechecks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            (root / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
            (root / "b.py").write_text("def b():\n    return 2\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            git_probe_cache = {}

            fingerprint, _ = inspection_index.repository_fingerprint(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
                git_probe_cache=git_probe_cache,
            )
            first = inspection_index.discover_source_files(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
                repository_state_fingerprint=fingerprint,
                git_probe_cache=git_probe_cache,
            )

            with mock.patch.object(
                inspection_index,
                "language_for_path",
                side_effect=AssertionError("cached git discovery record should not re-run per-file language checks"),
            ):
                second = inspection_index.discover_source_files(
                    discovered,
                    cache_dir=cache_dir,
                    excluded_paths={cache_dir},
                    repository_state_fingerprint=fingerprint,
                    git_probe_cache=git_probe_cache,
                )

        self.assertEqual(
            sorted((path.relative_to(root).as_posix() for _item, path, _rel in first)),
            sorted((path.relative_to(root).as_posix() for _item, path, _rel in second)),
        )

    def test_discover_source_files_exact_fingerprint_process_cache_skips_git_refresh(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            (root / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
            (root / "b.py").write_text("def b():\n    return 2\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            git_probe_cache = {}

            fingerprint, _ = inspection_index.repository_fingerprint(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
                git_probe_cache=git_probe_cache,
            )
            first = inspection_index.discover_source_files(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
                repository_state_fingerprint=fingerprint,
                git_probe_cache=git_probe_cache,
            )
            inspection_index._discovery_manifest_path(cache_dir).unlink(missing_ok=True)
            inspection_index._discovery_fingerprint_manifest_path(cache_dir, str(fingerprint)).unlink(missing_ok=True)

            with mock.patch.object(
                inspection_index,
                "_cached_git_source_files",
                side_effect=AssertionError("exact discovery process cache should bypass cached git refresh"),
            ), mock.patch.object(
                inspection_index,
                "_iter_source_candidates",
                side_effect=AssertionError("exact discovery process cache should bypass source re-enumeration"),
            ):
                second = inspection_index.discover_source_files(
                    discovered,
                    cache_dir=cache_dir,
                    excluded_paths={cache_dir},
                    repository_state_fingerprint=fingerprint,
                    git_probe_cache=git_probe_cache,
                )

        self.assertEqual(
            sorted((path.relative_to(root).as_posix() for _item, path, _rel in first)),
            sorted((path.relative_to(root).as_posix() for _item, path, _rel in second)),
        )

    def test_discover_source_files_uses_broker_touched_paths_hint_without_git_refresh(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            (root / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
            (root / "b.py").write_text("def b():\n    return 2\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            git_probe_cache = {}

            first_fp, _ = inspection_index.repository_fingerprint(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
                git_probe_cache=git_probe_cache,
            )
            inspection_index.discover_source_files(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
                repository_state_fingerprint=first_fp,
                git_probe_cache=git_probe_cache,
            )

            (root / "a.py").write_text("def a():\n    value = 2\n    return value\n", encoding="utf-8")
            second_fp, _ = inspection_index.repository_fingerprint(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
                git_probe_cache={},
            )

            with mock.patch.object(
                inspection_index,
                "_cached_scoped_status_snapshot",
                side_effect=AssertionError("touched-path hint should bypass scoped status refresh"),
            ), mock.patch.object(
                inspection_index,
                "_git_scope_inventory_identity",
                side_effect=AssertionError("touched-path hint should bypass scope inventory identity"),
            ), mock.patch.object(
                inspection_index,
                "_git_scope_status_paths",
                side_effect=AssertionError("touched-path hint should bypass status-path refresh"),
            ), mock.patch.object(
                inspection_index,
                "_scoped_git_inventory_delta_paths",
                side_effect=AssertionError("touched-path hint should bypass inventory-delta refresh"),
            ):
                second = inspection_index.discover_source_files(
                    discovered,
                    cache_dir=cache_dir,
                    excluded_paths={cache_dir},
                    repository_state_fingerprint=second_fp,
                    git_probe_cache={},
                    prefer_inventory_delta=True,
                    touched_paths_hint=("a.py",),
                )

        self.assertEqual(
            sorted(path.relative_to(root).as_posix() for _item, path, _rel in second),
            ["a.py", "b.py"],
        )

    def test_fingerprint_state_touched_paths_hint_extracts_unique_paths(self):
        self.assertEqual(
            inspection_pipeline._fingerprint_state_touched_paths_hint(
                [
                    {"kind": "git", "dirty_paths": ["a.py", "b.py", "a.py"]},
                    {"kind": "git", "dirty_paths": ["sub/c.py"]},
                    {"kind": "input_manifest", "fingerprint": "sha256:test"},
                ]
            ),
            ("a.py", "b.py", "sub/c.py"),
        )
        self.assertEqual(
            inspection_pipeline._fingerprint_state_touched_paths_hint([{"kind": "git"}]),
            (),
        )
        self.assertEqual(
            inspection_index._touched_paths_hint_from_repository_fingerprint_state(
                [{"kind": "git", "dirty_paths": ["b.py", "a.py", "./a.py", "b.py"]}]
            ),
            ("a.py", "b.py"),
        )

    def test_build_syntax_chunks_uses_repository_fingerprint_state_touched_paths_hint_without_git_refresh(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            (root / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
            (root / "b.py").write_text("def b():\n    return 2\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            git_probe_cache = {}

            first_fp, first_state = inspection_index.repository_fingerprint(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
                git_probe_cache=git_probe_cache,
            )
            inspection_index.build_syntax_chunks(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
                repository_state_fingerprint=first_fp,
                repository_fingerprint_state=first_state,
                git_probe_cache=git_probe_cache,
            )

            (root / "a.py").write_text("def a():\n    value = 2\n    return value\n", encoding="utf-8")
            second_fp, second_state = inspection_index.repository_fingerprint(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
                git_probe_cache={},
            )

            with mock.patch.object(
                inspection_index,
                "_cached_scoped_status_snapshot",
                side_effect=AssertionError("fingerprint-state touched paths should bypass scoped status refresh"),
            ), mock.patch.object(
                inspection_index,
                "_git_scope_inventory_identity",
                side_effect=AssertionError("fingerprint-state touched paths should bypass scope inventory identity"),
            ), mock.patch.object(
                inspection_index,
                "_git_scope_status_paths",
                side_effect=AssertionError("fingerprint-state touched paths should bypass status-path refresh"),
            ), mock.patch.object(
                inspection_index,
                "_scoped_git_inventory_delta_paths",
                side_effect=AssertionError("fingerprint-state touched paths should bypass inventory-delta refresh"),
            ):
                chunks, stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    excluded_paths={cache_dir},
                    repository_state_fingerprint=second_fp,
                    repository_fingerprint_state=second_state,
                    git_probe_cache={},
                    return_diagnostics=True,
                )

        self.assertEqual(
            sorted(chunk["path"] for chunk in chunks),
            ["a.py", "b.py"],
        )
        self.assertEqual(stats["rebuilt_files"], 1)

    def test_build_syntax_chunks_uses_clean_worktree_files_hint_without_discovery_refresh(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            (root / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
            (root / "b.py").write_text("def b():\n    return 2\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            repository_fp, repository_state = inspection_index.repository_fingerprint(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
            )

            with mock.patch.object(
                inspection_index,
                "discover_source_files",
                side_effect=AssertionError("clean worktree file hint should bypass discovery refresh"),
            ):
                chunks, stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    excluded_paths={cache_dir},
                    repository_state_fingerprint=repository_fp,
                    repository_fingerprint_state=repository_state,
                    return_diagnostics=True,
                )

        self.assertEqual(stats["total_files"], 2)
        self.assertEqual(stats["rebuilt_files"], 2)
        self.assertTrue(chunks)

    def test_build_syntax_chunks_clean_worktree_files_hint_supports_subdirectory_scope(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "target"
            sibling = root / "sibling"
            cache_dir = root / "cache"
            target.mkdir()
            sibling.mkdir()
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            (target / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
            (sibling / "b.py").write_text("def b():\n    return 2\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": target}]

            repository_fp, repository_state = inspection_index.repository_fingerprint(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
            )

            with mock.patch.object(
                inspection_index,
                "discover_source_files",
                side_effect=AssertionError("subdirectory clean worktree file hint should bypass discovery refresh"),
            ):
                chunks, stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    excluded_paths={cache_dir},
                    repository_state_fingerprint=repository_fp,
                    repository_fingerprint_state=repository_state,
                    return_diagnostics=True,
                )

        self.assertEqual(stats["total_files"], 1)
        self.assertEqual(stats["rebuilt_files"], 1)
        self.assertEqual(sorted({chunk["path"] for chunk in chunks}), ["a.py"])

    def test_nonsource_git_change_keeps_working_discovery_manifest_stable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            (root / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            git_probe_cache = {}

            first_fp, _ = inspection_index.repository_fingerprint(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
                git_probe_cache=git_probe_cache,
            )
            inspection_index.discover_source_files(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
                repository_state_fingerprint=first_fp,
                git_probe_cache=git_probe_cache,
            )
            working_manifest_path = inspection_index._discovery_manifest_path(cache_dir)
            first_manifest = inspection_index._load_discovery_working_manifest(working_manifest_path)

            nonsource = root / ".gitignore"
            nonsource.write_text("cache/\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", ".gitignore"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "nonsource"], check=True)
            second_fp, _ = inspection_index.repository_fingerprint(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
                git_probe_cache=git_probe_cache,
            )
            inspection_index.discover_source_files(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
                repository_state_fingerprint=second_fp,
                git_probe_cache=git_probe_cache,
                prefer_inventory_delta=True,
            )
            second_manifest = inspection_index._load_discovery_working_manifest(working_manifest_path)
            fingerprint_manifest = inspection_index._load_discovery_working_manifest(
                inspection_index._discovery_fingerprint_manifest_path(cache_dir, str(second_fp))
            )

        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(
            str((fingerprint_manifest or {}).get(str(root.resolve(strict=False)), {}).get("repository_state_fingerprint") or ""),
            str(second_fp),
        )

    def test_partial_dirty_without_snapshot_reuses_manifest_signatures_and_skips_git_signature_probe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            dirty_file = root / "dirty.py"
            clean_file = root / "clean.py"
            dirty_file.write_text("def dirty():\n    return 1\n", encoding="utf-8")
            clean_file.write_text("def clean():\n    return 2\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            first_fp, _ = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)
            inspection_index.build_syntax_chunks(
                discovered,
                cache_dir=cache_dir,
                repository_state_fingerprint=first_fp,
                return_diagnostics=True,
            )
            dirty_file.write_text("def dirty():\n    value = 3\n    return value\n", encoding="utf-8")
            second_fp, _ = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

            original_load_snapshot = inspection_index.load_cached_chunk_snapshot

            def no_snapshot(cache_dir_arg, *args, **kwargs):
                repository_fp = str(kwargs.get("repository_state_fingerprint") or "")
                if repository_fp:
                    return None
                return original_load_snapshot(cache_dir_arg, *args, **kwargs)

            with mock.patch.object(
                inspection_index,
                "load_cached_chunk_snapshot",
                side_effect=no_snapshot,
            ), mock.patch.object(
                inspection_index,
                "_git_file_signatures",
                side_effect=AssertionError("partial-dirty run should reuse previous manifest signatures without git signature probe"),
            ):
                chunks, stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    repository_state_fingerprint=second_fp,
                    return_diagnostics=True,
                )

        self.assertTrue(chunks)
        self.assertEqual(stats["total_files"], 2)
        self.assertEqual(stats["reused_files"], 1)
        self.assertGreaterEqual(stats["rebuilt_files"], 1)

    def test_build_syntax_chunks_writes_direct_shared_chunk_state_entries_without_loading_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            shared_cache_dir = root / "shared-cache"
            (root / "one.py").write_text("def one():\n    return 1\n", encoding="utf-8")
            (root / "two.py").write_text("def two():\n    return 2\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            original_load = inspection_index._load_shared_file_chunk_state_manifest
            original_flush = inspection_index._flush_shared_file_chunk_state_manifest_cache
            load_calls = 0
            flush_calls = 0

            def counted_load():
                nonlocal load_calls
                load_calls += 1
                return original_load()

            def counted_flush(manifest_cache):
                nonlocal flush_calls
                flush_calls += 1
                return original_flush(manifest_cache)

            with mock.patch.dict(
                os.environ,
                {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)},
                clear=False,
            ), mock.patch.object(
                inspection_index, "_load_shared_file_chunk_state_manifest", side_effect=counted_load
            ), mock.patch.object(
                inspection_index, "_flush_shared_file_chunk_state_manifest_cache", side_effect=counted_flush
            ):
                chunks, stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    return_diagnostics=True,
                )
                entry_dir = shared_cache_dir / "file-chunk-state-entries"
                entry_dir_exists = entry_dir.exists()
                has_entry_files = any(entry_dir.rglob("*.json")) if entry_dir_exists else False

        self.assertTrue(chunks)
        self.assertEqual(stats["rebuilt_files"], 2)
        self.assertEqual(load_calls, 0)
        self.assertEqual(flush_calls, 1)
        self.assertTrue(entry_dir_exists)
        self.assertTrue(has_entry_files)

    def test_build_syntax_chunks_writes_local_snapshot_metadata_even_with_shared_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            shared_cache_dir = root / "shared-cache"
            source = root / "worker.py"
            source.write_text("def worker():\n    return 1\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            with mock.patch.dict(
                os.environ,
                {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)},
                clear=False,
            ):
                repository_fp, _ = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)
                chunks, stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    repository_state_fingerprint=repository_fp,
                    return_diagnostics=True,
                )

            local_snapshot_path = inspection_index._file_chunk_snapshot_path(cache_dir)
            local_metadata_path = inspection_index._file_chunk_snapshot_metadata_path(cache_dir)
            metadata = inspection_index.load_cached_chunk_snapshot_metadata(
                discovered,
                cache_dir=cache_dir,
                excluded_paths={cache_dir},
                repository_state_fingerprint=repository_fp,
            ) or {}
            shared_snapshot_path = inspection_index._shared_file_chunk_snapshot_path(
                repository_fp,
                str(metadata.get("build_config_digest") or ""),
                create=False,
            )
            self.assertTrue(chunks)
            self.assertGreaterEqual(stats["rebuilt_files"], 1)
            self.assertTrue(local_snapshot_path.exists())
            self.assertTrue(local_metadata_path.exists())
            self.assertTrue(shared_snapshot_path is None or shared_snapshot_path.exists())

    def test_partial_dirty_build_keeps_transient_chunk_and_lexical_publication_local(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            shared_cache_dir = root / "shared-cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            alpha = root / "alpha.py"
            beta = root / "beta.py"
            alpha.write_text("def alpha():\n    return 'alpha'\n", encoding="utf-8")
            beta.write_text("def beta():\n    return 'beta'\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            git_probe_cache = {}

            with mock.patch.dict(
                os.environ,
                {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)},
                clear=False,
            ):
                clean_fp, _ = inspection_index.repository_fingerprint(
                    discovered,
                    cache_dir=cache_dir,
                    excluded_paths={cache_dir, shared_cache_dir},
                    git_probe_cache=git_probe_cache,
                )
                clean_chunks = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    excluded_paths={cache_dir, shared_cache_dir},
                    repository_state_fingerprint=clean_fp,
                    git_probe_cache=git_probe_cache,
                )
                build_config_digest = (
                    inspection_index._load_file_chunk_working_manifest(
                        inspection_index._file_chunk_manifest_path(cache_dir)
                    )
                    or {}
                ).get("build_config_digest", "")
                clean_index_fp = inspection_index.inspection_index_fingerprint(clean_fp, clean_chunks)
                inspection_index.ensure_lexical_index(
                    clean_chunks,
                    cache_dir,
                    clean_index_fp,
                    build_config_digest=build_config_digest,
                )

                shared_latest_manifest_path = inspection_index._shared_file_chunk_latest_manifest_path(
                    build_config_digest,
                    create=False,
                )
                shared_latest_snapshot_path = inspection_index._shared_file_chunk_snapshot_path(
                    clean_fp,
                    build_config_digest,
                    create=False,
                )
                shared_latest_lexical_manifest_path = inspection_index._shared_latest_lexical_manifest_path(
                    build_config_digest,
                    create=False,
                )
                latest_manifest_mtime = (
                    shared_latest_manifest_path.stat().st_mtime_ns if shared_latest_manifest_path and shared_latest_manifest_path.exists() else None
                )
                latest_lexical_manifest_mtime = (
                    shared_latest_lexical_manifest_path.stat().st_mtime_ns
                    if shared_latest_lexical_manifest_path and shared_latest_lexical_manifest_path.exists()
                    else None
                )
                self.assertTrue(shared_latest_snapshot_path is None or shared_latest_snapshot_path.exists())

                beta.write_text("def beta():\n    value = 'gamma'\n    return value\n", encoding="utf-8")
                dirty_fp, _ = inspection_index.repository_fingerprint(
                    discovered,
                    cache_dir=cache_dir,
                    excluded_paths={cache_dir, shared_cache_dir},
                    git_probe_cache=git_probe_cache,
                )
                dirty_chunks, dirty_stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    excluded_paths={cache_dir, shared_cache_dir},
                    repository_state_fingerprint=dirty_fp,
                    git_probe_cache=git_probe_cache,
                    return_diagnostics=True,
                )
                dirty_index_fp = inspection_index.inspection_index_fingerprint(dirty_fp, dirty_chunks)
                inspection_index.ensure_lexical_index(
                    dirty_chunks,
                    cache_dir,
                    dirty_index_fp,
                    build_config_digest=build_config_digest,
                )

                dirty_shared_manifest_path = inspection_index._shared_file_chunk_working_manifest_path(
                    dirty_fp,
                    build_config_digest,
                    create=False,
                )
                dirty_shared_snapshot_path = inspection_index._shared_file_chunk_snapshot_path(
                    dirty_fp,
                    build_config_digest,
                    create=False,
                )
                dirty_shared_snapshot_metadata_path = inspection_index._shared_file_chunk_snapshot_metadata_path(
                    dirty_fp,
                    build_config_digest,
                    create=False,
                )
                dirty_shared_lexical_index_path = inspection_index._shared_lexical_index_path(
                    dirty_index_fp,
                    create=False,
                )
                dirty_shared_lexical_manifest_path = inspection_index._shared_lexical_manifest_path(
                    dirty_index_fp,
                    create=False,
                )

        self.assertEqual(dirty_stats["rebuilt_files"], 1)
        self.assertTrue(getattr(dirty_chunks, "_skip_shared_cache_publication", False))
        self.assertTrue(dirty_shared_manifest_path is None or not dirty_shared_manifest_path.exists())
        self.assertTrue(dirty_shared_snapshot_path is None or not dirty_shared_snapshot_path.exists())
        self.assertTrue(dirty_shared_snapshot_metadata_path is None or not dirty_shared_snapshot_metadata_path.exists())
        self.assertTrue(dirty_shared_lexical_index_path is None or not dirty_shared_lexical_index_path.exists())
        self.assertTrue(dirty_shared_lexical_manifest_path is None or not dirty_shared_lexical_manifest_path.exists())
        if shared_latest_manifest_path is not None and shared_latest_manifest_path.exists() and latest_manifest_mtime is not None:
            self.assertEqual(shared_latest_manifest_path.stat().st_mtime_ns, latest_manifest_mtime)
        if (
            shared_latest_lexical_manifest_path is not None
            and shared_latest_lexical_manifest_path.exists()
            and latest_lexical_manifest_mtime is not None
        ):
            self.assertEqual(shared_latest_lexical_manifest_path.stat().st_mtime_ns, latest_lexical_manifest_mtime)

    def test_shared_chunk_state_lookup_falls_back_to_legacy_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            shared_cache_dir = root / "shared-cache"
            source = root / "worker.py"
            source.write_text("def worker():\n    return 'token'\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            with mock.patch.dict(
                os.environ,
                {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)},
                clear=False,
            ):
                chunks = inspection_index.build_syntax_chunks(discovered, cache_dir=cache_dir)
                manifest = dict(getattr(chunks, "_lexical_manifest", {}) or {})
                file_key = next(iter(manifest))
                record = manifest[file_key]
                config_key = next(
                    iter(
                        (
                            inspection_index._load_file_chunk_working_manifest(
                                inspection_index._file_chunk_manifest_path(cache_dir)
                            )
                            or {}
                        ).get("files", {})
                    )
                )
                working_manifest = inspection_index._load_file_chunk_working_manifest(
                    inspection_index._file_chunk_manifest_path(cache_dir)
                ) or {}
                manifest_entry = next(iter((working_manifest.get("files") or {}).values()))
                state_signature = str(manifest_entry.get("signature") or "")
                cache_key = str(manifest_entry.get("cache_key") or "")
                self.assertTrue(cache_key)
                entry_dir = shared_cache_dir / "file-chunk-state-entries"
                for path in entry_dir.rglob("*.json"):
                    path.unlink()
                legacy_path = shared_cache_dir / "file-chunk-state-manifest.json"
                legacy_path.parent.mkdir(parents=True, exist_ok=True)
                state_key = inspection_index._shared_file_chunk_state_key(
                    str(manifest_entry.get("config_key") or ""),
                    state_signature,
                )
                legacy_payload = {
                    "schema": inspection_index.SHARED_FILE_CHUNK_STATE_MANIFEST_SCHEMA,
                    "entries": {
                        state_key: {
                            "cache_key": cache_key,
                            "empty": False,
                        }
                    },
                }
                legacy_path.write_text(json.dumps(legacy_payload), encoding="utf-8")
                loaded = inspection_index._load_file_chunk_cache_by_state_signature(
                    cache_dir,
                    str(manifest_entry.get("config_key") or ""),
                    state_signature,
                )

        self.assertIsNotNone(loaded)
        loaded_cache_key, loaded_chunks = loaded
        self.assertEqual(loaded_cache_key, cache_key)
        self.assertTrue(loaded_chunks)

    def test_shared_chunk_state_lookup_reads_legacy_entry_without_sharded_write_through(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            shared_cache_dir = root / "shared-cache"
            source = root / "worker.py"
            source.write_text("def worker():\n    return 'token'\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            with mock.patch.dict(
                os.environ,
                {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)},
                clear=False,
            ):
                chunks = inspection_index.build_syntax_chunks(discovered, cache_dir=cache_dir)
                working_manifest = inspection_index._load_file_chunk_working_manifest(
                    inspection_index._file_chunk_manifest_path(cache_dir)
                ) or {}
                manifest_entry = next(iter((working_manifest.get("files") or {}).values()))
                state_signature = str(manifest_entry.get("signature") or "")
                cache_key = str(manifest_entry.get("cache_key") or "")
                config_key = str(manifest_entry.get("config_key") or "")
                state_key = inspection_index._shared_file_chunk_state_key(config_key, state_signature)
                sharded_path = inspection_index._shared_file_chunk_state_entry_path(state_key, create=False)
                legacy_path = inspection_index._legacy_shared_file_chunk_state_entry_path(state_key, create=True)
                payload = json.loads(sharded_path.read_text(encoding="utf-8"))
                sharded_path.unlink()
                legacy_path.write_text(json.dumps(payload), encoding="utf-8")

                loaded = inspection_index._load_file_chunk_cache_by_state_signature(
                    cache_dir,
                    config_key,
                    state_signature,
                )
                promoted_exists = sharded_path.exists()

        self.assertTrue(chunks)
        self.assertIsNotNone(loaded)
        loaded_cache_key, loaded_chunks = loaded
        self.assertEqual(loaded_cache_key, cache_key)
        self.assertTrue(loaded_chunks)
        self.assertFalse(promoted_exists)

    def test_shared_chunk_state_entry_reuses_in_process_memory_cache_on_second_hit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            shared_cache_dir = root / "shared-cache"
            state_key = "sha256:state-entry"

            with mock.patch.dict(
                os.environ,
                {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)},
                clear=False,
            ):
                path = inspection_index._shared_file_chunk_state_entry_path(state_key, create=True)
                path.write_text(
                    json.dumps(
                        {
                            "schema": inspection_index.SHARED_FILE_CHUNK_STATE_MANIFEST_SCHEMA,
                            "cache_key": "sha256:cache",
                            "empty": False,
                        }
                    ),
                    encoding="utf-8",
                )
                first = inspection_index._load_shared_file_chunk_state_entry(state_key)
                with mock.patch.object(Path, "read_text", side_effect=AssertionError("expected shared state entry memory cache hit")):
                    second = inspection_index._load_shared_file_chunk_state_entry(state_key)

        self.assertEqual(first, {"cache_key": "sha256:cache", "empty": False})
        self.assertEqual(second, first)

    def test_shared_chunk_state_lookup_caches_miss_within_manifest_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            manifest_cache = inspection_index._shared_file_chunk_state_manifest_cache()
            config_key = "sha256:config"
            state_signature = "sha256:missing"

            with mock.patch.object(
                inspection_index,
                "_load_shared_file_chunk_state_entry",
                return_value=None,
            ) as load_entry:
                first = inspection_index._load_file_chunk_cache_by_state_signature(
                    cache_dir,
                    config_key,
                    state_signature,
                    manifest_cache=manifest_cache,
                )
                second = inspection_index._load_file_chunk_cache_by_state_signature(
                    cache_dir,
                    config_key,
                    state_signature,
                    manifest_cache=manifest_cache,
                )

        self.assertIsNone(first)
        self.assertIsNone(second)
        self.assertEqual(load_entry.call_count, 1)

    def test_seeded_shared_chunk_state_manifest_cache_avoids_state_entry_read(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            shared_cache_dir = root / "shared-cache"
            source = root / "worker.py"
            source.write_text("def worker():\n    return 'token'\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            with mock.patch.dict(
                os.environ,
                {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)},
                clear=False,
            ):
                chunks = inspection_index.build_syntax_chunks(discovered, cache_dir=cache_dir)
                self.assertTrue(chunks)
                working_manifest = inspection_index._load_file_chunk_working_manifest(
                    inspection_index._file_chunk_manifest_path(cache_dir)
                ) or {}
                manifest_entry = next(iter((working_manifest.get("files") or {}).values()))
                config_key = str(manifest_entry.get("config_key") or "")
                state_signature = str(manifest_entry.get("signature") or "")
                cache_key = str(manifest_entry.get("cache_key") or "")
                manifest_cache = inspection_index._shared_file_chunk_state_manifest_cache()
                inspection_index._seed_shared_state_manifest_cache_from_working_manifest(
                    manifest_cache,
                    working_manifest.get("files") or {},
                )

                with mock.patch.object(
                    inspection_index,
                    "_load_shared_file_chunk_state_entry",
                    side_effect=AssertionError("expected seeded shared state manifest cache hit"),
                ):
                    loaded = inspection_index._load_file_chunk_cache_by_state_signature(
                        cache_dir,
                        config_key,
                        state_signature,
                        manifest_cache=manifest_cache,
                    )

        self.assertIsNotNone(loaded)
        loaded_cache_key, loaded_chunks = loaded
        self.assertEqual(loaded_cache_key, cache_key)
        self.assertTrue(loaded_chunks)

    def test_partial_dirty_direct_bundle_hit_skips_shared_state_lookup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            target_cache_dir = temp_root / "cache-target"
            scratch_cache_dir = temp_root / "cache-scratch"
            shared_cache_dir = temp_root / "shared-cache"
            source = root / "worker.py"
            source.write_text("def worker():\n    return 'v1'\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            subprocess.run(["git", "-C", str(root), "add", "worker.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "initial"], check=True)

            with mock.patch.dict(
                os.environ,
                {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)},
                clear=False,
            ):
                first_fp, _ = inspection_index.repository_fingerprint(discovered, cache_dir=target_cache_dir)
                first_chunks, first_stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=target_cache_dir,
                    repository_state_fingerprint=first_fp,
                    return_diagnostics=True,
                )
                self.assertTrue(first_chunks)
                self.assertGreaterEqual(first_stats["rebuilt_files"], 1)

                source.write_text("def worker():\n    return 'v2'\n", encoding="utf-8")
                dirty_fp, _ = inspection_index.repository_fingerprint(discovered, cache_dir=target_cache_dir)

                scratch_chunks, scratch_stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=scratch_cache_dir,
                    repository_state_fingerprint=dirty_fp,
                    return_diagnostics=True,
                )
                self.assertTrue(scratch_chunks)
                self.assertGreaterEqual(scratch_stats["rebuilt_files"], 1)

                with mock.patch.object(
                    inspection_index,
                    "_load_file_chunk_cache_by_state_signature",
                    side_effect=AssertionError("expected direct dirty bundle hit before shared state lookup"),
                ):
                    second_chunks, second_stats = inspection_index.build_syntax_chunks(
                        discovered,
                        cache_dir=target_cache_dir,
                        repository_state_fingerprint=dirty_fp,
                        return_diagnostics=True,
                    )

        self.assertTrue(second_chunks)
        self.assertEqual(second_stats["reused_files"], second_stats["total_files"])
        self.assertEqual(second_stats["rebuilt_files"], 0)

    def test_shared_git_probe_cache_avoids_duplicate_status_calls_within_one_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            (root / "worker.py").write_text("def worker():\n    return 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "worker.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            git_probe_cache = {}

            original_run = inspection_index.subprocess.run
            status_calls = []

            def counting_run(args, *pargs, **kwargs):
                if git_args_match(args, root, "status"):
                    start = 4 if args[3] == "--no-optional-locks" else 3
                    status_calls.append(tuple(args[start + 1 :]))
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=counting_run):
                inspection_index.repository_fingerprint(
                    discovered,
                    excluded_paths={cache_dir},
                    cache_dir=cache_dir,
                    git_probe_cache=git_probe_cache,
                )
                inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    excluded_paths={cache_dir},
                    git_probe_cache=git_probe_cache,
                    return_diagnostics=True,
                )

        repo_scope_status = [call for call in status_calls if "--porcelain=v1" in call and "--untracked-files=all" in call]
        self.assertEqual(len(repo_scope_status), 1)
        self.assertEqual(len(status_calls), 1)

    def test_shared_git_probe_cache_reuses_parsed_status_between_fingerprint_and_build(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "worker.py"
            source.write_text("def worker():\n    return 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "worker.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            source.write_text("def worker():\n    value = 2\n    return value\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            git_probe_cache = {}

            original_parse = inspection_index._parse_git_status_entries
            parse_calls = 0

            def counted_parse(status_output):
                nonlocal parse_calls
                parse_calls += 1
                return original_parse(status_output)

            with mock.patch.object(inspection_index, "_parse_git_status_entries", side_effect=counted_parse):
                repository_fp, _ = inspection_index.repository_fingerprint(
                    discovered,
                    excluded_paths={cache_dir},
                    cache_dir=cache_dir,
                    git_probe_cache=git_probe_cache,
                )
                chunks, stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    excluded_paths={cache_dir},
                    repository_state_fingerprint=repository_fp,
                    git_probe_cache=git_probe_cache,
                    return_diagnostics=True,
                )

        self.assertTrue(chunks)
        self.assertEqual(stats["total_files"], 1)
        self.assertEqual(parse_calls, 1)

    def test_shared_probe_cache_avoids_duplicate_non_git_source_discovery_within_one_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            (root / "alpha.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
            (root / "beta.py").write_text("def beta():\n    return 2\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            git_probe_cache = {}

            repository_fp, _ = inspection_index.repository_fingerprint(
                discovered,
                excluded_paths={cache_dir},
                cache_dir=cache_dir,
                git_probe_cache=git_probe_cache,
            )

            with mock.patch.object(
                inspection_index,
                "_iter_source_candidates",
                side_effect=AssertionError("shared probe cache should avoid duplicate non-git discovery"),
            ):
                chunks, stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    excluded_paths={cache_dir},
                    repository_state_fingerprint=repository_fp,
                    git_probe_cache=git_probe_cache,
                    return_diagnostics=True,
                )

        self.assertEqual(stats, {"total_files": 2, "reused_files": 0, "rebuilt_files": 2, "snapshot_cache_hit": False})
        self.assertEqual(len(chunks), 2)

    def test_file_chunk_cache_hot_run_does_not_reopen_metadata_unchanged_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            source = root / "worker.py"
            source.write_text("def worker():\n    return 1\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            inspection_index.build_syntax_chunks(discovered, cache_dir=cache_dir, return_diagnostics=True)

            original_read_text = Path.read_text

            def guarded_read_text(path_self, *args, **kwargs):
                if path_self == source:
                    raise AssertionError("metadata-unchanged file should not be reopened on hot run")
                return original_read_text(path_self, *args, **kwargs)

            with mock.patch.object(Path, "read_text", autospec=True, side_effect=guarded_read_text):
                chunks, stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    return_diagnostics=True,
                )

        self.assertTrue(chunks)
        self.assertEqual(stats, {"total_files": 1, "reused_files": 1, "rebuilt_files": 0, "snapshot_cache_hit": False})

    def test_file_chunk_cache_reuses_shared_chunk_cache_across_local_cache_roots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            shared_cache_dir = temp_root / "shared-cache"
            first_cache_dir = temp_root / "cache-one"
            second_cache_dir = temp_root / "cache-two"
            source = root / "worker.py"
            source.write_text("def worker():\n    return 'token'\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            fingerprint = "sha256:shared"

            with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
                first_chunks, first_stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=first_cache_dir,
                    repository_state_fingerprint=fingerprint,
                    return_diagnostics=True,
                )

                original_read_text = Path.read_text

                def guarded_read_text(path_self, *args, **kwargs):
                    if path_self == source:
                        raise AssertionError("shared chunk cache should avoid reopening the source file")
                    return original_read_text(path_self, *args, **kwargs)

                with mock.patch.object(Path, "read_text", autospec=True, side_effect=guarded_read_text):
                    with mock.patch.object(
                        inspection_index,
                        "discover_source_files",
                        side_effect=AssertionError("shared snapshot restore should avoid rediscovery"),
                    ):
                        second_chunks, second_stats = inspection_index.build_syntax_chunks(
                            discovered,
                            cache_dir=second_cache_dir,
                            repository_state_fingerprint=fingerprint,
                            return_diagnostics=True,
                        )
        self.assertEqual(first_chunks, second_chunks)
        self.assertEqual(first_stats["rebuilt_files"], 1)
        self.assertEqual(second_stats["reused_files"], 1)
        self.assertEqual(second_stats["rebuilt_files"], 0)
        self.assertTrue(second_stats["snapshot_cache_hit"])

    def test_load_file_chunk_cache_bundle_restores_shared_bundle_without_local_write_through(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            shared_cache_dir = temp_root / "shared-cache"
            first_cache_dir = temp_root / "cache-one"
            second_cache_dir = temp_root / "cache-two"
            source = root / "worker.py"
            source.write_text("def worker():\n    return 'token'\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
                inspection_index.build_syntax_chunks(discovered, cache_dir=first_cache_dir, return_diagnostics=True)
                first_manifest = inspection_index._load_file_chunk_working_manifest(
                    inspection_index._file_chunk_manifest_path(first_cache_dir)
                ) or {}
                first_entry = next(iter((first_manifest.get("files") or {}).values()))
                cache_key = str(first_entry.get("cache_key") or "")
                local_chunk_cache_path = inspection_index._file_chunk_cache_path(second_cache_dir, cache_key)
                with mock.patch.object(
                    inspection_index.json,
                    "dump",
                    side_effect=AssertionError("shared chunk bundle restore should not reserialize the shared cache file"),
                ):
                    bundle = inspection_index._load_file_chunk_cache_bundle(second_cache_dir, cache_key)
                    local_exists = local_chunk_cache_path.exists()

        self.assertIsNotNone(bundle)
        self.assertFalse(local_exists)

    def test_load_file_chunk_cache_payload_reuses_in_process_memory_cache_on_second_hit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            cache_dir = temp_root / "cache"
            source = root / "worker.py"
            source.write_text("def worker():\n    return 'token'\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            chunks = inspection_index.build_syntax_chunks(discovered, cache_dir=cache_dir)
            self.assertTrue(chunks)
            working_manifest = inspection_index._load_file_chunk_working_manifest(
                inspection_index._file_chunk_manifest_path(cache_dir)
            ) or {}
            manifest_entry = next(
                iter(record for record in (working_manifest.get("files") or {}).values() if not bool(record.get("empty")))
            )
            cache_key = str(manifest_entry.get("cache_key") or "")
            payload_path = inspection_index._file_chunk_cache_path(cache_dir, cache_key)

            first = inspection_index._load_file_chunk_cache_payload(payload_path)
            with mock.patch.object(Path, "read_text", side_effect=AssertionError("expected file chunk payload memory cache hit")):
                second = inspection_index._load_file_chunk_cache_payload(payload_path)

        self.assertIsNotNone(first)
        self.assertEqual(second, first)

    def test_load_file_chunk_working_manifest_reuses_in_process_cache_on_second_hit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            manifest_path = temp_root / "manifest.json"
            manifest_payload = {
                "schema": inspection_index.FILE_CHUNK_WORKING_MANIFEST_SCHEMA,
                "repository_state_fingerprint": "sha256:repo",
                "build_config_digest": "sha256:build",
                "files": {
                    "repo/worker.py": {
                        "signature": "git:abc",
                        "cache_key": "sha256:cache",
                        "config_key": "sha256:config",
                        "empty": False,
                    }
                },
            }
            manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
            inspection_index._FILE_CHUNK_WORKING_MANIFEST_CACHE.clear()

            first = inspection_index._load_file_chunk_working_manifest(manifest_path)
            original_read_text = Path.read_text

            def guarded_read_text(path_self, *args, **kwargs):
                if path_self == manifest_path:
                    raise AssertionError("second working manifest load should reuse in-process cache")
                return original_read_text(path_self, *args, **kwargs)

            with mock.patch.object(Path, "read_text", autospec=True, side_effect=guarded_read_text):
                second = inspection_index._load_file_chunk_working_manifest(manifest_path)

        self.assertEqual(first, second)

    def test_load_lexical_working_manifest_reuses_in_process_cache_on_second_hit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            manifest_path = temp_root / "lexical-working-manifest.json"
            manifest_payload = {
                "schema": inspection_index.LEXICAL_WORKING_MANIFEST_SCHEMA,
                "fingerprint": "sha256:fingerprint",
                "chunk_count": 1,
                "build_config_digest": "sha256:build",
                "files": {
                    "repo/worker.py": {
                        "file_key": "repo/worker.py",
                        "path": "worker.py",
                        "repository_path": "worker.py",
                        "source_namespace": "repo",
                        "signature": "sha256:sig",
                        "chunks": [
                            {
                                "chunk_id": "chunk-1",
                                "symbol": "worker",
                                "line_start": 1,
                                "line_end": 1,
                                "content_hash": "sha256:content",
                                "token_estimate": 1,
                            }
                        ],
                    }
                },
            }
            manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
            inspection_index._LEXICAL_WORKING_MANIFEST_CACHE.clear()

            first = inspection_index._load_lexical_working_manifest(manifest_path)
            original_read_text = Path.read_text

            def guarded_read_text(path_self, *args, **kwargs):
                if path_self == manifest_path:
                    raise AssertionError("second lexical manifest load should reuse in-process cache")
                return original_read_text(path_self, *args, **kwargs)

            with mock.patch.object(Path, "read_text", autospec=True, side_effect=guarded_read_text):
                second = inspection_index._load_lexical_working_manifest(manifest_path)

        self.assertEqual(first, second)

    def test_lexical_index_reuses_shared_working_index_across_local_cache_roots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            shared_cache_dir = temp_root / "shared-cache"
            first_cache_dir = temp_root / "cache-one"
            second_cache_dir = temp_root / "cache-two"
            source = root / "worker.py"
            source.write_text("def worker():\n    return 'token'\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
                chunks = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=first_cache_dir,
                    repository_state_fingerprint="sha256:shared",
                )
                fingerprint = inspection_index.inspection_index_fingerprint("sha256:shared", chunks)
                first_path, first_hit, first_stats = inspection_index.ensure_lexical_index(chunks, first_cache_dir, fingerprint)

                with mock.patch.object(
                    inspection_index,
                    "_rebuild_working_lexical_index",
                    side_effect=AssertionError("shared lexical index should avoid rebuilding on fresh local cache"),
                ):
                    second_path, second_hit, second_stats = inspection_index.ensure_lexical_index(
                        chunks, second_cache_dir, fingerprint
                    )

        self.assertFalse(first_hit)
        self.assertEqual(first_path, first_cache_dir / "lexical-working.sqlite3")
        self.assertEqual(first_stats["inserted_chunks"], len(chunks))
        self.assertTrue(second_hit)
        self.assertEqual(second_path, second_cache_dir / "lexical-working.sqlite3")
        self.assertTrue(second_stats["working_cache_hit"])
        self.assertTrue(second_stats["shared_restore"])

    def test_lexical_helper_reuses_shared_copy_across_local_cache_roots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            shared_cache_dir = temp_root / "shared-cache"
            first_cache_dir = temp_root / "cache-one"
            second_cache_dir = temp_root / "cache-two"
            source = root / "worker.py"
            source.write_text("def worker():\n    return 'token'\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
                chunks = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=first_cache_dir,
                    repository_state_fingerprint="sha256:shared",
                )
                fingerprint = inspection_index.inspection_index_fingerprint("sha256:shared", chunks)
                first_path, _, _ = inspection_index.ensure_lexical_index(chunks, first_cache_dir, fingerprint)
                first_ranked = inspection_index.lexical_search(first_path, "worker", chunks)
                self.assertTrue(first_ranked)

                second_path, _, second_stats = inspection_index.ensure_lexical_index(chunks, second_cache_dir, fingerprint)
                self.assertTrue(second_stats["shared_restore"])
                self.assertFalse((second_cache_dir / "lexical-helper.pkl").exists())
                inspection_index._LEXICAL_HELPER_CACHE.clear()
                inspection_index._LEXICAL_RESULT_CACHE.clear()
                with (
                    mock.patch.object(
                        inspection_index,
                        "_build_lexical_helper",
                        side_effect=AssertionError("shared lexical helper should avoid rebuilding from chunks"),
                    ),
                    mock.patch.object(
                        inspection_index,
                        "_build_lexical_helper_from_index",
                        side_effect=AssertionError("shared lexical helper should avoid rebuilding from index"),
                    ),
                ):
                    second_ranked = inspection_index.lexical_search(second_path, "worker", chunks)

        self.assertEqual(first_ranked, second_ranked)

    def test_lexical_index_shared_restore_uses_manifest_match_without_opening_shared_sqlite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            shared_cache_dir = temp_root / "shared-cache"
            first_cache_dir = temp_root / "cache-one"
            second_cache_dir = temp_root / "cache-two"
            source = root / "worker.py"
            source.write_text("def worker():\n    return 'token'\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
                chunks = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=first_cache_dir,
                    repository_state_fingerprint="sha256:shared",
                )
                fingerprint = inspection_index.inspection_index_fingerprint("sha256:shared", chunks)
                inspection_index.ensure_lexical_index(chunks, first_cache_dir, fingerprint)

                with mock.patch.object(
                    inspection_index,
                    "_lexical_index_path_is_current",
                    side_effect=AssertionError("exact-fingerprint shared lexical restore should trust matching manifest metadata"),
                ):
                    second_path, second_hit, second_stats = inspection_index.ensure_lexical_index(
                        chunks, second_cache_dir, fingerprint
                    )

        self.assertTrue(second_hit)
        self.assertEqual(second_path, second_cache_dir / "lexical-working.sqlite3")
        self.assertTrue(second_stats["shared_restore"])

    def test_load_chunks_from_lexical_manifest_uses_persisted_language_without_path_probe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            cache_dir = temp_root / "cache"
            source = root / "worker.py"
            source.write_text("def worker():\n    return 'token'\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            chunks = inspection_index.build_syntax_chunks(
                discovered,
                cache_dir=cache_dir,
                repository_state_fingerprint="sha256:shared",
            )
            fingerprint = inspection_index.inspection_index_fingerprint("sha256:shared", chunks)
            inspection_index.ensure_lexical_index(chunks, cache_dir, fingerprint)
            index_path = cache_dir / "lexical-working.sqlite3"

            with mock.patch.object(
                inspection_index,
                "language_for_path",
                side_effect=AssertionError("manifest reload should use persisted language"),
            ):
                reloaded = inspection_index.load_chunks_from_lexical_manifest(index_path, fingerprint)

        self.assertTrue(reloaded)
        self.assertEqual(len(reloaded), len(chunks))
        self.assertTrue(all(chunk.get("language") == "python" for chunk in reloaded))

    def test_lexical_index_reuses_shared_latest_copy_for_partial_dirty_fresh_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            shared_cache_dir = temp_root / "shared-cache"
            first_cache_dir = temp_root / "cache-one"
            second_cache_dir = temp_root / "cache-two"
            build_config_digest = "sha256:scope"
            alpha = root / "alpha.py"
            beta = root / "beta.py"
            alpha.write_text("def alpha():\n    return 'alpha'\n", encoding="utf-8")
            beta.write_text("def beta():\n    return 'beta'\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
                first_chunks = inspection_index.build_syntax_chunks(discovered, cache_dir=first_cache_dir)
                first_fp = inspection_index.inspection_index_fingerprint("sha256:first", first_chunks)
                inspection_index.ensure_lexical_index(
                    first_chunks,
                    first_cache_dir,
                    first_fp,
                    build_config_digest=build_config_digest,
                )

                beta.write_text("def beta():\n    return 'gamma'\n", encoding="utf-8")
                second_chunks = inspection_index.build_syntax_chunks(discovered, cache_dir=second_cache_dir)
                second_fp = inspection_index.inspection_index_fingerprint("sha256:second", second_chunks)
                beta_chunks = [chunk for chunk in second_chunks if chunk["repository_path"] == "beta.py"]

                with mock.patch.object(
                    inspection_index,
                    "_rebuild_working_lexical_index",
                    side_effect=AssertionError("shared latest lexical index should avoid full rebuild on partial dirty fresh cache"),
                ):
                    second_path, second_hit, second_stats = inspection_index.ensure_lexical_index(
                        second_chunks,
                        second_cache_dir,
                        second_fp,
                        build_config_digest=build_config_digest,
                    )

        self.assertFalse(second_hit)
        self.assertEqual(second_path, second_cache_dir / "lexical-working.sqlite3")
        self.assertTrue(second_stats["shared_restore"])
        self.assertEqual(second_stats["shared_restore_source"], "latest_build_config")
        self.assertEqual(second_stats["updated_files"], 1)
        self.assertEqual(second_stats["removed_files"], 0)
        self.assertEqual(second_stats["inserted_chunks"], len(beta_chunks))

    def test_lexical_index_shared_latest_partial_dirty_ignores_absolute_repo_root_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            shared_cache_dir = temp_root / "shared-cache"
            first_root = temp_root / "repo-one"
            second_root = temp_root / "repo-two"
            first_root.mkdir()
            second_root.mkdir()
            first_cache_dir = temp_root / "cache-one"
            second_cache_dir = temp_root / "cache-two"
            build_config_digest = "sha256:scope"

            (first_root / "alpha.py").write_text("def alpha():\n    return 'alpha'\n", encoding="utf-8")
            (first_root / "beta.py").write_text("def beta():\n    return 'beta'\n", encoding="utf-8")
            (second_root / "alpha.py").write_text("def alpha():\n    return 'alpha'\n", encoding="utf-8")
            (second_root / "beta.py").write_text("def beta():\n    return 'gamma'\n", encoding="utf-8")

            first_discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": first_root}]
            second_discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": second_root}]

            with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
                first_chunks = inspection_index.build_syntax_chunks(first_discovered, cache_dir=first_cache_dir)
                first_fp = inspection_index.inspection_index_fingerprint("sha256:first", first_chunks)
                inspection_index.ensure_lexical_index(
                    first_chunks,
                    first_cache_dir,
                    first_fp,
                    build_config_digest=build_config_digest,
                )

                second_chunks = inspection_index.build_syntax_chunks(second_discovered, cache_dir=second_cache_dir)
                second_fp = inspection_index.inspection_index_fingerprint("sha256:second", second_chunks)
                beta_chunks = [chunk for chunk in second_chunks if chunk["repository_path"] == "beta.py"]

                with mock.patch.object(
                    inspection_index,
                    "_rebuild_working_lexical_index",
                    side_effect=AssertionError("shared latest lexical index should still delta-update when repo root path changes"),
                ):
                    second_path, second_hit, second_stats = inspection_index.ensure_lexical_index(
                        second_chunks,
                        second_cache_dir,
                        second_fp,
                        build_config_digest=build_config_digest,
                    )

        self.assertFalse(second_hit)
        self.assertEqual(second_path, second_cache_dir / "lexical-working.sqlite3")
        self.assertTrue(second_stats["shared_restore"])
        self.assertEqual(second_stats["shared_restore_source"], "latest_build_config")
        self.assertEqual(second_stats["updated_files"], 1)
        self.assertEqual(second_stats["removed_files"], 0)
        self.assertEqual(second_stats["inserted_chunks"], len(beta_chunks))

    def test_partial_dirty_local_only_lexical_update_skips_shared_latest_restore_probe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            shared_cache_dir = temp_root / "shared-cache"
            cache_dir = temp_root / "cache"
            alpha = root / "alpha.py"
            beta = root / "beta.py"
            alpha.write_text("def alpha():\n    return 'alpha'\n", encoding="utf-8")
            beta.write_text("def beta():\n    return 'beta'\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
                first_chunks = inspection_index.build_syntax_chunks(discovered, cache_dir=cache_dir)
                first_fp = inspection_index.inspection_index_fingerprint("sha256:first", first_chunks)
                inspection_index.ensure_lexical_index(
                    first_chunks,
                    cache_dir,
                    first_fp,
                    build_config_digest="sha256:scope",
                )

                beta.write_text("def beta():\n    return 'gamma'\n", encoding="utf-8")
                second_chunks = inspection_index.build_syntax_chunks(discovered, cache_dir=cache_dir)
                second_chunks._skip_shared_cache_publication = True
                second_fp = inspection_index.inspection_index_fingerprint("sha256:second", second_chunks)

                with mock.patch.object(
                    inspection_index,
                    "_restore_matching_shared_latest_lexical_index",
                    side_effect=AssertionError("local-only dirty lexical updates should not probe shared latest restore"),
                ):
                    second_path, second_hit, second_stats = inspection_index.ensure_lexical_index(
                        second_chunks,
                        cache_dir,
                        second_fp,
                        build_config_digest="sha256:scope",
                    )

        self.assertFalse(second_hit)
        self.assertEqual(second_path, cache_dir / "lexical-working.sqlite3")
        self.assertFalse(second_stats.get("shared_restore"))
        self.assertEqual(second_stats["updated_files"], 1)

    def test_symbol_marker_cache_reuses_markers_across_cache_roots_when_chunk_config_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            shared_cache_dir = temp_root / "shared-cache"
            first_cache_dir = temp_root / "cache-one"
            second_cache_dir = temp_root / "cache-two"
            source = root / "worker.py"
            source.write_text("def worker():\n    return 'token'\n\nclass Example:\n    def method(self):\n        return 'x'\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
                inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=first_cache_dir,
                    repository_state_fingerprint="sha256:shared",
                    max_lines=120,
                    overlap=12,
                )

                with mock.patch.object(
                    inspection_index,
                    "symbol_markers",
                    side_effect=AssertionError("shared symbol marker cache should avoid recomputing markers"),
                ):
                    second_chunks, second_stats = inspection_index.build_syntax_chunks(
                        discovered,
                        cache_dir=second_cache_dir,
                        repository_state_fingerprint="sha256:shared",
                        max_lines=40,
                        overlap=4,
                        return_diagnostics=True,
                    )

        self.assertTrue(second_chunks)
        self.assertEqual(second_stats["reused_files"], 0)
        self.assertEqual(second_stats["rebuilt_files"], 1)

    def test_partial_dirty_local_only_rebuild_skips_shared_symbol_marker_publication(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            shared_cache_dir = temp_root / "shared-cache"
            cache_dir = temp_root / "cache"
            source = root / "worker.py"
            source.write_text("def worker():\n    return 'alpha'\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
                first_fp = "sha256:first"
                first_chunks, first_stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    repository_state_fingerprint=first_fp,
                    return_diagnostics=True,
                )
                self.assertEqual(first_stats["rebuilt_files"], 1)

                source.write_text("def worker():\n    return 'beta'\n", encoding="utf-8")
                dirty_fp = "sha256:dirty"
                dirty_chunks, dirty_stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    repository_state_fingerprint=dirty_fp,
                    touched_paths_hint=("worker.py",),
                    return_diagnostics=True,
                )

                dirty_text_hash = inspection_index.sha256_text(source.read_text(encoding="utf-8"))
                dirty_cache_key = inspection_index._symbol_marker_cache_key("python", dirty_text_hash)
                dirty_local_symbol_path = inspection_index._symbol_marker_cache_path(cache_dir, dirty_cache_key)
                dirty_shared_symbol_path = inspection_index._shared_symbol_marker_cache_path(dirty_cache_key, create=False)

        self.assertEqual(dirty_stats["rebuilt_files"], 1)
        self.assertTrue(getattr(dirty_chunks, "_skip_shared_cache_publication", False))
        self.assertFalse(dirty_local_symbol_path.exists())
        self.assertTrue(dirty_shared_symbol_path is None or not dirty_shared_symbol_path.exists())

    def test_file_chunk_snapshot_reuses_cached_chunks_without_rediscovery_when_repo_fingerprint_unchanged(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            alpha = root / "alpha.py"
            beta = root / "beta.py"
            alpha.write_text("def alpha():\n    return 1\n", encoding="utf-8")
            beta.write_text("def beta():\n    return 2\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            repository_fp, _ = inspection_index.repository_fingerprint(
                discovered,
                excluded_paths=[cache_dir],
                cache_dir=cache_dir,
            )
            first_chunks, first_stats = inspection_index.build_syntax_chunks(
                discovered,
                cache_dir=cache_dir,
                excluded_paths=[cache_dir],
                repository_state_fingerprint=repository_fp,
                return_diagnostics=True,
            )

            self.assertEqual(
                first_stats,
                {"total_files": 2, "reused_files": 0, "rebuilt_files": 2, "snapshot_cache_hit": False},
            )

            second_fp, _ = inspection_index.repository_fingerprint(
                discovered,
                excluded_paths=[cache_dir],
                cache_dir=cache_dir,
            )
            self.assertEqual(repository_fp, second_fp)

            with mock.patch.object(
                inspection_index,
                "discover_source_files",
                side_effect=AssertionError("unchanged hot run should reuse cached snapshot without rediscovery"),
            ):
                second_chunks, second_stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    excluded_paths=[cache_dir],
                    repository_state_fingerprint=second_fp,
                    return_diagnostics=True,
                )

        self.assertEqual(
            second_stats,
            {"total_files": 2, "reused_files": 2, "rebuilt_files": 0, "snapshot_cache_hit": True},
        )
        self.assertEqual(first_chunks, second_chunks)

    def test_file_chunk_snapshot_prefers_combined_snapshot_over_per_file_cache_reads(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            alpha = root / "alpha.py"
            beta = root / "beta.py"
            alpha.write_text("def alpha():\n    return 1\n", encoding="utf-8")
            beta.write_text("def beta():\n    return 2\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            repository_fp, _ = inspection_index.repository_fingerprint(
                discovered,
                excluded_paths=[cache_dir],
                cache_dir=cache_dir,
            )
            first_chunks, first_stats = inspection_index.build_syntax_chunks(
                discovered,
                cache_dir=cache_dir,
                excluded_paths=[cache_dir],
                repository_state_fingerprint=repository_fp,
                return_diagnostics=True,
            )

            self.assertEqual(
                first_stats,
                {"total_files": 2, "reused_files": 0, "rebuilt_files": 2, "snapshot_cache_hit": False},
            )

            second_fp, _ = inspection_index.repository_fingerprint(
                discovered,
                excluded_paths=[cache_dir],
                cache_dir=cache_dir,
            )
            self.assertEqual(repository_fp, second_fp)

            with mock.patch.object(
                inspection_index,
                "discover_source_files",
                side_effect=AssertionError("combined snapshot should avoid rediscovery"),
            ), mock.patch.object(
                inspection_index,
                "_load_file_chunk_cache",
                side_effect=AssertionError("combined snapshot should avoid per-file cache reloads"),
            ):
                second_chunks, second_stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    excluded_paths=[cache_dir],
                    repository_state_fingerprint=second_fp,
                    return_diagnostics=True,
                )

        self.assertEqual(
            second_stats,
            {"total_files": 2, "reused_files": 2, "rebuilt_files": 0, "snapshot_cache_hit": True},
        )
        self.assertEqual(first_chunks, second_chunks)

    def test_partial_dirty_run_reuses_previous_snapshot_without_reloading_unchanged_file_bundles(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            alpha = root / "alpha.py"
            beta = root / "beta.py"
            gamma = root / "gamma.py"
            alpha.write_text("def alpha():\n    return 1\n", encoding="utf-8")
            beta.write_text("def beta():\n    return 2\n", encoding="utf-8")
            gamma.write_text("def gamma():\n    return 3\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)

            first_fp, _ = inspection_index.repository_fingerprint(
                discovered,
                excluded_paths=[cache_dir],
                cache_dir=cache_dir,
            )
            first_chunks, first_stats = inspection_index.build_syntax_chunks(
                discovered,
                cache_dir=cache_dir,
                excluded_paths=[cache_dir],
                repository_state_fingerprint=first_fp,
                return_diagnostics=True,
            )
            self.assertEqual(
                first_stats,
                {"total_files": 3, "reused_files": 0, "rebuilt_files": 3, "snapshot_cache_hit": False},
            )

            beta.write_text("def beta():\n    return 22\n", encoding="utf-8")
            second_fp, _ = inspection_index.repository_fingerprint(
                discovered,
                excluded_paths=[cache_dir],
                cache_dir=cache_dir,
            )

            cache_loads = []
            original_load = inspection_index._load_file_chunk_cache_bundle

            def recording_load(cache_root, cache_key):
                cache_loads.append(str(cache_key))
                return original_load(cache_root, cache_key)

            with mock.patch.object(inspection_index, "_load_file_chunk_cache_bundle", side_effect=recording_load):
                second_chunks, second_stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    excluded_paths=[cache_dir],
                    repository_state_fingerprint=second_fp,
                    return_diagnostics=True,
                )

        self.assertEqual(second_stats["total_files"], 3)
        self.assertEqual(second_stats["rebuilt_files"], 1)
        self.assertEqual(second_stats["reused_files"], 2)
        self.assertFalse(second_stats["snapshot_cache_hit"])
        self.assertLessEqual(len(cache_loads), 1)
        alpha_first = [chunk for chunk in first_chunks if chunk["repository_path"] == "alpha.py"]
        alpha_second = [chunk for chunk in second_chunks if chunk["repository_path"] == "alpha.py"]
        gamma_first = [chunk for chunk in first_chunks if chunk["repository_path"] == "gamma.py"]
        gamma_second = [chunk for chunk in second_chunks if chunk["repository_path"] == "gamma.py"]
        beta_first = [chunk for chunk in first_chunks if chunk["repository_path"] == "beta.py"]
        beta_second = [chunk for chunk in second_chunks if chunk["repository_path"] == "beta.py"]
        self.assertEqual(alpha_first, alpha_second)
        self.assertEqual(gamma_first, gamma_second)
        self.assertNotEqual(beta_first, beta_second)

    def test_nonsource_git_change_reuses_existing_snapshot_payload_with_updated_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            alpha = root / "alpha.py"
            alpha.write_text("def alpha():\n    return 1\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)

            first_fp, _ = inspection_index.repository_fingerprint(
                discovered,
                excluded_paths=[cache_dir],
                cache_dir=cache_dir,
            )
            first_chunks, first_stats = inspection_index.build_syntax_chunks(
                discovered,
                cache_dir=cache_dir,
                excluded_paths=[cache_dir],
                repository_state_fingerprint=first_fp,
                return_diagnostics=True,
            )
            self.assertEqual(
                first_stats,
                {"total_files": 1, "reused_files": 0, "rebuilt_files": 1, "snapshot_cache_hit": False},
            )
            build_config_digest = (
                inspection_index._load_file_chunk_working_manifest(
                    inspection_index._file_chunk_manifest_path(cache_dir)
                )
                or {}
            ).get("build_config_digest", "")
            local_snapshot_path = inspection_index._file_chunk_snapshot_path(cache_dir)
            first_snapshot_bytes = local_snapshot_path.read_bytes()

            nonsource = root / ".gitignore"
            nonsource.write_text("cache/\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", ".gitignore"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "nonsource"], check=True)
            second_fp, _ = inspection_index.repository_fingerprint(
                discovered,
                excluded_paths=[cache_dir],
                cache_dir=cache_dir,
            )
            second_chunks, second_stats = inspection_index.build_syntax_chunks(
                discovered,
                cache_dir=cache_dir,
                excluded_paths=[cache_dir],
                repository_state_fingerprint=second_fp,
                return_diagnostics=True,
            )
            second_snapshot_bytes = local_snapshot_path.read_bytes()
            metadata = inspection_index._load_file_chunk_snapshot_metadata(cache_dir)
            restored = inspection_index.load_cached_chunk_snapshot(
                cache_dir,
                repository_state_fingerprint=second_fp,
                build_config_digest=build_config_digest,
            )

        self.assertEqual(second_stats, {"total_files": 1, "reused_files": 1, "rebuilt_files": 0, "snapshot_cache_hit": False})
        self.assertEqual(first_snapshot_bytes, second_snapshot_bytes)
        self.assertEqual(metadata["repository_state_fingerprint"], second_fp)
        self.assertEqual([dict(chunk) for chunk in restored], [dict(chunk) for chunk in second_chunks])
        self.assertEqual([dict(chunk) for chunk in first_chunks], [dict(chunk) for chunk in second_chunks])

    def test_reuse_only_partial_dirty_run_skips_full_snapshot_write_and_next_run_restores_from_manifest_bundles(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            alpha = root / "alpha.py"
            beta = root / "beta.py"
            alpha.write_text("def alpha():\n    return 'alpha'\n", encoding="utf-8")
            beta.write_text("def beta():\n    return 'beta'\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)

            first_fp, _ = inspection_index.repository_fingerprint(
                discovered,
                excluded_paths=[cache_dir],
                cache_dir=cache_dir,
            )
            inspection_index.build_syntax_chunks(
                discovered,
                cache_dir=cache_dir,
                excluded_paths=[cache_dir],
                repository_state_fingerprint=first_fp,
            )
            local_snapshot_path = inspection_index._file_chunk_snapshot_path(cache_dir)
            before_snapshot_bytes = local_snapshot_path.read_bytes()

            beta.write_text("def beta():\n    return 'gamma'\n", encoding="utf-8")
            second_fp, _ = inspection_index.repository_fingerprint(
                discovered,
                excluded_paths=[cache_dir],
                cache_dir=cache_dir,
            )
            inspection_index.build_syntax_chunks(
                discovered,
                cache_dir=cache_dir,
                excluded_paths=[cache_dir],
                repository_state_fingerprint=second_fp,
            )

            beta.write_text("def beta():\n    return 'beta'\n", encoding="utf-8")
            third_fp, _ = inspection_index.repository_fingerprint(
                discovered,
                excluded_paths=[cache_dir],
                cache_dir=cache_dir,
            )
            self.assertFalse(local_snapshot_path.exists())

            with mock.patch.object(
                inspection_index,
                "_write_file_chunk_snapshot",
                side_effect=AssertionError("reuse-only partial-dirty run should skip full snapshot write"),
            ):
                third_chunks, third_stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    excluded_paths=[cache_dir],
                    repository_state_fingerprint=third_fp,
                    return_diagnostics=True,
                )

            metadata = inspection_index._load_file_chunk_snapshot_metadata(cache_dir)
            local_snapshot_path.unlink(missing_ok=True)
            with mock.patch.object(
                inspection_index,
                "discover_source_files",
                side_effect=AssertionError("manifest-plus-bundle warm restore should avoid rediscovery"),
            ):
                fourth_chunks, fourth_stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    excluded_paths=[cache_dir],
                    repository_state_fingerprint=third_fp,
                    return_diagnostics=True,
                )

        self.assertEqual(third_stats["rebuilt_files"], 0)
        self.assertEqual(third_stats["reused_files"], 2)
        self.assertEqual(local_snapshot_path.exists(), False)
        self.assertEqual(metadata["repository_state_fingerprint"], third_fp)
        self.assertNotEqual(before_snapshot_bytes, b"")
        self.assertTrue(fourth_stats["snapshot_cache_hit"])
        self.assertEqual([dict(chunk) for chunk in third_chunks], [dict(chunk) for chunk in fourth_chunks])

    def test_partial_dirty_rebuild_skips_full_snapshot_write_and_next_run_restores_from_manifest_bundles(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            alpha = root / "alpha.py"
            beta = root / "beta.py"
            alpha.write_text("def alpha():\n    return 'alpha'\n", encoding="utf-8")
            beta.write_text("def beta():\n    return 'beta'\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)

            first_fp, _ = inspection_index.repository_fingerprint(
                discovered,
                excluded_paths=[cache_dir],
                cache_dir=cache_dir,
            )
            inspection_index.build_syntax_chunks(
                discovered,
                cache_dir=cache_dir,
                excluded_paths=[cache_dir],
                repository_state_fingerprint=first_fp,
            )

            beta.write_text("def beta():\n    return 'gamma'\n", encoding="utf-8")
            second_fp, _ = inspection_index.repository_fingerprint(
                discovered,
                excluded_paths=[cache_dir],
                cache_dir=cache_dir,
            )
            local_snapshot_path = inspection_index._file_chunk_snapshot_path(cache_dir)
            before_snapshot_bytes = local_snapshot_path.read_bytes()

            with mock.patch.object(
                inspection_index,
                "_write_file_chunk_snapshot",
                side_effect=AssertionError("partial-dirty local rebuild should skip full snapshot write"),
            ):
                second_chunks, second_stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    excluded_paths=[cache_dir],
                    repository_state_fingerprint=second_fp,
                    return_diagnostics=True,
                )

            metadata = inspection_index._load_file_chunk_snapshot_metadata(cache_dir)
            self.assertFalse(local_snapshot_path.exists())
            self.assertEqual(metadata["repository_state_fingerprint"], second_fp)
            self.assertNotEqual(before_snapshot_bytes, b"")
            self.assertEqual(second_stats["rebuilt_files"], 1)
            self.assertEqual(second_stats["reused_files"], 1)

            with mock.patch.object(
                inspection_index,
                "discover_source_files",
                side_effect=AssertionError("manifest-plus-bundle warm restore should avoid rediscovery"),
            ):
                third_chunks, third_stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    excluded_paths=[cache_dir],
                    repository_state_fingerprint=second_fp,
                    return_diagnostics=True,
                )

        self.assertTrue(third_stats["snapshot_cache_hit"])
        self.assertEqual([dict(chunk) for chunk in second_chunks], [dict(chunk) for chunk in third_chunks])

    def test_lexical_index_updates_only_changed_files_and_then_hits_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            alpha = root / "alpha.py"
            beta = root / "beta.py"
            alpha.write_text("def alpha():\n    return 'alpha token'\n", encoding="utf-8")
            beta.write_text("def beta():\n    return 'beta token'\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            first_chunks = inspection_index.build_syntax_chunks(discovered, cache_dir=cache_dir)
            first_fp = inspection_index.inspection_index_fingerprint("sha256:first", first_chunks)
            first_path, first_hit, first_stats = inspection_index.ensure_lexical_index(first_chunks, cache_dir, first_fp)
            self.assertFalse(first_hit)
            self.assertEqual(first_stats["updated_files"], 2)
            self.assertEqual(first_stats["removed_files"], 0)
            self.assertEqual(first_stats["inserted_chunks"], len(first_chunks))

            beta.write_text("def beta():\n    return 'gamma token'\n", encoding="utf-8")
            second_chunks = inspection_index.build_syntax_chunks(discovered, cache_dir=cache_dir)
            second_fp = inspection_index.inspection_index_fingerprint("sha256:second", second_chunks)
            second_path, second_hit, second_stats = inspection_index.ensure_lexical_index(second_chunks, cache_dir, second_fp)
            self.assertFalse(second_hit)
            self.assertEqual(second_stats["updated_files"], 1)
            self.assertEqual(second_stats["removed_files"], 0)
            beta_chunks = [chunk for chunk in second_chunks if chunk["repository_path"] == "beta.py"]
            self.assertEqual(second_stats["inserted_chunks"], len(beta_chunks))

            with sqlite3.connect(second_path) as conn:
                paths = [row[0] for row in conn.execute("SELECT DISTINCT repository_path FROM chunks ORDER BY repository_path")]
            self.assertEqual(paths, ["alpha.py", "beta.py"])

            gamma_ranked = inspection_index.lexical_search(second_path, "gamma", second_chunks)
            alpha_ranked = inspection_index.lexical_search(second_path, "alpha", second_chunks)
            chunk_by_id = {chunk["chunk_id"]: chunk for chunk in second_chunks}
            self.assertEqual(chunk_by_id[gamma_ranked[0]["chunk_id"]]["repository_path"], "beta.py")
            self.assertEqual(chunk_by_id[alpha_ranked[0]["chunk_id"]]["repository_path"], "alpha.py")

            third_path, third_hit, third_stats = inspection_index.ensure_lexical_index(second_chunks, cache_dir, second_fp)
            self.assertTrue(third_hit)
            self.assertEqual(third_path, second_path)
            self.assertTrue(third_stats["working_cache_hit"])
            self.assertEqual(third_stats["updated_files"], 0)
            self.assertEqual(third_stats["removed_files"], 0)
            self.assertEqual(third_stats["inserted_chunks"], 0)

    def test_lexical_index_delta_recovers_when_cached_chunks_by_file_is_partial(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            service = root / "service.py"
            mcp = root / "mcp.go"
            service.write_text(
                "def retry_job(job_id):\n    return submit_job(job_id)\n\ndef submit_job(job_id):\n    return job_id\n",
                encoding="utf-8",
            )
            mcp.write_text(
                "package mcp\n\nfunc InspectRepo(query string) string {\n\treturn query\n}\n",
                encoding="utf-8",
            )
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            first_chunks = inspection_index.build_syntax_chunks(discovered, cache_dir=cache_dir)
            first_fp = inspection_index.inspection_index_fingerprint("sha256:first", first_chunks)
            _, _, first_stats = inspection_index.ensure_lexical_index(first_chunks, cache_dir, first_fp)
            self.assertEqual(first_stats["updated_files"], 2)

            mcp.write_text(
                "package mcp\n\nfunc InspectRepo(query string) string {\n\tvalue := query\n\treturn value\n}\n",
                encoding="utf-8",
            )
            second_chunks = inspection_index.build_syntax_chunks(discovered, cache_dir=cache_dir)
            second_chunks._chunks_by_file = {
                key: value
                for key, value in dict(getattr(second_chunks, "_chunks_by_file", {}) or {}).items()
                if not str(key).endswith("\x00mcp.go")
            }
            second_fp = inspection_index.inspection_index_fingerprint("sha256:second", second_chunks)

            second_path, second_hit, second_stats = inspection_index.ensure_lexical_index(second_chunks, cache_dir, second_fp)

        self.assertFalse(second_hit)
        self.assertEqual(second_path, cache_dir / "lexical-working.sqlite3")
        self.assertEqual(second_stats["updated_files"], 1)
        self.assertGreaterEqual(second_stats["inserted_chunks"], 1)

    def test_lexical_index_delta_removes_deleted_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            alpha = root / "alpha.py"
            beta = root / "beta.py"
            alpha.write_text("def alpha():\n    return 'alpha token'\n", encoding="utf-8")
            beta.write_text("def beta():\n    return 'beta token'\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            first_chunks = inspection_index.build_syntax_chunks(discovered, cache_dir=cache_dir)
            first_fp = inspection_index.inspection_index_fingerprint("sha256:first", first_chunks)
            _, _, first_stats = inspection_index.ensure_lexical_index(first_chunks, cache_dir, first_fp)
            self.assertEqual(first_stats["updated_files"], 2)

            beta.unlink()
            second_chunks = inspection_index.build_syntax_chunks(discovered, cache_dir=cache_dir)
            second_fp = inspection_index.inspection_index_fingerprint("sha256:second", second_chunks)
            second_path, second_hit, second_stats = inspection_index.ensure_lexical_index(second_chunks, cache_dir, second_fp)
            self.assertFalse(second_hit)
            self.assertEqual(second_stats["updated_files"], 0)
            self.assertEqual(second_stats["removed_files"], 1)
            self.assertEqual(second_stats["inserted_chunks"], 0)

            with sqlite3.connect(second_path) as conn:
                paths = [row[0] for row in conn.execute("SELECT DISTINCT repository_path FROM chunks ORDER BY repository_path")]
            self.assertEqual(paths, ["alpha.py"])

            ranked = inspection_index.lexical_search(second_path, "beta", second_chunks)
            self.assertTrue(ranked)
            chunk_by_id = {chunk["chunk_id"]: chunk for chunk in second_chunks}
            self.assertTrue(all(chunk_by_id[item["chunk_id"]]["repository_path"] == "alpha.py" for item in ranked))

    def test_lexical_index_reuses_single_working_db_across_fingerprint_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            alpha = root / "alpha.py"
            beta = root / "beta.py"
            alpha.write_text("def alpha():\n    return 1\n", encoding="utf-8")
            beta.write_text("def beta():\n    return 2\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            first_chunks = inspection_index.build_syntax_chunks(discovered, cache_dir=cache_dir)
            first_fp = inspection_index.inspection_index_fingerprint("sha256:first", first_chunks)
            first_path, first_hit, _ = inspection_index.ensure_lexical_index(first_chunks, cache_dir, first_fp)

            beta.write_text("def beta():\n    return 3\n", encoding="utf-8")
            second_chunks = inspection_index.build_syntax_chunks(discovered, cache_dir=cache_dir)
            second_fp = inspection_index.inspection_index_fingerprint("sha256:second", second_chunks)
            second_path, second_hit, _ = inspection_index.ensure_lexical_index(second_chunks, cache_dir, second_fp)

            sqlite_files = sorted(cache_dir.glob("*.sqlite3"))

        self.assertFalse(first_hit)
        self.assertFalse(second_hit)
        self.assertEqual(first_path, cache_dir / "lexical-working.sqlite3")
        self.assertEqual(second_path, cache_dir / "lexical-working.sqlite3")
        self.assertEqual(sqlite_files, [cache_dir / "lexical-working.sqlite3"])

    def test_lexical_index_hot_hit_skips_sqlite_open_when_manifest_meta_matches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            source = root / "worker.py"
            source.write_text("def worker():\n    return 'token'\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            chunks = inspection_index.build_syntax_chunks(discovered, cache_dir=cache_dir)
            fingerprint = inspection_index.inspection_index_fingerprint("sha256:repo", chunks)
            first_path, first_hit, _ = inspection_index.ensure_lexical_index(chunks, cache_dir, fingerprint)

            with mock.patch.object(
                inspection_index.sqlite3,
                "connect",
                side_effect=AssertionError("matching lexical manifest metadata should avoid sqlite open on hot hit"),
            ):
                second_path, second_hit, second_stats = inspection_index.ensure_lexical_index(chunks, cache_dir, fingerprint)

        self.assertFalse(first_hit)
        self.assertTrue(second_hit)
        self.assertEqual(first_path, second_path)
        self.assertTrue(second_stats["working_cache_hit"])

    def test_lexical_index_fingerprint_only_change_skips_chunk_rewrites(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            source = root / "worker.py"
            source.write_text("def worker():\n    return 'token'\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            chunks = inspection_index.build_syntax_chunks(discovered, cache_dir=cache_dir)
            first_fp = inspection_index.inspection_index_fingerprint("sha256:first", chunks)
            first_path, first_hit, _ = inspection_index.ensure_lexical_index(chunks, cache_dir, first_fp)

            with mock.patch.object(
                inspection_index.sqlite3,
                "connect",
                side_effect=AssertionError("fingerprint-only lexical refresh should avoid sqlite open when lexical manifest is unchanged"),
            ), mock.patch.object(
                inspection_index,
                "_insert_lexical_chunks",
                side_effect=AssertionError("fingerprint-only lexical refresh should not rewrite chunk rows"),
            ):
                second_fp = inspection_index.inspection_index_fingerprint("sha256:second", chunks)
                second_path, second_hit, second_stats = inspection_index.ensure_lexical_index(
                    chunks,
                    cache_dir,
                    second_fp,
                )

            reloaded = inspection_index.load_chunks_from_lexical_index(second_path, second_fp, include_content=False)

        self.assertFalse(first_hit)
        self.assertTrue(second_hit)
        self.assertEqual(first_path, second_path)
        self.assertTrue(second_stats["working_cache_hit"])
        self.assertEqual(second_stats["updated_files"], 0)
        self.assertEqual(second_stats["removed_files"], 0)
        self.assertEqual(second_stats["inserted_chunks"], 0)
        self.assertGreaterEqual(len(reloaded or ()), 1)

    def test_lexical_path_catalog_reuses_cached_helper_backed_catalog(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            source = root / "worker.py"
            source.write_text("def worker():\n    return 'token'\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            chunks = inspection_index.build_syntax_chunks(discovered, cache_dir=cache_dir)
            fingerprint = inspection_index.inspection_index_fingerprint("sha256:repo", chunks)
            index_path, _, _ = inspection_index.ensure_lexical_index(chunks, cache_dir, fingerprint)
            cache_key = inspection_index.lexical_cache_key(index_path, chunks)

            first = inspection_index.lexical_path_catalog(index_path, chunks, cache_key=cache_key)
            second = inspection_index.lexical_path_catalog(index_path, chunks, cache_key=cache_key)

        self.assertIs(first, second)
        self.assertEqual(first["unique_paths"], ("worker.py",))

    def test_delete_lexical_fts_chunk_ids_batches_single_statement(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "lexical.sqlite3"
            with inspection_index.sqlite3.connect(path) as conn:
                inspection_index._configure_lexical_rebuild_connection(conn)
                inspection_index._initialize_lexical_schema(conn)
                conn.executemany(
                    "INSERT INTO chunks_fts(chunk_id,path,symbol,content) VALUES(?,?,?,?)",
                    [
                        ("a", "worker.py", "worker", "alpha"),
                        ("b", "worker.py", "worker", "beta"),
                        ("c", "worker.py", "worker", "gamma"),
                    ],
                )
                inspection_index._delete_lexical_fts_chunk_ids(conn, ["a", "b"])
                remaining = conn.execute("SELECT chunk_id FROM chunks_fts ORDER BY chunk_id").fetchall()

        self.assertEqual(remaining, [("c",)])

    def test_pipeline_reuses_single_lexical_helper_for_catalog_and_search(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "service.py").write_text(
                "def retry_job(job_id):\n    return submit_job(job_id)\n\ndef submit_job(job_id):\n    return job_id\n",
                encoding="utf-8",
            )
            (root / "mcp.go").write_text(
                "package mcp\n\nfunc InspectRepo(query string) string {\n\treturn query\n}\n",
                encoding="utf-8",
            )
            discovered = [{"id": "input_0", "type": "repo", "classification": "internal", "path": root}]
            factory = SemanticDiagnosticsFactory()
            original_get = inspection_index._get_lexical_helper
            calls = []

            def counting_get(*args, **kwargs):
                calls.append(1)
                return original_get(*args, **kwargs)

            with mock.patch.object(inspection_index, "_get_lexical_helper", side_effect=counting_get):
                payload = inspection_pipeline.run_inspection(
                    discovered,
                    "Trace the retry_job service call chain",
                    mode="evidence",
                    task_params={"index_cache_dir": str(root / ".broker" / "inspection-test")},
                    services=all_services(),
                    client_factory=factory,
                    output_dir=root / "out",
                )["payload"]

        self.assertFalse(payload["retrieval"]["query_stage_cache_hit"])
        self.assertEqual(len(calls), 1)

    def test_pipeline_skips_path_catalog_when_query_has_no_path_tokens(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "service.py").write_text(
                "def retry_job(job_id):\n    return submit_job(job_id)\n\ndef submit_job(job_id):\n    return job_id\n",
                encoding="utf-8",
            )
            (root / "mcp.go").write_text(
                "package mcp\n\nfunc InspectRepo(query string) string {\n\treturn query\n}\n",
                encoding="utf-8",
            )
            discovered = [{"id": "input_0", "type": "repo", "classification": "internal", "path": root}]
            factory = SemanticDiagnosticsFactory()

            with mock.patch.object(
                inspection_pipeline,
                "lexical_path_catalog",
                side_effect=AssertionError("path catalog should be skipped when query has no path tokens"),
            ):
                payload = inspection_pipeline.run_inspection(
                    discovered,
                    "Trace the retry_job service call chain",
                    mode="evidence",
                    task_params={"index_cache_dir": str(root / ".broker" / "inspection-test")},
                    services=all_services(),
                    client_factory=factory,
                    output_dir=root / "out",
                )["payload"]

        self.assertFalse(payload["retrieval"]["query_stage_cache_hit"])
        self.assertEqual(payload["retrieval"]["stage_timings_ms"]["path_catalog_ms"], 0.0)
        self.assertEqual(payload["retrieval"]["stage_timings_ms"]["named_paths_ms"], 0.0)

    def test_dehydrate_chunks_preserves_chunk_diagnostics_attributes(self):
        chunks = type("ChunkListWithAttrs", (list,), {})()
        chunks.append({"chunk_id": "chunk_1", "content": "hello", "path": "worker.py"})
        chunks._chunk_restore_diagnostics = {"snapshot_restore_source": "shared"}
        chunks._chunk_build_substage_timings = {"discover_source_files_ms": 1.25}
        chunks._chunk_count = 1

        dehydrated = inspection_pipeline._dehydrate_chunks(chunks)

        self.assertEqual(dehydrated, [{"chunk_id": "chunk_1", "path": "worker.py"}])
        self.assertEqual(getattr(dehydrated, "_chunk_restore_diagnostics", {}), {"snapshot_restore_source": "shared"})
        self.assertEqual(
            getattr(dehydrated, "_chunk_build_substage_timings", {}),
            {"discover_source_files_ms": 1.25},
        )
        self.assertEqual(getattr(dehydrated, "_chunk_count", 0), 1)

    def test_lexical_index_uses_attached_manifest_without_rebuilding_from_chunks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            source = root / "worker.py"
            source.write_text("def worker():\n    return 'token'\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            chunks = inspection_index.build_syntax_chunks(discovered, cache_dir=cache_dir)
            fingerprint = inspection_index.inspection_index_fingerprint("sha256:repo", chunks)
            with mock.patch.object(
                inspection_index,
                "_lexical_manifest_for_chunks",
                side_effect=AssertionError("attached lexical manifest should be reused"),
            ):
                path, hit, stats = inspection_index.ensure_lexical_index(chunks, cache_dir, fingerprint)

        self.assertFalse(hit)
        self.assertEqual(path, cache_dir / "lexical-working.sqlite3")
        self.assertEqual(stats["updated_files"], 1)

    def test_lexical_index_delta_uses_attached_grouped_chunks_without_regrouping_all_chunks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            alpha = root / "alpha.py"
            beta = root / "beta.py"
            alpha.write_text("def alpha():\n    return 'alpha'\n", encoding="utf-8")
            beta.write_text("def beta():\n    return 'beta'\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            first_chunks = inspection_index.build_syntax_chunks(discovered, cache_dir=cache_dir)
            first_fp = inspection_index.inspection_index_fingerprint("sha256:first", first_chunks)
            inspection_index.ensure_lexical_index(first_chunks, cache_dir, first_fp)

            beta.write_text("def beta():\n    return 'gamma'\n", encoding="utf-8")
            second_chunks = inspection_index.build_syntax_chunks(discovered, cache_dir=cache_dir)
            second_fp = inspection_index.inspection_index_fingerprint("sha256:second", second_chunks)

            class NoIterChunkList(inspection_index.ChunkList):
                def __iter__(self):
                    raise AssertionError("attached grouped chunks should avoid regrouping whole chunk list")

            guarded = NoIterChunkList(list(second_chunks))
            guarded._lexical_manifest = second_chunks._lexical_manifest
            guarded._index_manifest = second_chunks._index_manifest
            guarded._chunks_by_file = second_chunks._chunks_by_file

            path, hit, stats = inspection_index.ensure_lexical_index(guarded, cache_dir, second_fp)

        self.assertFalse(hit)
        self.assertEqual(path, cache_dir / "lexical-working.sqlite3")
        self.assertEqual(stats["updated_files"], 1)
        self.assertEqual(stats["removed_files"], 0)

    def test_partial_dirty_chunk_rebuild_reuses_previous_snapshot_and_skips_file_bundle_reload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            alpha = root / "alpha.py"
            beta = root / "beta.py"
            alpha.write_text("def alpha():\n    return 'alpha'\n", encoding="utf-8")
            beta.write_text("def beta():\n    return 'beta'\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)

            first_fp, _ = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)
            inspection_index.build_syntax_chunks(
                discovered,
                cache_dir=cache_dir,
                repository_state_fingerprint=first_fp,
            )

            beta.write_text("def beta():\n    return 'gamma'\n", encoding="utf-8")
            second_fp, _ = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

            cache_loads = []
            original_load = inspection_index._load_file_chunk_cache_bundle

            def recording_load(cache_root, cache_key):
                cache_loads.append(str(cache_key))
                return original_load(cache_root, cache_key)

            with mock.patch.object(
                inspection_index,
                "_load_file_chunk_cache_bundle",
                side_effect=recording_load,
            ):
                chunks, stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    repository_state_fingerprint=second_fp,
                    return_diagnostics=True,
                )

        self.assertEqual(stats["rebuilt_files"], 1)
        self.assertEqual(stats["reused_files"], 1)
        self.assertLessEqual(len(cache_loads), 1)
        self.assertEqual(len(chunks), 2)

    def test_partial_dirty_warm_snapshot_skips_git_file_signatures_for_dirty_entries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            alpha = root / "alpha.py"
            beta = root / "beta.py"
            alpha.write_text("def alpha():\n    return 'alpha'\n", encoding="utf-8")
            beta.write_text("def beta():\n    return 'beta'\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)

            first_fp, _ = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)
            inspection_index.build_syntax_chunks(
                discovered,
                cache_dir=cache_dir,
                repository_state_fingerprint=first_fp,
            )

            beta.write_text("def beta():\n    return 'gamma'\n", encoding="utf-8")
            second_fp, _ = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

            with mock.patch.object(
                inspection_index,
                "_git_file_signatures",
                side_effect=AssertionError("warm partial-dirty snapshot reuse should not request clean git blob signatures"),
            ):
                chunks, stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=cache_dir,
                    repository_state_fingerprint=second_fp,
                    return_diagnostics=True,
                )

        self.assertEqual(stats["rebuilt_files"], 1)
        self.assertEqual(stats["reused_files"], 1)
        self.assertEqual(len(chunks), 2)

    def test_repeated_partial_dirty_jobs_reuse_previous_manifest_from_shared_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            shared_cache = temp_root / "shared-cache"
            run1_cache = temp_root / "run1-cache"
            run2_cache = temp_root / "run2-cache"
            run3_cache = temp_root / "run3-cache"
            service = root / "service.py"
            helper = root / "helper.py"
            service.write_text(
                "def retry_job(job_id):\n    return submit_job(job_id)\n\ndef submit_job(job_id):\n    return job_id\n",
                encoding="utf-8",
            )
            helper.write_text("def helper():\n    return 1\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)

            with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache)}, clear=False):
                first_fp, _ = inspection_index.repository_fingerprint(
                    discovered,
                    excluded_paths=[run1_cache],
                    cache_dir=run1_cache,
                )
                _first_chunks, first_stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=run1_cache,
                    excluded_paths=[run1_cache],
                    repository_state_fingerprint=first_fp,
                    return_diagnostics=True,
                )
                self.assertEqual(
                    first_stats,
                    {"total_files": 2, "reused_files": 0, "rebuilt_files": 2, "snapshot_cache_hit": False},
                )

                service.write_text(
                    "def retry_job(job_id):\n    value = 1\n    return submit_job(job_id + value)\n\ndef submit_job(job_id):\n    return job_id\n",
                    encoding="utf-8",
                )
                second_fp, _ = inspection_index.repository_fingerprint(
                    discovered,
                    excluded_paths=[run2_cache],
                    cache_dir=run2_cache,
                )
                _second_chunks, second_stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=run2_cache,
                    excluded_paths=[run2_cache],
                    repository_state_fingerprint=second_fp,
                    return_diagnostics=True,
                )
                self.assertEqual(second_stats["reused_files"], 1)
                self.assertEqual(second_stats["rebuilt_files"], 1)

                service.write_text(
                    "def retry_job(job_id):\n    value = 2\n    return submit_job(job_id + value)\n\ndef submit_job(job_id):\n    return job_id\n",
                    encoding="utf-8",
                )
                third_fp, _ = inspection_index.repository_fingerprint(
                    discovered,
                    excluded_paths=[run3_cache],
                    cache_dir=run3_cache,
                )
                _third_chunks, third_stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=run3_cache,
                    excluded_paths=[run3_cache],
                    repository_state_fingerprint=third_fp,
                    return_diagnostics=True,
                )

        self.assertEqual(third_stats["reused_files"], 1)

    def test_partial_dirty_shared_state_reuses_changed_file_across_different_repo_roots_with_same_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            shared_cache = temp_root / "shared-cache"
            run1_root = temp_root / "repo-one"
            run2_root = temp_root / "repo-two"
            run1_root.mkdir()
            run2_root.mkdir()
            run1_cache = temp_root / "run1-cache"
            run2_cache = temp_root / "run2-cache"

            for root in (run1_root, run2_root):
                (root / "service.py").write_text(
                    "def retry_job(job_id):\n    return submit_job(job_id)\n\ndef submit_job(job_id):\n    return job_id\n",
                    encoding="utf-8",
                )
                (root / "helper.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
                subprocess.run(["git", "init", "-q", str(root)], check=True)
                subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
                subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
                subprocess.run(["git", "-C", str(root), "add", "."], check=True)
                subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)

            run1_discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": run1_root}]
            run2_discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": run2_root}]

            with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache)}, clear=False):
                first_fp, _ = inspection_index.repository_fingerprint(
                    run1_discovered,
                    excluded_paths=[run1_cache],
                    cache_dir=run1_cache,
                )
                inspection_index.build_syntax_chunks(
                    run1_discovered,
                    cache_dir=run1_cache,
                    excluded_paths=[run1_cache],
                    repository_state_fingerprint=first_fp,
                    return_diagnostics=True,
                )

                updated_text = (
                    "def retry_job(job_id):\n    value = 1\n    return submit_job(job_id + value)\n\n"
                    "def submit_job(job_id):\n    return job_id\n"
                )
                (run1_root / "service.py").write_text(updated_text, encoding="utf-8")
                (run2_root / "service.py").write_text(updated_text, encoding="utf-8")

                second_fp, _ = inspection_index.repository_fingerprint(
                    run1_discovered,
                    excluded_paths=[run1_cache],
                    cache_dir=run1_cache,
                )
                inspection_index.build_syntax_chunks(
                    run1_discovered,
                    cache_dir=run1_cache,
                    excluded_paths=[run1_cache],
                    repository_state_fingerprint=second_fp,
                    return_diagnostics=True,
                )

                third_fp, _ = inspection_index.repository_fingerprint(
                    run2_discovered,
                    excluded_paths=[run2_cache],
                    cache_dir=run2_cache,
                )
                third_chunks, third_stats = inspection_index.build_syntax_chunks(
                    run2_discovered,
                    cache_dir=run2_cache,
                    excluded_paths=[run2_cache],
                    repository_state_fingerprint=third_fp,
                    return_diagnostics=True,
                )

        self.assertTrue(third_chunks)
        self.assertEqual(third_stats["reused_files"], 2)
        self.assertEqual(third_stats["rebuilt_files"], 0)
        self.assertFalse(third_stats["snapshot_cache_hit"])

    def test_repository_fingerprint_ignores_broker_owned_cache_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "repo-inspection-cache"
            cache_dir.mkdir(exist_ok=True)
            source = root / "worker.py"
            source.write_text("def worker():\n    return 1\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            before, _ = inspection_index.repository_fingerprint(discovered, excluded_paths=[cache_dir])
            (cache_dir / "artifact.json").write_text("{\"cache\":true}\n", encoding="utf-8")
            after, _ = inspection_index.repository_fingerprint(discovered, excluded_paths=[cache_dir])

        self.assertEqual(before, after)

    def test_non_git_repository_fingerprint_hot_run_does_not_reread_unchanged_file_with_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            source = root / "worker.py"
            source.write_text("def worker():\n    return 1\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            first, _ = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

            original_read_bytes = Path.read_bytes

            def guarded_read_bytes(path_self, *args, **kwargs):
                if path_self == source:
                    raise AssertionError("unchanged non-git file should not be reread for fingerprint")
                return original_read_bytes(path_self, *args, **kwargs)

            with mock.patch.object(Path, "read_bytes", autospec=True, side_effect=guarded_read_bytes):
                second, states = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

        self.assertEqual(first, second)
        self.assertEqual(states[0]["kind"], "metadata")

    def test_non_git_repository_fingerprint_cache_rehashes_changed_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            source = root / "worker.py"
            source.write_text("def worker():\n    return 1\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            first, _ = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)
            source.write_text("def worker():\n    return 2\n", encoding="utf-8")
            second, _ = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

        self.assertNotEqual(first, second)

    def test_non_git_repository_fingerprint_hot_run_skips_manifest_rewrite_when_unchanged(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            source = root / "worker.py"
            source.write_text("def worker():\n    return 1\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            first, _ = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

            with mock.patch.object(
                inspection_index.os,
                "replace",
                side_effect=AssertionError("unchanged metadata fingerprint manifest should not be rewritten"),
            ):
                second, _ = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

        self.assertEqual(first, second)

    def test_hotpath_non_git_repository_fingerprint_reuses_process_cache_when_unchanged(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            source = root / "worker.py"
            source.write_text("def worker():\n    return 1\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            inspection_hotpath._METADATA_REPOSITORY_STATE_CACHE.clear()
            first, _ = inspection_hotpath.repository_fingerprint(discovered, cache_dir=cache_dir)

            with mock.patch.object(
                inspection_hotpath,
                "_iter_source_candidates",
                side_effect=AssertionError("warm metadata fingerprint should reuse process cache"),
            ):
                second, states = inspection_hotpath.repository_fingerprint(discovered, cache_dir=cache_dir)

        self.assertEqual(first, second)
        self.assertEqual(states[0]["kind"], "metadata")

    def test_hotpath_non_git_repository_fingerprint_process_cache_invalidates_on_new_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            source = root / "worker.py"
            source.write_text("def worker():\n    return 1\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            inspection_hotpath._METADATA_REPOSITORY_STATE_CACHE.clear()
            first, _ = inspection_hotpath.repository_fingerprint(discovered, cache_dir=cache_dir)
            (root / "helper.py").write_text("def helper():\n    return 2\n", encoding="utf-8")
            second, states = inspection_hotpath.repository_fingerprint(discovered, cache_dir=cache_dir)

        self.assertNotEqual(first, second)
        self.assertEqual(states[0]["kind"], "metadata")

    def test_non_git_repository_fingerprint_reuses_shared_manifest_across_local_cache_roots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            shared_cache_dir = temp_root / "shared-cache"
            first_cache_dir = temp_root / "cache-one"
            second_cache_dir = temp_root / "cache-two"
            source = root / "worker.py"
            source.write_text("def worker():\n    return 1\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
                first, _ = inspection_index.repository_fingerprint(discovered, cache_dir=first_cache_dir)

                original_read_bytes = Path.read_bytes

                def guarded_read_bytes(path_self, *args, **kwargs):
                    if path_self == source:
                        raise AssertionError("shared metadata fingerprint manifest should avoid rereading unchanged file content")
                    return original_read_bytes(path_self, *args, **kwargs)

                with mock.patch.object(Path, "read_bytes", autospec=True, side_effect=guarded_read_bytes):
                    second, _ = inspection_index.repository_fingerprint(discovered, cache_dir=second_cache_dir)

        self.assertEqual(first, second)
        self.assertFalse(inspection_index._metadata_manifest_path(second_cache_dir, root).exists())

    def test_repository_fingerprint_for_subdirectory_ignores_unrelated_sibling_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test User"], check=True)
            target = root / "target"
            target.mkdir()
            sibling = root / "other"
            sibling.mkdir()
            (target / "worker.py").write_text("def worker():\n    return 1\n", encoding="utf-8")
            (sibling / "noise.py").write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "init"], check=True)
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": target}]

            before, states_before = inspection_index.repository_fingerprint(discovered)
            (sibling / "noise.py").write_text("value = 2\n", encoding="utf-8")
            after, states_after = inspection_index.repository_fingerprint(discovered)

        self.assertEqual(before, after)
        self.assertEqual(states_before[0]["kind"], "git")
        self.assertEqual(states_before[0].get("scope_paths"), ["target"])
        self.assertEqual(states_after[0].get("scope_paths"), ["target"])

    def test_repository_fingerprint_ignores_absolute_git_root_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            first = base / "first"
            second = base / "second"
            for root in (first, second):
                root.mkdir()
                subprocess.run(["git", "init", "-q", str(root)], check=True)
                subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
                subprocess.run(["git", "-C", str(root), "config", "user.name", "Test User"], check=True)
                target = root / "target"
                target.mkdir()
                (target / "worker.py").write_text("def worker():\n    return 1\n", encoding="utf-8")
                subprocess.run(["git", "-C", str(root), "add", "."], check=True)
                subprocess.run(["git", "-C", str(root), "commit", "-qm", "init"], check=True)
            discovered_first = [{"id": "repo", "type": "repo", "classification": "internal", "path": first / "target"}]
            discovered_second = [{"id": "repo", "type": "repo", "classification": "internal", "path": second / "target"}]

            fingerprint_first, states_first = inspection_index.repository_fingerprint(discovered_first)
            fingerprint_second, states_second = inspection_index.repository_fingerprint(discovered_second)

        self.assertEqual(fingerprint_first, fingerprint_second)
        self.assertNotIn("root", states_first[0])
        self.assertNotIn("root", states_second[0])

    def test_git_top_uses_parent_git_marker_before_rev_parse(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            nested = root / "src" / "pkg"
            nested.mkdir(parents=True)
            inspection_index._GIT_TOP_CACHE.clear()

            with mock.patch.object(
                inspection_index,
                "_run_git",
                side_effect=AssertionError("git rev-parse should not be required when a parent .git marker exists"),
            ):
                resolved = inspection_index._git_top(nested)

        self.assertEqual(resolved, root.resolve())

    def test_git_repository_fingerprint_uses_head_for_whole_repo_scope(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            (root / "main.py").write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "main.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]

            with mock.patch.object(
                inspection_index,
                "_git_scope_head_oid",
                side_effect=AssertionError("whole-repo scope should use HEAD directly"),
            ):
                fingerprint, states = inspection_index.repository_fingerprint(discovered)

        self.assertTrue(str(fingerprint).startswith("sha256:"))
        self.assertEqual(states[0]["kind"], "git")

    def test_prepare_prefetched_state_uses_head_for_whole_repo_scope(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            (root / "main.py").write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "main.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]

            with mock.patch.object(
                inspection_hotpath,
                "_git_scope_head_oid",
                side_effect=AssertionError("whole-repo scope should use HEAD directly during prefetch"),
            ):
                state = inspection_hotpath.prepare_prefetched_state(
                    discovered,
                    "trace main",
                    mode="evidence",
                    constraints={},
                    task_params={},
                    execution_plan={"repo_inspection_cache_path": str(root / ".broker" / "cache")},
                    output_dir=root / "out",
                )

        self.assertTrue(str(state["repository_state_fingerprint"]).startswith("sha256:"))

    def test_git_repository_fingerprint_clean_hot_run_avoids_git_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "main.py"
            source.write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "main.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]

            first, _ = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

            original_run = inspection_index.subprocess.run

            def guarded_run(args, *pargs, **kwargs):
                if git_args_match_legacy_clean_probe(args, root):
                    raise AssertionError("clean hot fingerprint should use one status probe instead of legacy clean probes")
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=guarded_run):
                second, states = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

        self.assertEqual(first, second)
        self.assertEqual(states[0]["kind"], "git")

    def test_git_repository_fingerprint_clean_hot_run_avoids_name_only_diff_probe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "main.py"
            source.write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "main.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]

            first, _ = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

            original_run = inspection_index.subprocess.run

            def guarded_run(args, *pargs, **kwargs):
                if git_args_match(args, root, "diff") and "--name-only" in args:
                    raise AssertionError("clean hot fingerprint should avoid name-only diff probe")
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=guarded_run):
                second, states = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

        self.assertEqual(first, second)
        self.assertEqual(states[0]["kind"], "git")

    def test_git_repository_fingerprint_hot_run_skips_manifest_rewrite_when_clean(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "main.py"
            source.write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "main.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]

            first, _ = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

            with mock.patch.object(
                inspection_index.os,
                "replace",
                side_effect=AssertionError("unchanged git fingerprint manifest should not be rewritten"),
            ):
                second, _ = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

        self.assertEqual(first, second)

    def test_git_repository_fingerprint_clean_hot_run_skips_status_for_small_repo_fastpath(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "main.py"
            source.write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "main.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]

            first, _ = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)
            status_calls = 0
            original_run = inspection_index.subprocess.run

            def recording_run(args, *pargs, **kwargs):
                nonlocal status_calls
                if git_args_match(args, root, "status"):
                    status_calls += 1
                if git_args_match_legacy_clean_probe(args, root):
                    raise AssertionError("clean hot fingerprint should not use legacy clean probes")
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=recording_run):
                second, states = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

        self.assertEqual(first, second)
        self.assertEqual(states[0]["kind"], "git")
        self.assertEqual(status_calls, 0)

    def test_git_repository_fingerprint_clean_hot_run_reuses_directory_signature_fastpath_without_tree_walk(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "pkg" / "main.py"
            source.parent.mkdir(parents=True)
            source.write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]

            first, _ = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

            with mock.patch.object(
                inspection_index,
                "_git_clean_fastpath_capture_files",
                side_effect=AssertionError("clean hot fingerprint should not need tree-walk validation when directory signatures match"),
            ):
                second, states = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

        self.assertEqual(first, second)
        self.assertEqual(states[0]["kind"], "git")

    def test_git_repository_fingerprint_clean_hot_run_uses_status_probe_above_small_repo_fastpath_threshold(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            for index in range(inspection_index.SMALL_GIT_FINGERPRINT_FASTPATH_FILE_THRESHOLD + 1):
                (root / f"file_{index:03d}.py").write_text(f"value = {index}\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]

            first, _ = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)
            status_calls = 0
            original_run = inspection_index.subprocess.run

            def recording_run(args, *pargs, **kwargs):
                nonlocal status_calls
                if git_args_match(args, root, "status"):
                    status_calls += 1
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=recording_run):
                second, states = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

        self.assertEqual(first, second)
        self.assertEqual(states[0]["kind"], "git")
        self.assertEqual(status_calls, 1)

    def test_git_repository_fingerprint_repeated_unstaged_only_hot_run_skips_status_for_small_repo_fastpath(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "main.py"
            helper = root / "helper.py"
            source.write_text("value = 1\n", encoding="utf-8")
            helper.write_text("helper = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]

            inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)
            source.write_text("value = 2\n", encoding="utf-8")
            first_dirty, first_states = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)
            source.write_text("value = 3\n", encoding="utf-8")

            status_calls = 0
            original_run = inspection_index.subprocess.run

            def recording_run(args, *pargs, **kwargs):
                nonlocal status_calls
                if git_args_match(args, root, "status"):
                    status_calls += 1
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=recording_run):
                second_dirty, second_states = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

        self.assertNotEqual(first_dirty, second_dirty)
        self.assertEqual(first_states[0]["kind"], "git")
        self.assertEqual(second_states[0]["kind"], "git")
        self.assertEqual(second_states[0].get("dirty_paths"), ["main.py"])
        self.assertEqual(status_calls, 0)

    def test_git_repository_fingerprint_first_unstaged_only_run_after_clean_skips_status_for_small_repo_fastpath(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "main.py"
            helper = root / "helper.py"
            source.write_text("value = 1\n", encoding="utf-8")
            helper.write_text("helper = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]

            clean_fingerprint, clean_states = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)
            source.write_text("value = 2\n", encoding="utf-8")

            status_calls = 0
            original_run = inspection_index.subprocess.run

            def recording_run(args, *pargs, **kwargs):
                nonlocal status_calls
                if git_args_match(args, root, "status"):
                    status_calls += 1
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=recording_run):
                dirty_fingerprint, dirty_states = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

        self.assertNotEqual(clean_fingerprint, dirty_fingerprint)
        self.assertEqual(clean_states[0]["kind"], "git")
        self.assertEqual(dirty_states[0]["kind"], "git")
        self.assertEqual(dirty_states[0].get("dirty_paths"), ["main.py"])
        self.assertEqual(status_calls, 0)

    def test_git_repository_fingerprint_first_untracked_run_after_clean_skips_status_for_small_repo_fastpath(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            (root / "main.py").write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]

            clean_fingerprint, _clean_states = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)
            (root / "new.py").write_text("value = 2\n", encoding="utf-8")

            status_calls = 0
            original_run = inspection_index.subprocess.run

            def recording_run(args, *pargs, **kwargs):
                nonlocal status_calls
                if git_args_match(args, root, "status"):
                    status_calls += 1
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=recording_run):
                dirty_fingerprint, dirty_states = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

        self.assertNotEqual(clean_fingerprint, dirty_fingerprint)
        self.assertEqual(dirty_states[0]["kind"], "git")
        self.assertEqual(dirty_states[0].get("dirty_paths"), ["new.py"])
        untracked_entries = dirty_states[0].get("untracked")
        self.assertEqual(len(untracked_entries), 1)
        self.assertEqual(untracked_entries[0][0], "new.py")
        self.assertTrue(str(untracked_entries[0][1]).startswith("sha256:"))
        self.assertEqual(status_calls, 0)

    def test_git_repository_fingerprint_first_deleted_run_after_clean_skips_status_for_small_repo_fastpath(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "main.py"
            source.write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]

            clean_fingerprint, _clean_states = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)
            source.unlink()

            status_calls = 0
            original_run = inspection_index.subprocess.run

            def recording_run(args, *pargs, **kwargs):
                nonlocal status_calls
                if git_args_match(args, root, "status"):
                    status_calls += 1
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=recording_run):
                dirty_fingerprint, dirty_states = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

        self.assertNotEqual(clean_fingerprint, dirty_fingerprint)
        self.assertEqual(dirty_states[0]["kind"], "git")
        self.assertEqual(dirty_states[0].get("dirty_paths"), ["main.py"])
        self.assertEqual(status_calls, 0)

    def test_git_repository_fingerprint_repeated_unstaged_only_hot_run_preserves_correct_state_when_index_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "main.py"
            helper = root / "helper.py"
            source.write_text("value = 1\n", encoding="utf-8")
            helper.write_text("helper = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]

            inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)
            source.write_text("value = 2\n", encoding="utf-8")
            inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)
            subprocess.run(["git", "-C", str(root), "add", "main.py"], check=True)

            status_calls = 0
            original_run = inspection_index.subprocess.run

            def recording_run(args, *pargs, **kwargs):
                nonlocal status_calls
                if git_args_match(args, root, "status"):
                    status_calls += 1
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=recording_run):
                fingerprint, states = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

        self.assertTrue(str(fingerprint).startswith("sha256:"))
        self.assertEqual(states[0]["kind"], "git")
        self.assertEqual(states[0].get("dirty_paths"), ["main.py"])
        self.assertEqual(states[0].get("untracked"), [])
        self.assertEqual(status_calls, 0)

    def test_git_repository_fingerprint_first_staged_only_run_after_clean_skips_status_for_small_repo_fastpath(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "main.py"
            source.write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]

            clean_fingerprint, _clean_states = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)
            source.write_text("value = 2\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "main.py"], check=True)

            status_calls = 0
            original_run = inspection_index.subprocess.run

            def recording_run(args, *pargs, **kwargs):
                nonlocal status_calls
                if git_args_match(args, root, "status"):
                    status_calls += 1
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=recording_run):
                staged_fingerprint, staged_states = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

        self.assertNotEqual(clean_fingerprint, staged_fingerprint)
        self.assertEqual(staged_states[0]["kind"], "git")
        self.assertEqual(staged_states[0].get("dirty_paths"), ["main.py"])
        self.assertEqual(staged_states[0].get("untracked"), [])
        self.assertEqual(status_calls, 0)

    def test_git_repository_fingerprint_staged_fastpath_uses_diff_index_raw_probe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "main.py"
            source.write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]

            inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)
            source.write_text("value = 2\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "main.py"], check=True)

            diff_index_calls = 0
            cached_diff_calls = 0
            original_run = inspection_index.subprocess.run

            def recording_run(args, *pargs, **kwargs):
                nonlocal diff_index_calls, cached_diff_calls
                if git_args_match(args, root, "diff-index") and "--cached" in args and "--raw" in args:
                    diff_index_calls += 1
                if git_args_match(args, root, "diff") and "--cached" in args and "--raw" in args:
                    cached_diff_calls += 1
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=recording_run):
                fingerprint, states = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

        self.assertTrue(str(fingerprint).startswith("sha256:"))
        self.assertEqual(states[0]["kind"], "git")
        self.assertGreaterEqual(diff_index_calls, 1)
        self.assertEqual(cached_diff_calls, 0)

    def test_git_repository_fingerprint_staged_then_modified_after_add_preserves_unstaged_state_without_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "main.py"
            source.write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]

            inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)
            source.write_text("value = 2\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "main.py"], check=True)
            source.write_text("value = 3\n", encoding="utf-8")

            status_calls = 0
            original_run = inspection_index.subprocess.run

            def recording_run(args, *pargs, **kwargs):
                nonlocal status_calls
                if git_args_match(args, root, "status"):
                    status_calls += 1
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=recording_run):
                fingerprint, states = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

        self.assertTrue(str(fingerprint).startswith("sha256:"))
        self.assertEqual(states[0]["kind"], "git")
        self.assertEqual(states[0].get("dirty_paths"), ["main.py"])
        self.assertEqual(states[0].get("untracked"), [])
        self.assertNotEqual(states[0].get("unstaged"), f"sha256:{inspection_index._empty_git_status_digest()}")
        self.assertEqual(status_calls, 0)

    def test_git_repository_fingerprint_repeated_same_staged_state_reuses_cached_staged_entries_without_diff_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "main.py"
            source.write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]

            inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)
            source.write_text("value = 2\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "main.py"], check=True)
            first_fingerprint, first_states = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

            diff_index_calls = 0
            original_run = inspection_index.subprocess.run

            def recording_run(args, *pargs, **kwargs):
                nonlocal diff_index_calls
                if git_args_match(args, root, "diff-index") and "--cached" in args and "--raw" in args:
                    diff_index_calls += 1
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=recording_run):
                second_fingerprint, second_states = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

        self.assertEqual(first_fingerprint, second_fingerprint)
        self.assertEqual(first_states[0]["kind"], "git")
        self.assertEqual(second_states[0]["kind"], "git")
        self.assertEqual(second_states[0].get("dirty_paths"), ["main.py"])
        self.assertEqual(diff_index_calls, 0)

    def test_git_repository_fingerprint_repeated_staged_state_with_new_unstaged_content_skips_diff_index_and_updates_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "main.py"
            source.write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]

            inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)
            source.write_text("value = 2\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "main.py"], check=True)
            first_fingerprint, _first_states = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)
            source.write_text("value = 3\n", encoding="utf-8")

            diff_index_calls = 0
            original_run = inspection_index.subprocess.run

            def recording_run(args, *pargs, **kwargs):
                nonlocal diff_index_calls
                if git_args_match(args, root, "diff-index") and "--cached" in args and "--raw" in args:
                    diff_index_calls += 1
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=recording_run):
                second_fingerprint, second_states = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

        self.assertNotEqual(first_fingerprint, second_fingerprint)
        self.assertEqual(second_states[0]["kind"], "git")
        self.assertEqual(second_states[0].get("dirty_paths"), ["main.py"])
        self.assertNotEqual(second_states[0].get("unstaged"), f"sha256:{inspection_index._empty_git_status_digest()}")
        self.assertEqual(diff_index_calls, 0)

    def test_git_repository_fingerprint_unstaged_only_change_skips_index_blob_lookup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "main.py"
            source.write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "main.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            source.write_text("value = 2\n", encoding="utf-8")
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]

            with mock.patch.object(
                inspection_index,
                "_git_index_blob_oids",
                side_effect=AssertionError(
                    "unstaged-only fingerprint should not load index blob oids"
                ),
            ):
                fingerprint, states = inspection_index.repository_fingerprint(
                    discovered, cache_dir=cache_dir
                )

        self.assertTrue(str(fingerprint).startswith("sha256:"))
        self.assertEqual(states[0]["kind"], "git")

    def test_git_repository_fingerprint_hot_run_uses_clean_probe_when_local_clean_manifest_matches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "main.py"
            source.write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "main.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]

            first, _ = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

            original_run = inspection_index.subprocess.run
            def guarded_run(args, *pargs, **kwargs):
                if git_args_match_legacy_clean_probe(args, root):
                    raise AssertionError("clean local git fingerprint manifest should avoid legacy clean probes")
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=guarded_run):
                second, states = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

        self.assertEqual(first, second)
        self.assertEqual(states[0]["kind"], "git")

    def test_git_repository_fingerprint_ignores_broker_owned_cache_path_on_clean_hot_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "repo-inspection-cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "main.py"
            source.write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "main.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]

            before, _ = inspection_index.repository_fingerprint(
                discovered,
                excluded_paths=[cache_dir],
                cache_dir=cache_dir,
            )
            cache_dir.mkdir(exist_ok=True)
            (cache_dir / "artifact.json").write_text("{\"cache\":true}\n", encoding="utf-8")

            original_run = inspection_index.subprocess.run

            def guarded_run(args, *pargs, **kwargs):
                if git_args_match_legacy_clean_probe(args, root):
                    raise AssertionError("excluded cache path should not force legacy clean probes on hot run")
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=guarded_run):
                after, states = inspection_index.repository_fingerprint(
                    discovered,
                    excluded_paths=[cache_dir],
                    cache_dir=cache_dir,
                )

        self.assertEqual(before, after)
        self.assertEqual(states[0]["kind"], "git")

    def test_git_repository_fingerprint_reuses_shared_manifest_across_local_cache_roots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            shared_cache_dir = temp_root / "shared-cache"
            first_cache_dir = temp_root / "cache-one"
            second_cache_dir = temp_root / "cache-two"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "main.py"
            source.write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "main.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]

            with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
                first, _ = inspection_index.repository_fingerprint(discovered, cache_dir=first_cache_dir)

                original_run = inspection_index.subprocess.run

                def guarded_run(args, *pargs, **kwargs):
                    if git_args_match_legacy_clean_probe(args, root):
                        raise AssertionError("shared git fingerprint manifest should avoid legacy clean probes on fresh local cache")
                    return original_run(args, *pargs, **kwargs)

                with mock.patch.object(inspection_index.subprocess, "run", side_effect=guarded_run):
                    second, states = inspection_index.repository_fingerprint(discovered, cache_dir=second_cache_dir)

        self.assertEqual(first, second)
        self.assertEqual(states[0]["kind"], "git")
        manifest_path = inspection_index._git_fingerprint_manifest_path(second_cache_dir, root, [])
        self.assertFalse(manifest_path.exists())

    def test_prepare_prefetched_state_reuses_shared_git_fingerprint_manifest_across_local_cache_roots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            shared_cache_dir = temp_root / "shared-cache"
            output_one = temp_root / "out-one"
            output_two = temp_root / "out-two"
            output_one.mkdir()
            output_two.mkdir()
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "main.py"
            source.write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "main.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]

            with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
                first = inspection_hotpath.prepare_prefetched_state(
                    discovered,
                    "trace main",
                    mode="evidence",
                    constraints={},
                    task_params={},
                    execution_plan={"repo_inspection_cache_path": str(temp_root / "cache-one")},
                    output_dir=output_one,
                )

                original_run = subprocess.run

                def guarded_run(args, *pargs, **kwargs):
                    if git_args_match_legacy_clean_probe(args, root):
                        raise AssertionError("shared git fingerprint manifest should avoid legacy clean probes during prefetch")
                    return original_run(args, *pargs, **kwargs)

                with mock.patch.object(subprocess, "run", side_effect=guarded_run):
                    second = inspection_hotpath.prepare_prefetched_state(
                        discovered,
                        "trace main",
                        mode="evidence",
                        constraints={},
                        task_params={},
                        execution_plan={"repo_inspection_cache_path": str(temp_root / "cache-two")},
                        output_dir=output_two,
                    )

        self.assertEqual(first["repository_state_fingerprint"], second["repository_state_fingerprint"])
        manifest_path = inspection_hotpath._git_fingerprint_manifest_path(temp_root / "cache-two", root, [])
        self.assertFalse(manifest_path.exists())

    def test_prepare_prefetched_state_skips_git_status_when_local_clean_manifest_matches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            output_dir = temp_root / "out"
            output_dir.mkdir()
            cache_dir = temp_root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "main.py"
            source.write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "main.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]

            first = inspection_hotpath.prepare_prefetched_state(
                discovered,
                "trace main",
                mode="evidence",
                constraints={},
                task_params={},
                execution_plan={"repo_inspection_cache_path": str(cache_dir)},
                output_dir=output_dir,
            )

            original_run = subprocess.run

            def guarded_run(args, *pargs, **kwargs):
                if git_args_match_legacy_clean_probe(args, root):
                    raise AssertionError("clean local prefetch manifest should avoid legacy clean probes")
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(subprocess, "run", side_effect=guarded_run):
                second = inspection_hotpath.prepare_prefetched_state(
                    discovered,
                    "trace main",
                    mode="evidence",
                    constraints={},
                    task_params={},
                    execution_plan={"repo_inspection_cache_path": str(cache_dir)},
                    output_dir=output_dir,
                )

        self.assertEqual(first["repository_state_fingerprint"], second["repository_state_fingerprint"])

    def test_repository_fingerprint_skips_git_status_when_local_clean_manifest_matches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            cache_dir = temp_root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "main.py"
            source.write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "main.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]

            first, _ = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

            original_run = subprocess.run

            def guarded_run(args, *pargs, **kwargs):
                if git_args_match_legacy_clean_probe(args, root):
                    raise AssertionError("clean local fingerprint manifest should avoid legacy clean probes")
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(subprocess, "run", side_effect=guarded_run):
                second, states = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

        self.assertEqual(first, second)
        self.assertEqual(states[0]["kind"], "git")

    def test_hotpath_repository_fingerprint_reuses_default_git_probe_cache_without_explicit_cache_argument(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "main.py"
            source.write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "main.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]

            inspection_hotpath._DEFAULT_GIT_PROBE_CACHE.clear()
            first, _ = inspection_hotpath.repository_fingerprint(discovered, cache_dir=cache_dir)

            original_run = subprocess.run

            def guarded_run(args, *pargs, **kwargs):
                if git_args_match_legacy_clean_probe(args, root):
                    raise AssertionError("hotpath default git probe cache should avoid legacy clean probes")
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(subprocess, "run", side_effect=guarded_run):
                second, states = inspection_hotpath.repository_fingerprint(discovered, cache_dir=cache_dir)

        self.assertEqual(first, second)
        self.assertEqual(states[0]["kind"], "git")

    def test_hotpath_repository_fingerprint_default_cache_invalidates_stale_worktree_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "main.py"
            source.write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "main.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]

            inspection_hotpath._DEFAULT_GIT_PROBE_CACHE.clear()
            clean_fingerprint, clean_states = inspection_hotpath.repository_fingerprint(
                discovered,
                cache_dir=cache_dir,
            )
            source.write_text("value = 2\n", encoding="utf-8")
            dirty_fingerprint, dirty_states = inspection_hotpath.repository_fingerprint(
                discovered,
                cache_dir=cache_dir,
            )

        self.assertNotEqual(clean_fingerprint, dirty_fingerprint)
        self.assertEqual(dirty_states[0]["kind"], "git")
        self.assertNotEqual(
            str(dirty_states[0].get("unstaged") or ""),
            str(clean_states[0].get("unstaged") or ""),
        )

    def test_prepare_prefetched_state_clean_hot_run_avoids_name_only_diff_probe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "repo"
            root.mkdir()
            output_dir = temp_root / "out"
            output_dir.mkdir()
            cache_dir = temp_root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "main.py"
            source.write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "main.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]

            inspection_hotpath.prepare_prefetched_state(
                discovered,
                "trace main",
                mode="evidence",
                constraints={},
                task_params={},
                execution_plan={"repo_inspection_cache_path": str(cache_dir)},
                output_dir=output_dir,
            )

            original_run = subprocess.run

            def guarded_run(args, *pargs, **kwargs):
                if git_args_match(args, root, "diff") and "--name-only" in args:
                    raise AssertionError("clean prefetch hot run should avoid name-only diff probe")
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(subprocess, "run", side_effect=guarded_run):
                second = inspection_hotpath.prepare_prefetched_state(
                    discovered,
                    "trace main",
                    mode="evidence",
                    constraints={},
                    task_params={},
                    execution_plan={"repo_inspection_cache_path": str(cache_dir)},
                    output_dir=output_dir,
                )

        self.assertTrue(str(second["repository_state_fingerprint"]).startswith("sha256:"))

    def test_run_inspection_skips_gpu_service_import_when_registry_not_configured(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "out"
            output_dir.mkdir()
            (root / "worker.py").write_text("def retry_job(job_id):\n    return job_id\n", encoding="utf-8")
            discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]

            with mock.patch.object(
                inspection_pipeline,
                "_gpu_symbols",
                side_effect=AssertionError("gpu client module should not load when registry is not configured"),
            ):
                payload = inspection_pipeline.run_inspection(
                    discovered,
                    "trace retry_job",
                    mode="evidence",
                    constraints={},
                    task_params={},
                    execution_plan={},
                    output_dir=output_dir,
                    services=None,
                    client_factory=SemanticDiagnosticsFactory(),
                )["payload"]

        self.assertLess(float(payload["retrieval"]["setup_timings_ms"]["load_services_ms"]), 1.0)

    def test_main_run_repo_inspection_job_passes_prefetched_state_to_pipeline(self):
        discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": Path("/tmp/repo")}]
        prefetched_state = {
            "repository_state_fingerprint": "git:test",
            "fingerprint_state": [{"kind": "git", "head": "abc"}],
            "cache_dir": Path("/tmp/cache"),
            "excluded": set(),
            "git_probe_cache": {"scoped_status_output": {}},
        }
        payload = {
            "mode": "evidence",
            "query": "trace main",
            "evidence": [],
            "quality": {"result": "evidence_only", "answer_ready": False},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            heartbeat_path = output_dir / "heartbeat.json"
            with mock.patch.object(rag_main, "write_repo_inspection_artifacts", return_value=[]), mock.patch.object(
                rag_main, "write_json"
            ), mock.patch.object(rag_main, "emit_heartbeat"), mock.patch.object(
                rag_main, "highest_classification", return_value="internal"
            ), mock.patch.object(
                inspection_hotpath, "prepare_prefetched_state", return_value=prefetched_state
            ) as prepare_mock, mock.patch.object(
                inspection_hotpath, "cached_lexical_fallback_from_context", return_value=None
            ) as cached_mock, mock.patch.object(
                rag_main,
                "run_inspection",
                return_value={"payload": payload, "artifact_payloads": {}},
            ) as run_mock:
                exit_code = rag_main.run_repo_inspection_job(
                    {"job_id": "job-1"},
                    {"query": "trace main", "mode": "evidence"},
                    {},
                    "trace main",
                    "evidence",
                    discovered,
                    {"repo_inspection_cache_path": str(output_dir / "cache")},
                    output_dir,
                    heartbeat_path,
                )

        self.assertEqual(exit_code, 0)
        prepare_mock.assert_called_once()
        cached_mock.assert_called_once_with(prefetched_state)
        self.assertIs(run_mock.call_args.kwargs["prefetched_state"], prefetched_state)

    def test_main_run_repo_inspection_job_returns_cached_lexical_fallback_without_pipeline(self):
        discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": Path("/tmp/repo")}]
        prefetched_state = {"cached_query_stage": {"query": "trace main"}}
        cached_run = {
            "payload": {
                "mode": "evidence",
                "query": "trace main",
                "evidence": [{"id": "chunk-1"}],
                "quality": {"result": "evidence_only", "answer_ready": False},
            },
            "artifact_payloads": {"runtime_diagnostics": {"query_stage_cache_hit": True}},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            heartbeat_path = output_dir / "heartbeat.json"
            with mock.patch.object(inspection_hotpath, "prepare_prefetched_state", return_value=prefetched_state), mock.patch.object(
                inspection_hotpath, "cached_lexical_fallback_from_context", return_value=cached_run
            ), mock.patch.object(
                rag_main, "write_repo_inspection_artifacts", return_value=[]
            ), mock.patch.object(
                rag_main, "write_json"
            ), mock.patch.object(
                rag_main, "highest_classification", return_value="internal"
            ), mock.patch.object(
                rag_main, "run_inspection", side_effect=AssertionError("cached lexical fallback should skip pipeline")
            ), mock.patch.object(rag_main, "emit_heartbeat") as heartbeat_mock:
                exit_code = rag_main.run_repo_inspection_job(
                    {"job_id": "job-2"},
                    {"query": "trace main", "mode": "evidence"},
                    {},
                    "trace main",
                    "evidence",
                    discovered,
                    {"repo_inspection_cache_path": str(output_dir / "cache")},
                    output_dir,
                    heartbeat_path,
                )

        self.assertEqual(exit_code, 0)
        final_call = heartbeat_mock.call_args_list[-1]
        self.assertEqual(final_call.args[2], "completed")
        self.assertEqual(final_call.args[5], "Repository inspection completed from persisted lexical-fallback cache")

    def test_main_run_repo_inspection_job_reuses_precomputed_cached_lexical_fallback_run(self):
        discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": Path("/tmp/repo")}]
        cached_run = {
            "payload": {
                "mode": "evidence",
                "query": "trace main",
                "evidence": [{"id": "chunk-1"}],
                "quality": {"result": "evidence_only", "answer_ready": False},
            },
            "artifact_payloads": {"runtime_diagnostics": {"query_stage_cache_hit": True}},
        }
        prefetched_state = {
            "cached_query_stage": {"query": "trace main"},
            "cached_lexical_fallback_run": cached_run,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            heartbeat_path = output_dir / "heartbeat.json"
            with mock.patch.object(
                inspection_hotpath, "prepare_prefetched_state", return_value=prefetched_state
            ), mock.patch.object(
                inspection_hotpath,
                "cached_lexical_fallback_from_context",
                side_effect=AssertionError("precomputed cached lexical fallback should be reused"),
            ), mock.patch.object(
                rag_main, "write_repo_inspection_artifacts", return_value=[]
            ), mock.patch.object(
                rag_main, "write_json"
            ), mock.patch.object(
                rag_main, "highest_classification", return_value="internal"
            ), mock.patch.object(
                rag_main, "run_inspection", side_effect=AssertionError("cached lexical fallback should skip pipeline")
            ), mock.patch.object(rag_main, "emit_heartbeat") as heartbeat_mock:
                exit_code = rag_main.run_repo_inspection_job(
                    {"job_id": "job-precomputed"},
                    {"query": "trace main", "mode": "evidence"},
                    {},
                    "trace main",
                    "evidence",
                    discovered,
                    {"repo_inspection_cache_path": str(output_dir / "cache")},
                    output_dir,
                    heartbeat_path,
                )

        self.assertEqual(exit_code, 0)
        final_call = heartbeat_mock.call_args_list[-1]
        self.assertEqual(final_call.args[2], "completed")

    def test_main_write_json_if_changed_skips_identical_rewrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "runtime_diagnostics.json"
            payload = {"query_stage_cache_hit": True, "attempts": []}
            rag_main.write_json_if_changed(path, payload)
            first_stat = path.stat()
            first_text = path.read_text(encoding="utf-8")
            time.sleep(0.01)
            rag_main.write_json_if_changed(path, payload)
            second_stat = path.stat()
            second_text = path.read_text(encoding="utf-8")

        self.assertEqual(first_text, second_text)
        self.assertEqual(first_stat.st_mtime_ns, second_stat.st_mtime_ns)
        self.assertEqual(first_stat.st_ino, second_stat.st_ino)

    def test_main_write_repo_inspection_artifacts_skips_identical_rewrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            artifact_payloads = {
                "runtime_diagnostics": {"query_stage_cache_hit": True},
                "retrieval_result": {"fingerprint": "git:test", "selected": []},
            }
            rag_main.write_repo_inspection_artifacts(output_dir, artifact_payloads, "internal")
            runtime_path = output_dir / "runtime_diagnostics.json"
            retrieval_path = output_dir / "retrieval_result.json"
            first_runtime = runtime_path.stat()
            first_retrieval = retrieval_path.stat()
            time.sleep(0.01)
            rag_main.write_repo_inspection_artifacts(output_dir, artifact_payloads, "internal")
            second_runtime = runtime_path.stat()
            second_retrieval = retrieval_path.stat()

        self.assertEqual(first_runtime.st_mtime_ns, second_runtime.st_mtime_ns)
        self.assertEqual(first_runtime.st_ino, second_runtime.st_ino)
        self.assertEqual(first_retrieval.st_mtime_ns, second_retrieval.st_mtime_ns)
        self.assertEqual(first_retrieval.st_ino, second_retrieval.st_ino)

    def test_main_write_repo_inspection_artifacts_omits_unreleased_trace_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            artifact_payloads = {
                "evidence_pack": {"evidence": []},
                "runtime_diagnostics": {"query_stage_cache_hit": True},
                "retrieval_result": {"fingerprint": "git:test", "selected": []},
            }

            artifacts = rag_main.write_repo_inspection_artifacts(
                output_dir,
                artifact_payloads,
                "internal",
                include_full_trace=False,
            )
            self.assertTrue((output_dir / "evidence_pack.json").exists())
            self.assertFalse((output_dir / "runtime_diagnostics.json").exists())
            self.assertFalse((output_dir / "retrieval_result.json").exists())
            self.assertEqual([artifact["artifact_type"] for artifact in artifacts], ["evidence_pack"])

    def test_git_fingerprint_skips_untracked_symlink_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            root = base / "repo"
            root.mkdir()
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            outside = base / "outside.py"
            outside.write_text("value = 1\n", encoding="utf-8")
            (root / "linked.py").symlink_to(outside)
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]
            before, _ = inspection_index.repository_fingerprint(discovered)
            outside.write_text("value = 2\n", encoding="utf-8")
            after, states = inspection_index.repository_fingerprint(discovered)

        self.assertEqual(before, after)
        self.assertEqual(states[0]["untracked"], [])

    def test_git_fingerprint_ignores_broker_live_test_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "main.py"
            source.write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "main.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]
            before, _ = inspection_index.repository_fingerprint(discovered)
            artifact_dir = root / ".broker-live-tests" / "run-1"
            artifact_dir.mkdir(parents=True)
            (artifact_dir / "result.json").write_text('{"ok":true}\n', encoding="utf-8")
            after, states = inspection_index.repository_fingerprint(discovered)

        self.assertEqual(before, after)
        self.assertEqual(states[0]["untracked"], [])

    def test_git_fingerprint_ignores_slurm_output_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "main.py"
            source.write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "main.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]
            before, _ = inspection_index.repository_fingerprint(discovered)
            (root / "slurm-123.out").write_text("noise\n", encoding="utf-8")
            after, states = inspection_index.repository_fingerprint(discovered)

        self.assertEqual(before, after)
        self.assertEqual(states[0]["untracked"], [])

    def test_git_fingerprint_tracks_unstaged_staged_and_untracked_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "main.py"
            source.write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "main.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]
            clean, _ = inspection_index.repository_fingerprint(discovered)
            source.write_text("value = 2\n", encoding="utf-8")
            unstaged, _ = inspection_index.repository_fingerprint(discovered)
            subprocess.run(["git", "-C", str(root), "add", "main.py"], check=True)
            staged, _ = inspection_index.repository_fingerprint(discovered)
            untracked_path = root / "new.py"
            untracked_path.write_text("value = 3\n", encoding="utf-8")
            untracked_a, _ = inspection_index.repository_fingerprint(discovered)
            untracked_path.write_text("value = 4\n", encoding="utf-8")
            untracked_b, _ = inspection_index.repository_fingerprint(discovered)

        self.assertEqual(len({clean, unstaged, staged, untracked_a, untracked_b}), 5)

    def test_git_fingerprint_cache_dir_still_tracks_unstaged_content_when_status_shape_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "main.py"
            source.write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "main.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]

            source.write_text("value = 2\n", encoding="utf-8")
            first, _ = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)
            source.write_text("value = 3\n", encoding="utf-8")
            second, _ = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

        self.assertNotEqual(first, second)

    def test_git_fingerprint_cache_dir_still_tracks_untracked_content_when_status_shape_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "main.py"
            source.write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "main.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]

            untracked = root / "new.py"
            untracked.write_text("value = 2\n", encoding="utf-8")
            first, _ = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)
            untracked.write_text("value = 3\n", encoding="utf-8")
            second, _ = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

        self.assertNotEqual(first, second)

    def test_git_fingerprint_reuses_unchanged_dirty_worktree_signature_from_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            one = root / "one.py"
            two = root / "two.py"
            one.write_text("value = 1\n", encoding="utf-8")
            two.write_text("value = 2\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]

            one.write_text("value = 10\n", encoding="utf-8")
            two.write_text("value = 20\n", encoding="utf-8")
            inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)
            one.write_text("value = 11\n", encoding="utf-8")

            original_read_bytes = Path.read_bytes

            def guarded_read_bytes(path_self):
                if path_self.resolve(strict=False) == two.resolve(strict=False):
                    raise AssertionError("unchanged dirty file should reuse cached worktree signature")
                return original_read_bytes(path_self)

            with mock.patch.object(Path, "read_bytes", autospec=True, side_effect=guarded_read_bytes):
                fingerprint, states = inspection_index.repository_fingerprint(discovered, cache_dir=cache_dir)

        self.assertTrue(str(fingerprint).startswith("sha256:"))
        self.assertEqual(states[0]["kind"], "git")

    def test_git_repository_fingerprint_dirty_run_avoids_binary_git_diff(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            source = root / "main.py"
            source.write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "main.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            source.write_text("value = 2\n", encoding="utf-8")
            discovered = [{"id": "i", "type": "repo", "classification": "internal", "path": root}]

            original_run = inspection_index.subprocess.run

            def guarded_run(args, *pargs, **kwargs):
                if git_args_match(args, root, "diff"):
                    raise AssertionError("dirty fingerprint should not rerun git diff")
                return original_run(args, *pargs, **kwargs)

            with mock.patch.object(inspection_index.subprocess, "run", side_effect=guarded_run):
                fingerprint, states = inspection_index.repository_fingerprint(discovered)

        self.assertTrue(str(fingerprint).startswith("sha256:"))
        self.assertEqual(states[0]["kind"], "git")

    def test_sqlite_fts_prefers_exact_identifier_and_path(self):
        chunks = [
            {
                "chunk_id": "a",
                "path": "broker/pkg/mcp/server.go",
                "language": "go",
                "symbol": "InspectRepo",
                "line_start": 1,
                "line_end": 3,
                "content": "func InspectRepo() {}",
                "content_hash": "sha256:a",
                "token_estimate": 5,
            },
            {
                "chunk_id": "b",
                "path": "workers/rag/main.py",
                "language": "python",
                "symbol": "worker",
                "line_start": 1,
                "line_end": 3,
                "content": "def worker(): pass",
                "content_hash": "sha256:b",
                "token_estimate": 5,
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path, _, _ = inspection_index.ensure_lexical_index(chunks, temp_dir, "sha256:test")
            ranked = inspection_index.lexical_search(path, 'Find InspectRepo in "broker/pkg/mcp/server.go"', chunks)
        self.assertEqual(ranked[0]["chunk_id"], "a")

    def test_mcp_call_chain_prior_ranks_mcp_and_service_above_rag_worker(self):
        specs = [
            ("mcp", "broker/pkg/mcp/server.go", "go", "InspectRepo", "func InspectRepo() { service.Submit() }"),
            ("service", "broker/pkg/service/service_execution.go", "go", "execute", "func execute() { runWorker() }"),
            ("worker", "workers/rag-compression/main.py", "python", "main", "def main(): # inspect_repo call chain service mcp\n pass"),
            ("docs", "docs/golden_queries.md", "markdown", "", "inspect_repo MCP call chain service worker"),
            ("test", "tests/acceptance/test_evaluate.py", "python", "test_query", "inspect_repo MCP call chain service worker"),
        ]
        chunks = [
            {
                "chunk_id": chunk_id,
                "path": path,
                "language": language,
                "symbol": symbol,
                "line_start": 1,
                "line_end": 3,
                "content": content,
                "content_hash": f"sha256:{chunk_id}",
                "token_estimate": 10,
            }
            for chunk_id, path, language, symbol, content in specs
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path, _, _ = inspection_index.ensure_lexical_index(chunks, temp_dir, "sha256:mcp")
            ranked = inspection_index.lexical_search(
                path,
                "Trace the inspect_repo MCP call chain through the service to the worker",
                chunks,
            )
        positions = {item["chunk_id"]: item["rank"] for item in ranked}
        self.assertLess(positions["mcp"], positions["worker"])
        self.assertLess(positions["service"], positions["worker"])

    def test_lexical_search_reuses_query_independent_helper_cache(self):
        chunks = [
            {
                "chunk_id": "a",
                "path": "broker/pkg/mcp/server.go",
                "language": "go",
                "symbol": "InspectRepo",
                "line_start": 1,
                "line_end": 3,
                "content": "func InspectRepo() {}",
                "content_hash": "sha256:a",
                "token_estimate": 5,
            },
            {
                "chunk_id": "b",
                "path": "workers/rag/main.py",
                "language": "python",
                "symbol": "worker",
                "line_start": 1,
                "line_end": 3,
                "content": "def worker(): pass",
                "content_hash": "sha256:b",
                "token_estimate": 5,
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path, _, _ = inspection_index.ensure_lexical_index(chunks, temp_dir, "sha256:test")
            inspection_index._LEXICAL_HELPER_CACHE.clear()
            with mock.patch.object(
                inspection_index,
                "_build_lexical_helper",
                wraps=inspection_index._build_lexical_helper,
            ) as builder:
                inspection_index.lexical_search(path, "InspectRepo", chunks)
                inspection_index.lexical_search(path, "worker", chunks)
                self.assertEqual(builder.call_count, 1)

    def test_lexical_helper_compacts_chunks_by_path_to_chunk_ids(self):
        chunks = [
            {
                "chunk_id": "a",
                "path": "broker/pkg/mcp/server.go",
                "language": "go",
                "symbol": "InspectRepo",
                "line_start": 1,
                "line_end": 3,
                "content": "func InspectRepo() {}",
                "content_hash": "sha256:a",
                "token_estimate": 5,
            },
            {
                "chunk_id": "b",
                "path": "broker/pkg/mcp/server.go",
                "language": "go",
                "symbol": "InspectRepoInner",
                "line_start": 4,
                "line_end": 6,
                "content": "func InspectRepoInner() {}",
                "content_hash": "sha256:b",
                "token_estimate": 5,
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path, _, _ = inspection_index.ensure_lexical_index(chunks, temp_dir, "sha256:test")
            cache_key = inspection_index.lexical_cache_key(path, chunks)
            inspection_index._LEXICAL_HELPER_CACHE.clear()
            inspection_index.lexical_search(path, "InspectRepo", chunks, cache_key=cache_key)
            helper = inspection_index._LEXICAL_HELPER_CACHE[cache_key]

        self.assertEqual(helper["chunks_by_path"]["broker/pkg/mcp/server.go"], ["a", "b"])

    def test_lexical_search_reuses_term_match_caches_for_overlapping_queries(self):
        chunks = [
            {
                "chunk_id": "a",
                "path": "broker/pkg/mcp/server.go",
                "language": "go",
                "symbol": "InspectRepo",
                "line_start": 1,
                "line_end": 3,
                "content": "func InspectRepo() { service.Dispatch() }",
                "content_hash": "sha256:a",
                "token_estimate": 5,
            },
            {
                "chunk_id": "b",
                "path": "broker/pkg/service/service_execution.go",
                "language": "go",
                "symbol": "Dispatch",
                "line_start": 1,
                "line_end": 3,
                "content": "func Dispatch() { worker.Run() }",
                "content_hash": "sha256:b",
                "token_estimate": 5,
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path, _, _ = inspection_index.ensure_lexical_index(chunks, temp_dir, "sha256:test")
            inspection_index._LEXICAL_HELPER_CACHE.clear()
            cache_key = inspection_index.lexical_cache_key(path, chunks)
            inspection_index.lexical_search(path, "InspectRepo service", chunks, cache_key=cache_key)
            helper = inspection_index._LEXICAL_HELPER_CACHE[cache_key]
            first_chunk_terms = set(helper["term_chunk_ids_cache"])
            first_file_terms = set(helper["term_file_paths_cache"])
            first_path_terms = set(helper["path_term_paths_cache"])

            inspection_index.lexical_search(path, "InspectRepo dispatch", chunks, cache_key=cache_key)

            self.assertEqual(
                set(helper["term_chunk_ids_cache"]) - first_chunk_terms,
                {"dispatch"},
            )
            self.assertEqual(
                set(helper["term_file_paths_cache"]) - first_file_terms,
                {"dispatch"},
            )
            self.assertEqual(
                set(helper["path_term_paths_cache"]) - first_path_terms,
                {"dispatch"},
            )

    def test_wrapper_reference_bonus_still_surfaces_shell_launcher(self):
        chunks = [
            {
                "chunk_id": "wrapper",
                "path": "scripts/run.sh",
                "language": "shell",
                "symbol": "run",
                "line_start": 1,
                "line_end": 2,
                "content": "python workers/rag/main.py\n",
                "content_hash": "sha256:wrapper",
                "token_estimate": 5,
            },
            {
                "chunk_id": "target",
                "path": "workers/rag/main.py",
                "language": "python",
                "symbol": "main",
                "line_start": 1,
                "line_end": 3,
                "content": "def main():\n    return 'entrypoint'\n",
                "content_hash": "sha256:target",
                "token_estimate": 5,
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path, _, _ = inspection_index.ensure_lexical_index(chunks, temp_dir, "sha256:test")
            ranked = inspection_index.lexical_search(path, "find wrapper script entrypoint", chunks)
        positions = {item["chunk_id"]: item["rank"] for item in ranked}
        self.assertLessEqual(positions["wrapper"], positions["target"])

    def test_lexical_search_still_deboosts_docs_without_explicit_path_match(self):
        chunks = [
            {
                "chunk_id": "doc",
                "path": "docs/architecture.md",
                "language": "markdown",
                "symbol": "",
                "line_start": 1,
                "line_end": 3,
                "content": "retry_job service architecture entrypoint",
                "content_hash": "sha256:doc",
                "token_estimate": 5,
            },
            {
                "chunk_id": "code",
                "path": "broker/pkg/service/retry.go",
                "language": "go",
                "symbol": "retry_job",
                "line_start": 1,
                "line_end": 3,
                "content": "func retry_job() {}",
                "content_hash": "sha256:code",
                "token_estimate": 5,
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path, _, _ = inspection_index.ensure_lexical_index(chunks, temp_dir, "sha256:test")
            ranked = inspection_index.lexical_search(path, "retry_job service architecture", chunks)
        self.assertEqual(ranked[0]["chunk_id"], "code")

    def test_lexical_search_still_deboosts_examples_without_example_intent(self):
        chunks = [
            {
                "chunk_id": "example",
                "path": "examples/retry.go",
                "language": "go",
                "symbol": "retry_job",
                "line_start": 1,
                "line_end": 3,
                "content": "func retry_job() {}",
                "content_hash": "sha256:example",
                "token_estimate": 5,
            },
            {
                "chunk_id": "code",
                "path": "broker/pkg/service/retry.go",
                "language": "go",
                "symbol": "retry_job",
                "line_start": 1,
                "line_end": 3,
                "content": "func retry_job() {}",
                "content_hash": "sha256:code",
                "token_estimate": 5,
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path, _, _ = inspection_index.ensure_lexical_index(chunks, temp_dir, "sha256:test")
            ranked = inspection_index.lexical_search(path, "retry_job service", chunks)
        self.assertEqual(ranked[0]["chunk_id"], "code")

    def test_lexical_search_reuses_exact_query_result_cache(self):
        chunks = [
            {
                "chunk_id": "a",
                "path": "broker/pkg/mcp/server.go",
                "language": "go",
                "symbol": "InspectRepo",
                "line_start": 1,
                "line_end": 3,
                "content": "func InspectRepo() {}",
                "content_hash": "sha256:a",
                "token_estimate": 5,
            },
            {
                "chunk_id": "b",
                "path": "workers/rag/main.py",
                "language": "python",
                "symbol": "worker",
                "line_start": 1,
                "line_end": 3,
                "content": "def worker(): pass",
                "content_hash": "sha256:b",
                "token_estimate": 5,
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path, _, _ = inspection_index.ensure_lexical_index(chunks, temp_dir, "sha256:test")
            inspection_index._LEXICAL_RESULT_CACHE.clear()
            with mock.patch.object(
                inspection_index,
                "query_features",
                wraps=inspection_index.query_features,
            ) as feature_builder:
                first = inspection_index.lexical_search(path, "InspectRepo", chunks)
                second = inspection_index.lexical_search(path, "InspectRepo", chunks)
                self.assertEqual(feature_builder.call_count, 1)
                self.assertEqual(first, second)

    def test_lexical_search_uses_provided_cache_key_without_restatting_index(self):
        chunks = [
            {
                "chunk_id": "a",
                "path": "broker/pkg/mcp/server.go",
                "language": "go",
                "symbol": "InspectRepo",
                "line_start": 1,
                "line_end": 3,
                "content": "func InspectRepo() {}",
                "content_hash": "sha256:a",
                "token_estimate": 5,
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path, _, _ = inspection_index.ensure_lexical_index(chunks, temp_dir, "sha256:test")
            cache_key = inspection_index.lexical_cache_key(path, chunks)
            with mock.patch.object(Path, "stat", autospec=True, side_effect=AssertionError("stat should not be called")):
                ranked = inspection_index.lexical_search(path, "InspectRepo", chunks, cache_key=cache_key)
        self.assertEqual(ranked[0]["chunk_id"], "a")

    def test_lexical_search_skips_sqlite_fts_for_small_corpus(self):
        chunks = [
            {
                "chunk_id": "a",
                "path": "broker/pkg/mcp/server.go",
                "language": "go",
                "symbol": "InspectRepo",
                "line_start": 1,
                "line_end": 3,
                "content": "func InspectRepo() {}",
                "content_hash": "sha256:a",
                "token_estimate": 5,
            },
            {
                "chunk_id": "b",
                "path": "workers/rag/main.py",
                "language": "python",
                "symbol": "worker",
                "line_start": 1,
                "line_end": 3,
                "content": "def worker(): pass",
                "content_hash": "sha256:b",
                "token_estimate": 5,
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path, _, _ = inspection_index.ensure_lexical_index(chunks, temp_dir, "sha256:test")
            with mock.patch.object(
                inspection_index.sqlite3,
                "connect",
                side_effect=AssertionError("small-corpus lexical search should not open sqlite"),
            ):
                ranked = inspection_index.lexical_search(path, "InspectRepo", chunks)
        self.assertEqual(ranked[0]["chunk_id"], "a")

    def test_lexical_search_uses_provided_query_features_without_reparsing(self):
        chunks = [
            {
                "chunk_id": "a",
                "path": "broker/pkg/mcp/server.go",
                "language": "go",
                "symbol": "InspectRepo",
                "line_start": 1,
                "line_end": 3,
                "content": "func InspectRepo() {}",
                "content_hash": "sha256:a",
                "token_estimate": 5,
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path, _, _ = inspection_index.ensure_lexical_index(chunks, temp_dir, "sha256:test")
            features = inspection_index.query_features("InspectRepo")
            with mock.patch.object(
                inspection_index,
                "query_features",
                side_effect=AssertionError("query_features should not be called"),
            ):
                ranked = inspection_index.lexical_search(path, "InspectRepo", chunks, features=features)
        self.assertEqual(ranked[0]["chunk_id"], "a")

    def test_query_features_reuses_exact_query_cache(self):
        inspection_index._QUERY_FEATURE_CACHE.clear()
        with mock.patch.object(
            inspection_index,
            "identifier_pieces",
            wraps=inspection_index.identifier_pieces,
        ) as splitter:
            first = inspection_index.query_features('Find InspectRepo in "broker/pkg/mcp/server.go"')
            second = inspection_index.query_features('Find InspectRepo in "broker/pkg/mcp/server.go"')
        self.assertEqual(first, second)
        self.assertGreater(splitter.call_count, 0)
        # Second call should be cached and avoid re-running the splitter for every token.
        self.assertLess(splitter.call_count, 10)

    def test_lexical_search_works_with_dehydrated_chunks_via_sqlite_index(self):
        chunks = [
            {
                "chunk_id": "a",
                "path": "broker/pkg/mcp/server.go",
                "language": "go",
                "symbol": "InspectRepo",
                "line_start": 1,
                "line_end": 3,
                "content": "func InspectRepo() {}",
                "content_hash": "sha256:a",
                "token_estimate": 5,
            },
            {
                "chunk_id": "b",
                "path": "workers/rag/main.py",
                "language": "python",
                "symbol": "worker",
                "line_start": 1,
                "line_end": 3,
                "content": "def worker(): pass",
                "content_hash": "sha256:b",
                "token_estimate": 5,
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path, _, _ = inspection_index.ensure_lexical_index(chunks, temp_dir, "sha256:test")
            dehydrated = inspection_pipeline._dehydrate_chunks(chunks)
            ranked = inspection_index.lexical_search(path, "InspectRepo", dehydrated)
        self.assertEqual(ranked[0]["chunk_id"], "a")

    def test_explicitly_named_paths_reuses_exact_query_cache(self):
        chunks = [
            {
                "chunk_id": "0",
                "path": "large.py",
                "content_hash": "sha256:0",
                "content": "def f0(): pass",
                "token_estimate": 5,
                "line_start": 1,
                "line_end": 1,
                "language": "python",
                "symbol": "f0",
            }
        ]
        inspection_pipeline._NAMED_PATH_CACHE.clear()
        cache_token = ("lexical.sqlite3", 1, 1)
        first = inspection_pipeline.explicitly_named_paths("inspect large.py", chunks, cache_token=cache_token)
        with mock.patch.object(
            inspection_pipeline,
            "query_features",
            side_effect=AssertionError("query_features should not be called"),
        ):
            second = inspection_pipeline.explicitly_named_paths("inspect large.py", chunks, cache_token=cache_token)
        self.assertEqual(first, second)

    def test_explicitly_named_paths_uses_provided_path_catalog(self):
        chunks = [
            {
                "chunk_id": "0",
                "path": "large.py",
                "content_hash": "sha256:0",
                "content": "def f0(): pass",
                "token_estimate": 5,
                "line_start": 1,
                "line_end": 1,
                "language": "python",
                "symbol": "f0",
            }
        ]
        catalog = {
            "unique_paths": ("large.py",),
            "path_lower_by_path": {"large.py": "large.py"},
            "path_basename_lower_by_path": {"large.py": "large.py"},
        }
        result = inspection_pipeline.explicitly_named_paths(
            "inspect large.py",
            [],
            path_catalog=catalog,
        )
        self.assertEqual(result, {"large.py"})

    def test_rrf_and_two_chunk_file_diversity(self):
        chunks = []
        ranked = []
        for index in range(6):
            path = "large.py" if index < 5 else "other.py"
            chunks.append(
                {
                    "chunk_id": str(index),
                    "path": path,
                    "content_hash": f"sha256:{index}",
                    "content": f"def f{index}(): pass",
                    "token_estimate": 5,
                    "line_start": index + 1,
                    "line_end": index + 1,
                    "language": "python",
                    "symbol": f"f{index}",
                }
            )
            ranked.append({"chunk_id": str(index), "rank": index + 1, "rrf_score": 1.0 / (61 + index)})
        chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        diverse, _ = inspection_pipeline.select_diverse_chunks(
            ranked,
            "find functions",
            1_000,
            chunk_by_id=chunk_by_id,
        )
        named, _ = inspection_pipeline.select_diverse_chunks(
            ranked,
            "inspect large.py",
            1_000,
            chunk_by_id=chunk_by_id,
        )
        self.assertEqual(sum(chunk_by_id[item["chunk_id"]]["path"] == "large.py" for item in diverse), 2)
        self.assertEqual(sum(chunk_by_id[item["chunk_id"]]["path"] == "large.py" for item in named), 5)

    def test_select_diverse_chunks_uses_provided_empty_named_paths_without_recomputing(self):
        chunks = [
            {
                "chunk_id": "0",
                "path": "large.py",
                "content_hash": "sha256:0",
                "content": "def f0(): pass",
                "token_estimate": 5,
                "line_start": 1,
                "line_end": 1,
                "language": "python",
                "symbol": "f0",
            }
        ]
        ranked = [{"chunk_id": "0", "rank": 1, "rrf_score": 1.0}]
        chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        with mock.patch.object(
            inspection_pipeline,
            "explicitly_named_paths",
            side_effect=AssertionError("explicitly_named_paths should not be called"),
        ):
            selected, used = inspection_pipeline.select_diverse_chunks(
                ranked,
                "inspect unrelated.py",
                100,
                chunk_by_id=chunk_by_id,
                named_paths=set(),
            )
        self.assertEqual([item["chunk_id"] for item in selected], ["0"])
        self.assertEqual(used, 5)
        self.assertTrue(all(set(item.keys()) <= {"chunk_id", "rank"} for item in selected))


class PipelineTests(RepoFixture):
    def test_runtime_reports_node_local_cache_selection(self):
        scratch = self.root / "scratch"
        scratch.mkdir()
        with mock.patch.dict(
            os.environ,
            {
                "TMPDIR": str(scratch),
                "BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(self.root / "shared-cache"),
            },
            clear=False,
        ):
            payload = inspection_pipeline.run_inspection(
                self.discovered,
                "Trace the retry_job service call chain",
                mode="evidence",
                execution_plan={
                    "job_id": "job_runtime_diag",
                    "repo_inspection_cache_path": str(self.root / ".broker" / "shared-working-cache"),
                    "repo_inspection_use_node_local_cache": True,
                    "runtime_backend": "local",
                },
                services=[],
                client_factory=SemanticDiagnosticsFactory(),
                output_dir=self.root / "out",
            )["payload"]

        self.assertEqual(payload["runtime"]["local_cache_origin"], "node_local_tmpdir")
        self.assertTrue(payload["runtime"]["node_local_cache_selected"])
        self.assertTrue(
            str(payload["runtime"]["local_cache_path"]).startswith(str((scratch / "local-ai-broker" / "inspect-repo").resolve()))
        )
        self.assertEqual(
            payload["runtime"]["shared_cache_path"],
            str((self.root / "shared-cache").resolve()),
        )

    def test_retrieval_reports_chunk_and_lexical_restore_timings(self):
        payload = self.run_pipeline(mode="evidence", services=[], factory=SemanticDiagnosticsFactory())

        self.assertIn("chunk_manifest_restore_ms", payload["retrieval"])
        self.assertIn("chunk_shared_manifest_load_ms", payload["retrieval"])
        self.assertIn("chunk_snapshot_local_load_ms", payload["retrieval"])
        self.assertIn("chunk_snapshot_shared_load_ms", payload["retrieval"])
        self.assertIn("chunk_snapshot_restore_source", payload["retrieval"])
        self.assertIn("lexical_index_working_manifest_load_ms", payload["retrieval"])
        self.assertIn("lexical_index_working_check_ms", payload["retrieval"])
        self.assertIn("lexical_index_shared_restore_ms", payload["retrieval"])
        self.assertIn("lexical_index_sqlite_update_ms", payload["retrieval"])
        self.assertIn("lexical_index_sqlite_rebuild_ms", payload["retrieval"])
        self.assertIn("chunk_build_substage_timings_ms", payload["retrieval"])

    def test_dehydrate_and_rehydrate_chunk_contents_from_lexical_index(self):
        chunks = [
            {
                "chunk_id": "a",
                "path": "service.py",
                "repository_path": "service.py",
                "source_namespace": "",
                "language": "python",
                "symbol": "retry_job",
                "line_start": 1,
                "line_end": 2,
                "content": "def retry_job(job_id):\n    return submit_job(job_id)\n",
                "content_hash": "sha256:a",
                "token_estimate": 10,
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path, _, _ = inspection_index.ensure_lexical_index(chunks, temp_dir, "sha256:test")
            dehydrated = inspection_pipeline._dehydrate_chunks(chunks)
            self.assertNotIn("content", dehydrated[0])
            chunk_by_id = {chunk["chunk_id"]: dict(chunk) for chunk in dehydrated}
            inspection_pipeline._hydrate_chunk_contents(index_path, chunk_by_id, ["a"])

        self.assertEqual(
            chunk_by_id["a"]["content"],
            "def retry_job(job_id):\n    return submit_job(job_id)\n",
        )

    def test_semantic_sync_manifest_skip_rewrite_when_unchanged(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            documents = {
                "a": '{"chunk_id":"a","content_hash":"sha256:a","line_end":1,"line_start":1,"path":"main.py"}'
            }
            self.assertTrue(
                inspection_pipeline._write_semantic_sync_manifest(
                    cache_dir,
                    documents,
                    "sha256:fingerprint",
                    "sha256:scope",
                )
            )
            previous = inspection_pipeline._load_semantic_sync_manifest(cache_dir, "sha256:scope")
            with mock.patch.object(
                inspection_pipeline.os,
                "replace",
                side_effect=AssertionError("semantic manifest should not be rewritten"),
            ):
                rewritten = inspection_pipeline._write_semantic_sync_manifest(
                    cache_dir,
                    documents,
                    "sha256:fingerprint",
                    "sha256:scope",
                    previous=previous,
                )
        self.assertFalse(rewritten)

    def test_semantic_sync_manifest_restores_shared_manifest_without_local_write_through(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            shared_cache_dir = temp_root / "shared-cache"
            first_cache_dir = temp_root / "cache-one"
            second_cache_dir = temp_root / "cache-two"
            documents = {
                "a": '{"chunk_id":"a","content_hash":"sha256:a","line_end":1,"line_start":1,"path":"main.py"}'
            }

            with mock.patch.dict(
                os.environ,
                {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)},
                clear=False,
            ):
                self.assertTrue(
                    inspection_pipeline._write_semantic_sync_manifest(
                        first_cache_dir,
                        documents,
                        "sha256:fingerprint",
                        "sha256:scope",
                    )
                )
                self.assertFalse(inspection_pipeline._semantic_sync_manifest_path(first_cache_dir, "sha256:scope").exists())
                self.assertFalse(inspection_pipeline._semantic_sync_manifest_path(second_cache_dir, "sha256:scope").exists())
                payload = inspection_pipeline._load_semantic_sync_manifest(second_cache_dir, "sha256:scope")

        self.assertEqual(
            payload,
            {"fingerprint": "sha256:fingerprint", "documents": dict(documents)},
        )

    def test_semantic_sync_manifest_scopes_shared_manifests_by_build_and_model_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            shared_cache_dir = temp_root / "shared-cache"
            cache_dir = temp_root / "cache"
            documents_a = {"a": '{"chunk_id":"a","content_hash":"sha256:a","path":"repo_a.py"}'}
            documents_b = {"b": '{"chunk_id":"b","content_hash":"sha256:b","path":"repo_b.py"}'}

            with mock.patch.dict(
                os.environ,
                {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)},
                clear=False,
            ):
                self.assertTrue(
                    inspection_pipeline._write_semantic_sync_manifest(
                        cache_dir,
                        documents_a,
                        "sha256:fingerprint-a",
                        "sha256:scope-a",
                    )
                )
                self.assertTrue(
                    inspection_pipeline._write_semantic_sync_manifest(
                        cache_dir,
                        documents_b,
                        "sha256:fingerprint-b",
                        "sha256:scope-b",
                    )
                )
                payload_a = inspection_pipeline._load_semantic_sync_manifest(cache_dir, "sha256:scope-a")
                payload_b = inspection_pipeline._load_semantic_sync_manifest(cache_dir, "sha256:scope-b")

        self.assertEqual(payload_a, {"fingerprint": "sha256:fingerprint-a", "documents": dict(documents_a)})
        self.assertEqual(payload_b, {"fingerprint": "sha256:fingerprint-b", "documents": dict(documents_b)})

    def test_semantic_document_signatures_use_attached_chunk_metadata_without_iterating_chunks(self):
        class SemanticOnlyChunks(inspection_index.ChunkList):
            def __iter__(self):
                raise AssertionError("attached semantic signatures should avoid chunk iteration")

        chunks = SemanticOnlyChunks(
            [
                {
                    "chunk_id": "a",
                    "path": "main.py",
                    "line_start": 1,
                    "line_end": 1,
                    "content_hash": "sha256:a",
                }
            ]
        )
        chunks._semantic_document_signatures = {
            "a": '{"chunk_id":"a","content_hash":"sha256:a","line_end":1,"line_start":1,"path":"main.py"}'
        }

        signatures = inspection_pipeline._semantic_document_signatures(chunks)

        self.assertEqual(
            signatures,
            {"a": '{"chunk_id":"a","content_hash":"sha256:a","line_end":1,"line_start":1,"path":"main.py"}'},
        )

    def test_semantic_chunk_ids_use_attached_metadata_without_iterating_chunks(self):
        class SemanticOnlyChunks(inspection_index.ChunkList):
            def __iter__(self):
                raise AssertionError("attached semantic ids should avoid chunk iteration")

        chunks = SemanticOnlyChunks([{"chunk_id": "a"}, {"chunk_id": "b"}])
        chunks._chunk_ids = ("a", "b")

        self.assertEqual(inspection_pipeline._semantic_chunk_ids(chunks), ("a", "b"))

    def test_write_file_chunk_cache_uses_precomputed_metadata_and_signatures(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            chunks = [
                {
                    "chunk_id": "chunk_001",
                    "path": "main.py",
                    "repository_path": "main.py",
                    "source_namespace": "repo",
                    "language": "python",
                    "symbol": "main",
                    "line_start": 1,
                    "line_end": 1,
                    "content": "print('hi')",
                    "content_hash": "sha256:content",
                    "chunk_hash": "sha256:content",
                    "token_estimate": 3,
                }
            ]
            precomputed_lexical_record = {
                "file_key": "repo:main.py",
                "path": "main.py",
                "source_namespace": "repo",
                "repository_path": "main.py",
                "chunk_ids": ["chunk_001"],
                "content_hashes": ["sha256:content"],
                "line_ranges": [[1, 1]],
                "symbols": ["main"],
                "language": "python",
            }
            precomputed_index_signature = "sha256:index"
            precomputed_semantic_signatures = {"chunk_001": "sha256:semantic"}

            with mock.patch.object(
                inspection_index,
                "_manifest_records_for_file_chunks",
                side_effect=AssertionError("should use precomputed manifest metadata"),
            ), mock.patch.object(
                inspection_index,
                "semantic_chunk_signature",
                side_effect=AssertionError("should use precomputed semantic signatures"),
            ):
                inspection_index._write_file_chunk_cache(
                    cache_dir,
                    "cache-key",
                    chunks,
                    publish_shared=False,
                    lexical_record=precomputed_lexical_record,
                    index_signature=precomputed_index_signature,
                    semantic_document_signatures=precomputed_semantic_signatures,
                )

            payload = inspection_index._load_file_chunk_cache_payload(
                inspection_index._file_chunk_cache_path(cache_dir, "cache-key")
            )

        self.assertEqual(payload["lexical_record"], precomputed_lexical_record)
        self.assertEqual(payload["index_signature"], precomputed_index_signature)
        self.assertEqual(payload["semantic_document_signatures"], precomputed_semantic_signatures)

    def test_cpu_only_auto_is_evidence_only_and_never_synthesizes(self):
        payload = self.run_pipeline()
        self.assertNotIn("answer", payload)
        self.assertEqual(payload["findings"], [])
        self.assertEqual(
            payload["quality"],
            {
                "result": "evidence_only",
                "retrieval": "lexical_degraded",
                "reranking": "unavailable",
                "synthesis": "not_requested",
                "answer_ready": False,
            },
        )
        self.assertTrue(payload["evidence"][0]["source_refs"])

    def test_empty_repository_skips_gpu_without_retiring_a_service(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            factory = FakeClientFactory()
            payload = inspection_pipeline.run_inspection(
                [{"id": "empty", "type": "repo", "classification": "internal", "path": root}],
                "Where is the implementation?",
                mode="answer",
                services=all_services(),
                client_factory=factory,
                output_dir=root / "out",
            )["payload"]

        self.assertEqual(payload["quality"]["result"], "failed")
        self.assertEqual(payload["quality"]["retrieval"], "failed")
        self.assertEqual(payload["evidence"], [])
        self.assertEqual(payload["runtime"]["attempts"], [])
        self.assertIn("NO_SUPPORTED_REPOSITORY_SOURCES", payload["warnings"])
        self.assertEqual(factory.embed_calls, [])
        self.assertEqual(factory.rerank_calls, [])
        self.assertEqual(factory.chat_calls, [])

    def test_all_gpu_stages_are_required_for_answer_ready(self):
        factory = FakeClientFactory()
        payload = self.run_pipeline(services=all_services(), factory=factory)
        self.assertEqual(payload["quality"]["result"], "answer_ready")
        self.assertEqual(payload["quality"]["retrieval"], "gpu")
        self.assertEqual(payload["quality"]["reranking"], "gpu")
        self.assertEqual(payload["quality"]["synthesis"], "gpu")
        valid_ids = {item["id"] for item in payload["evidence"]}
        self.assertTrue(payload["answer"])
        self.assertTrue(all(set(item["evidence_refs"]) <= valid_ids for item in payload["findings"]))
        self.assertEqual(factory.chat_calls, ["p40-synthesis"])
        self.assertTrue(factory.rerank_calls)
        self.assertEqual(factory.rerank_calls[0][0], "p40-retrieval")

    def test_embed_only_service_falls_back_to_pure_lexical_evidence(self):
        services = all_services()
        services[0]["capabilities"] = ["embed", "rerank"]
        factory = FakeClientFactory()
        payload = self.run_pipeline(services=services, factory=factory)
        self.assertEqual(payload["quality"]["result"], "evidence_only")
        self.assertEqual(payload["quality"]["retrieval"], "lexical_degraded")
        self.assertEqual(factory.chat_calls, [])
        self.assertEqual(factory.embed_calls, [])
        semantic_attempt = next(
            item for item in payload["runtime"]["attempts"] if item["operation"] == "semantic_retrieval"
        )
        self.assertEqual(semantic_attempt["failure_category"], "availability")

        answer_payload = self.run_pipeline(mode="answer", services=services, factory=FakeClientFactory())
        self.assertEqual(answer_payload["quality"]["result"], "evidence_only")
        self.assertNotIn("answer", answer_payload)
        self.assertEqual(answer_payload["findings"], [])

    def test_unknown_citation_retries_same_tier_before_escalating(self):
        factory = FakeClientFactory(
            {
                "p40-synthesis": [
                    {"answer": "unsupported", "findings": [{"summary": "bad", "evidence_refs": ["ev_999"]}]},
                    {"answer": "supported", "findings": [{"summary": "good", "evidence_refs": ["ev_001"]}]},
                ]
            }
        )
        payload = self.run_pipeline(services=all_services(), factory=factory)
        synthesis_attempts = [item for item in payload["runtime"]["attempts"] if item["operation"] == "synthesis"]
        self.assertEqual([item["tier"] for item in synthesis_attempts], ["p40-synthesis", "p40-synthesis"])
        self.assertEqual(synthesis_attempts[0]["failure_category"], "unsupported_claim")
        self.assertEqual(synthesis_attempts[1]["status"], "succeeded")

    def test_v100_availability_failure_selects_single_a100(self):
        services = all_services()
        for record in services:
            if record["tier"] == "v100-reasoning":
                record["state"] = "starting"
                record["failure_category"] = "queue_delay"
        factory = FakeClientFactory(
            {"p40-synthesis": [gpu_client.GPUServiceError("service_failure", "down")]}
        )
        payload = self.run_pipeline(services=services, factory=factory)
        tiers = [item["tier"] for item in payload["runtime"]["attempts"] if item["operation"] == "synthesis"]
        self.assertEqual(tiers, ["p40-synthesis", "v100-reasoning", "a100-single"])
        self.assertEqual(payload["provenance"]["synthesis_tier"], "a100-single")

    def test_v100_oom_selects_four_gpu_a100(self):
        factory = FakeClientFactory(
            {
                "p40-synthesis": [gpu_client.GPUServiceError("service_failure", "down")],
                "v100-reasoning": [gpu_client.GPUServiceError("oom", "CUDA OOM")],
            }
        )
        payload = self.run_pipeline(services=all_services(), factory=factory)
        tiers = [item["tier"] for item in payload["runtime"]["attempts"] if item["operation"] == "synthesis"]
        self.assertEqual(tiers, ["p40-synthesis", "v100-reasoning", "a100-multigpu"])
        self.assertEqual(payload["provenance"]["synthesis_tier"], "a100-multigpu")

    def test_failed_demand_diagnostics_are_retained_in_attempt_history(self):
        error = gpu_client.GPUServiceError(
            "oom",
            "Slurm OUT_OF_MEMORY",
            service_diagnostics={
                "tier": "v100-reasoning",
                "slurm_job_id": "v100-job-42",
                "gpu": {"type": "v100", "count": 4},
                "model_profile": "v100-reasoning-profile",
                "endpoint": "http://must-not-be-retained.invalid",
                "model": "/models/must-not-be-retained",
            },
        )

        def requester(_tier, _failure_category, _reason):
            raise error

        attempts = []
        result, category, endpoint = inspection_pipeline._try_synthesis_tier(
            [],
            "v100-reasoning",
            4,
            "question",
            [],
            {},
            attempts,
            FakeClientFactory(),
            "p40_service_failure",
            service_requester=requester,
        )

        self.assertIsNone(result)
        self.assertEqual(category, "oom")
        self.assertIsNone(endpoint)
        self.assertEqual(
            attempts,
            [
                {
                    "operation": "synthesis",
                    "tier": "v100-reasoning",
                    "slurm_job_id": "v100-job-42",
                    "gpu_count": 4,
                    "gpu_type": "v100",
                    "model_profile": "v100-reasoning-profile",
                    "attempt": 1,
                    "status": "failed",
                    "failure_category": "oom",
                    "escalation_reason": "p40_service_failure",
                }
            ],
        )
        self.assertNotIn("endpoint", error.service_diagnostics)
        self.assertNotIn("model", error.service_diagnostics)

    def test_synthesis_exhaustion_is_evidence_only_for_auto_and_failed_for_answer(self):
        failures = {
            "p40-synthesis": [gpu_client.GPUServiceError("service_failure", "down")],
            "v100-reasoning": [gpu_client.GPUServiceError("service_failure", "down")],
            "a100-single": [gpu_client.GPUServiceError("service_failure", "down")],
        }
        auto = self.run_pipeline(services=all_services(), factory=FakeClientFactory(failures))
        answer = self.run_pipeline(mode="answer", services=all_services(), factory=FakeClientFactory(failures))
        self.assertEqual(auto["quality"]["result"], "evidence_only")
        self.assertEqual(answer["quality"]["result"], "failed")
        self.assertNotIn("answer", auto)
        self.assertNotIn("answer", answer)
        self.assertEqual(answer["findings"], [])
        self.assertEqual(
            [item["tier"] for item in answer["runtime"]["attempts"] if item["operation"] == "synthesis"],
            ["p40-synthesis", "v100-reasoning", "a100-single"],
        )

    def test_final_pack_budget_caps_evidence_and_answer_content(self):
        payload = self.run_pipeline(
            services=all_services(),
            factory=FakeClientFactory(),
            constraints={
                "retrieval_token_budget": 2_000,
                "evidence_token_budget": 2_000,
                "final_pack_token_budget": 700,
                "synthesis_context_token_budget": 4_000,
            },
        )
        self.assertLessEqual(inspection_pipeline.released_pack_tokens(payload), 700)
        self.assertEqual(payload["retrieval"]["final_pack_tokens"], inspection_pipeline.released_pack_tokens(payload))


class GPUClientTests(unittest.TestCase):
    def test_remote_semantic_index_uploads_only_on_cache_miss(self):
        chunks = [{
            "chunk_id": "known",
            "path": "main.py",
            "language": "python",
            "symbol": "main",
            "line_start": 1,
            "line_end": 1,
            "content_hash": "sha256:known",
            "content": "def main(): pass",
        }]
        client = gpu_client.GPUServiceClient(service("p40-retrieval", ["faiss_search"], 1))
        with mock.patch.object(
            client,
            "_post",
            side_effect=[
                {"ready": False, "document_count": 0},
                {"accepted": True, "embedded_documents": 1, "reused_documents": 0, "total_documents": 1},
                {"ready": True, "document_count": 1},
                {"results": [{"id": "known", "score": 1.0}]},
            ],
        ) as post:
            self.assertEqual(
                client.ensure_semantic_index(chunks, "sha256:index"),
                {
                    "cache_hit": False,
                    "document_count": 1,
                    "embedded_documents": 1,
                    "reused_documents": 0,
                },
            )
            ranked = client.semantic_search("main", chunks, "sha256:index", 10)

        self.assertEqual(ranked[0]["chunk_id"], "known")
        self.assertEqual([call.args[0] for call in post.call_args_list], [
            "index_status", "index_upsert", "index_status", "search",
        ])
        self.assertIn("documents", post.call_args_list[1].args[2])
        self.assertNotIn("documents", post.call_args_list[3].args[2])

        with mock.patch.object(client, "_post", return_value={"ready": True, "document_count": 1}) as post:
            self.assertEqual(
                client.ensure_semantic_index(chunks, "sha256:index"),
                {
                    "cache_hit": True,
                    "document_count": 1,
                    "embedded_documents": 0,
                    "reused_documents": 0,
                },
            )
        self.assertEqual(post.call_count, 1)

    def test_remote_semantic_index_delta_upload_uses_base_fingerprint_and_removed_ids(self):
        chunks = [
            {
                "chunk_id": "known",
                "path": "main.py",
                "language": "python",
                "symbol": "main",
                "line_start": 1,
                "line_end": 1,
                "content_hash": "sha256:known",
                "content": "def main(): pass",
            },
            {
                "chunk_id": "new",
                "path": "extra.py",
                "language": "python",
                "symbol": "extra",
                "line_start": 1,
                "line_end": 1,
                "content_hash": "sha256:new",
                "content": "def extra(): pass",
            },
        ]
        client = gpu_client.GPUServiceClient(service("p40-retrieval", ["faiss_search"], 1))
        with mock.patch.object(
            client,
            "_post",
            side_effect=[
                {"ready": False, "document_count": 0},
                {"accepted": True, "base_reused": True, "embedded_documents": 1, "reused_documents": 0, "total_documents": 2},
                {"ready": True, "document_count": 2},
            ],
        ) as post:
            result = client.ensure_semantic_index(
                chunks,
                "sha256:index-next",
                sync_plan={
                    "base_fingerprint": "sha256:index-prev",
                    "changed_ids": ["new"],
                    "removed_ids": ["gone"],
                },
            )

        self.assertEqual(result["embedded_documents"], 1)
        upsert_body = post.call_args_list[1].args[2]
        self.assertEqual(upsert_body["base_index_fingerprint"], "sha256:index-prev")
        self.assertEqual(upsert_body["removed_document_ids"], ["gone"])

    def test_remote_semantic_index_delta_uses_attached_file_mapping_without_iterating_all_chunks(self):
        class GuardedChunks(inspection_index.ChunkList):
            def __iter__(self):
                raise AssertionError("delta upload should avoid iterating whole chunk list when attached mappings exist")

        chunks = GuardedChunks(
            [
                {
                    "chunk_id": "keep",
                    "path": "main.py",
                    "language": "python",
                    "symbol": "keep",
                    "line_start": 1,
                    "line_end": 1,
                    "content_hash": "sha256:keep",
                    "content": "def keep(): pass",
                },
                {
                    "chunk_id": "new",
                    "path": "main.py",
                    "language": "python",
                    "symbol": "new",
                    "line_start": 2,
                    "line_end": 2,
                    "content_hash": "sha256:new",
                    "content": "def new(): pass",
                },
            ]
        )
        chunks._chunks_by_file = {
            "repo\0main.py": [dict(chunks[0]), dict(chunks[1])],
        }
        chunks._file_key_by_chunk_id = {
            "keep": "repo\0main.py",
            "new": "repo\0main.py",
        }
        client = gpu_client.GPUServiceClient(service("p40-retrieval", ["faiss_search"], 1))
        with mock.patch.object(
            client,
            "_post",
            side_effect=[
                {"ready": False, "document_count": 0},
                {"accepted": True, "base_reused": True, "embedded_documents": 1, "reused_documents": 0, "total_documents": 2},
                {"ready": True, "document_count": 2},
            ],
        ) as post:
            result = client.ensure_semantic_index(
                chunks,
                "sha256:index",
                sync_plan={
                    "base_fingerprint": "sha256:index-prev",
                    "changed_ids": ["new"],
                    "removed_ids": [],
                },
            )
        self.assertEqual(result["embedded_documents"], 1)
        upsert_body = post.call_args_list[1].args[2]
        self.assertEqual([doc["id"] for doc in upsert_body["documents"]], ["new"])
        self.assertEqual([document["id"] for document in upsert_body["documents"]], ["new"])

    def test_remote_semantic_index_delta_reuses_base_without_full_upload_when_documents_unchanged(self):
        chunks = [
            {
                "chunk_id": "known",
                "path": "main.py",
                "language": "python",
                "symbol": "main",
                "line_start": 1,
                "line_end": 1,
                "content_hash": "sha256:known",
                "content": "def main(): pass",
            }
        ]
        client = gpu_client.GPUServiceClient(service("p40-retrieval", ["faiss_search"], 1))
        with mock.patch.object(
            client,
            "_post",
            side_effect=[
                {"ready": False, "document_count": 0},
                {"accepted": True, "base_reused": True, "embedded_documents": 0, "reused_documents": 0, "total_documents": 1},
                {"ready": True, "document_count": 1},
            ],
        ) as post:
            result = client.ensure_semantic_index(
                chunks,
                "sha256:index-next",
                sync_plan={
                    "base_fingerprint": "sha256:index-prev",
                    "changed_ids": [],
                    "removed_ids": [],
                },
            )

        self.assertEqual(
            result,
            {
                "cache_hit": False,
                "document_count": 1,
                "embedded_documents": 0,
                "reused_documents": 0,
            },
        )
        self.assertEqual([call.args[0] for call in post.call_args_list], ["index_status", "index_upsert", "index_status"])
        upsert_body = post.call_args_list[1].args[2]
        self.assertEqual(upsert_body["base_index_fingerprint"], "sha256:index-prev")
        self.assertEqual(upsert_body["documents"], [])
        self.assertTrue(upsert_body["replace"])
        self.assertTrue(upsert_body["finalize"])

    def test_services_are_loaded_only_from_broker_registry_records(self):
        injected = service("p40-retrieval", ["embed"], 1)
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = Path(temp_dir) / "registry.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "schema": "gpu_service_registry_v1",
                        "records": [service("p40-synthesis", ["chat"], 1)],
                    }
                ),
                encoding="utf-8",
            )
            records = gpu_client.services_from_execution_plan(
                {"gpu_service_registry_path": str(registry_path)},
                {
                    "gpu_services": [injected],
                    "gpu_service_registry_path": "/caller/controlled.json",
                },
            )

        self.assertEqual([record["tier"] for record in records], ["p40-synthesis"])

    def test_services_from_execution_plan_reuses_unchanged_registry_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = Path(temp_dir) / "registry.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "schema": "gpu_service_registry_v1",
                        "records": [service("p40-synthesis", ["chat"], 1)],
                    }
                ),
                encoding="utf-8",
            )
            gpu_client._REGISTRY_CACHE.clear()
            original_json_load = gpu_client.json.load
            load_calls = 0

            def counting_load(handle):
                nonlocal load_calls
                load_calls += 1
                return original_json_load(handle)

            with mock.patch.object(gpu_client.json, "load", side_effect=counting_load):
                first = gpu_client.services_from_execution_plan({"gpu_service_registry_path": str(registry_path)})
                second = gpu_client.services_from_execution_plan({"gpu_service_registry_path": str(registry_path)})

        self.assertEqual(load_calls, 1)
        self.assertEqual([record["tier"] for record in first], ["p40-synthesis"])
        self.assertEqual([record["tier"] for record in second], ["p40-synthesis"])

    def test_endpoint_without_fresh_lease_and_heartbeat_is_not_routable(self):
        record = service("p40-retrieval", ["embed"], 1)
        record.pop("heartbeat_at")
        record.pop("lease_expires_at")
        self.assertFalse(gpu_client.endpoint_is_healthy(record))

    def test_endpoint_health_honors_earlier_absolute_lease(self):
        now = datetime.now(timezone.utc)
        record = service("v100-reasoning", ["chat"], 4)
        record["heartbeat_at"] = now.isoformat().replace("+00:00", "Z")
        record["lease_expires_at"] = (now + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
        record["absolute_lease_expires_at"] = (now - timedelta(seconds=1)).isoformat().replace("+00:00", "Z")

        self.assertFalse(gpu_client.endpoint_is_healthy(record, now=now))

    def test_endpoint_health_ignores_go_zero_time_for_warm_p40(self):
        now = datetime.now(timezone.utc)
        record = service("p40-retrieval", ["embed"], 1)
        record["heartbeat_at"] = now.isoformat().replace("+00:00", "Z")
        record["lease_expires_at"] = (now + timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
        record["absolute_lease_expires_at"] = "0001-01-01T00:00:00Z"

        self.assertTrue(gpu_client.endpoint_is_healthy(record, now=now))

    def test_embedding_request_uses_bearer_auth_and_configured_model(self):
        captured = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"data":[{"index":0,"embedding":[1,2]}]}'

        def fake_urlopen(request, timeout):
            captured["authorization"] = request.get_header("Authorization")
            captured["body"] = json.loads(request.data)
            return Response()

        record = service("p40-retrieval", ["embed"], 1)
        with mock.patch.object(gpu_client.urllib.request, "urlopen", side_effect=fake_urlopen):
            vectors = gpu_client.GPUServiceClient(record).embed(["hello"])

        self.assertEqual(vectors, [[1.0, 2.0]])
        self.assertEqual(captured["authorization"], "Bearer test-token")
        self.assertEqual(captured["body"]["model"], "/models/p40-retrieval")

    def test_index_operations_use_longer_timeout_than_search(self):
        observed = []

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"ready":true,"results":[]}'

        def fake_urlopen(request, timeout):
            observed.append((request.full_url, timeout))
            return Response()

        record = service("p40-retrieval", ["faiss_search"], 1)
        record["timeout_seconds"] = 17
        with mock.patch.dict(os.environ, {"BROKER_GPU_SERVICE_INDEX_TIMEOUT_SECONDS": "123"}, clear=False):
            with mock.patch.object(gpu_client.urllib.request, "urlopen", side_effect=fake_urlopen):
                client = gpu_client.GPUServiceClient(record)
                client._post("index_status", "v1/indexes/status", {"model": "m"})
                client._post("search", "v1/search", {"model": "m"})

        self.assertEqual(observed, [
            ("http://p40-retrieval.invalid/v1/indexes/status", 123.0),
            ("http://p40-retrieval.invalid/v1/search", 17.0),
        ])

    def test_stale_heartbeat_is_not_routed(self):
        record = service("p40-retrieval", ["embed"], 1)
        record["heartbeat_at"] = "2000-01-01T00:00:00Z"
        selected = gpu_client.select_endpoint(
            [record], "p40-retrieval", "embed", expected_gpu_count=1, health_interval_seconds=10
        )
        self.assertIsNone(selected)

    def test_semantic_search_rejects_unknown_or_duplicate_chunk_ids(self):
        chunks = [{
            "chunk_id": "known",
            "path": "main.py",
            "language": "python",
            "symbol": "main",
            "line_start": 1,
            "line_end": 1,
            "content_hash": "sha256:known",
            "content": "def main(): pass",
        }]

        class Response:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

        record = service("p40-retrieval", ["faiss_search"], 1)
        for results in (
            [{"id": "unknown", "score": 1}],
            [{"id": "known", "score": 1}, {"id": "known", "score": 0.5}],
        ):
            with mock.patch.object(
                gpu_client.urllib.request,
                "urlopen",
                return_value=Response({"results": results}),
            ):
                with self.assertRaises(gpu_client.GPUServiceError):
                    gpu_client.GPUServiceClient(record).semantic_search("main", chunks, "sha256:index", 10)

    def test_semantic_search_accepts_compact_chunk_id_iterable(self):
        class Response:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

        record = service("p40-retrieval", ["faiss_search"], 1)
        with mock.patch.object(
            gpu_client.urllib.request,
            "urlopen",
            return_value=Response({"results": [{"id": "known", "score": 1.0}]}),
        ):
            ranked = gpu_client.GPUServiceClient(record).semantic_search(
                "main",
                ("known",),
                "sha256:index",
                10,
            )
        self.assertEqual(ranked[0]["chunk_id"], "known")

    def test_lexical_cache_is_private_under_permissive_umask(self):
        chunk = {
            "chunk_id": "secret",
            "path": "secret.py",
            "language": "python",
            "symbol": "secret",
            "line_start": 1,
            "line_end": 1,
            "content_hash": "sha256:secret",
            "token_estimate": 2,
            "content": "secret source",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            old_umask = os.umask(0o022)
            try:
                index_path, _, _ = inspection_index.ensure_lexical_index(
                    [chunk], Path(temp_dir) / "cache", "sha256:index"
                )
            finally:
                os.umask(old_umask)

            self.assertEqual(index_path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(index_path.parent.stat().st_mode & 0o777, 0o700)

    def test_cache_directory_rejects_symlink_components(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            outside = base / "outside"
            outside.mkdir()
            repo = base / "repo"
            repo.mkdir()
            (repo / ".broker").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(OSError, "symlink"):
                inspection_index.ensure_lexical_index([], repo / ".broker" / "cache", "sha256:index")
            self.assertEqual(outside.stat().st_mode & 0o777, 0o755)

    def test_cache_dir_canonicalizes_broker_owned_symlink_alias(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            real_root = base / "real-root"
            real_root.mkdir()
            alias_parent = base / "alias-parent"
            alias_parent.mkdir()
            alias = alias_parent / "broker-root"
            alias.symlink_to(real_root, target_is_directory=True)

            cache_dir = inspection_pipeline._cache_dir(
                {"repo_inspection_cache_path": str(alias / "cache")},
                None,
            )

            self.assertEqual(cache_dir, (real_root / "cache").resolve())
            self.assertFalse(any(component.is_symlink() for component in (cache_dir, *cache_dir.parents[:2])))

    def test_cache_dir_prefers_tmpdir_when_execution_plan_requests_node_local_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            scratch = base / "node-local"
            scratch.mkdir()
            output_dir = base / "out"
            cache_dir = inspection_pipeline._cache_dir(
                {"repo_inspection_cache_path": str(base / "shared-cache"), "repo_inspection_use_node_local_cache": True, "job_id": "job_123"},
                output_dir,
            )

            with mock.patch.dict(os.environ, {"TMPDIR": str(scratch)}, clear=False):
                cache_dir = inspection_pipeline._cache_dir(
                    {
                        "repo_inspection_cache_path": str(base / "shared-cache"),
                        "repo_inspection_use_node_local_cache": True,
                        "job_id": "job_123",
                    },
                    output_dir,
                )

        self.assertEqual(cache_dir, (scratch / "local-ai-broker" / "inspect-repo" / "job_123").resolve())

    def test_hotpath_cache_dir_prefers_tmpdir_when_execution_plan_requests_node_local_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            scratch = base / "node-local"
            scratch.mkdir()
            output_dir = base / "out"

            with mock.patch.dict(os.environ, {"TMPDIR": str(scratch)}, clear=False):
                cache_dir = inspection_hotpath.cache_dir_for_execution(
                    {
                        "repo_inspection_cache_path": str(base / "shared-cache"),
                        "repo_inspection_use_node_local_cache": True,
                        "job_id": "job_456",
                    },
                    output_dir,
                )

        self.assertEqual(cache_dir, (scratch / "local-ai-broker" / "inspect-repo" / "job_456").resolve())

    def test_hotpath_cache_dir_prefers_node_local_cache_for_local_backend_when_requested(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            scratch = base / "node-local"
            scratch.mkdir()
            output_dir = base / "out"

            with mock.patch.dict(os.environ, {"TMPDIR": str(scratch)}, clear=False):
                cache_dir = inspection_hotpath.cache_dir_for_execution(
                    {
                        "runtime_backend": "local",
                        "repo_inspection_cache_path": str(base / "shared-cache"),
                        "repo_inspection_use_node_local_cache": True,
                        "job_id": "job_local",
                    },
                    output_dir,
                )

        self.assertEqual(cache_dir, (scratch / "local-ai-broker" / "inspect-repo" / "job_local").resolve())

    def test_hotpath_cache_dir_prefers_node_local_cache_for_local_execution_backend_when_requested(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            scratch = base / "node-local"
            scratch.mkdir()
            output_dir = base / "out"

            with mock.patch.dict(os.environ, {"TMPDIR": str(scratch)}, clear=False):
                cache_dir = inspection_hotpath.cache_dir_for_execution(
                    {
                        "runtime_backend": "deterministic",
                        "execution_profile": {"backend": "local"},
                        "repo_inspection_cache_path": str(base / "shared-cache"),
                        "repo_inspection_use_node_local_cache": True,
                        "job_id": "job_local_profile",
                    },
                    output_dir,
                )

        self.assertEqual(cache_dir, (scratch / "local-ai-broker" / "inspect-repo" / "job_local_profile").resolve())

    def test_cache_dir_prefers_node_local_cache_for_local_execution_backend_when_requested(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            scratch = base / "node-local"
            scratch.mkdir()
            output_dir = base / "out"

            with mock.patch.dict(os.environ, {"TMPDIR": str(scratch)}, clear=False):
                cache_dir = inspection_pipeline._cache_dir(
                    {
                        "runtime_backend": "deterministic",
                        "execution_profile": {"backend": "local"},
                        "repo_inspection_cache_path": str(base / "shared-cache"),
                        "repo_inspection_use_node_local_cache": True,
                        "job_id": "job_local_profile",
                    },
                    output_dir,
                )

        self.assertEqual(cache_dir, (scratch / "local-ai-broker" / "inspect-repo" / "job_local_profile").resolve())

    def test_cache_dir_falls_back_to_python_tempdir_when_node_local_requested_without_tmp_env(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            output_dir = base / "out"
            with mock.patch.dict(os.environ, {"TMPDIR": "", "TMP": "", "TEMP": ""}, clear=False):
                with mock.patch("tempfile.gettempdir", return_value=str(base / "python-tmp")):
                    cache_dir = inspection_pipeline._cache_dir(
                        {
                            "repo_inspection_cache_path": str(base / "shared-cache"),
                            "repo_inspection_use_node_local_cache": True,
                            "job_id": "job_789",
                        },
                        output_dir,
                    )

        self.assertEqual(cache_dir, (base / "python-tmp" / "local-ai-broker" / "inspect-repo" / "job_789").resolve())

    def test_hotpath_runtime_reports_python_tempdir_fallback_when_node_local_requested_without_tmp_env(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            cache_dir = base / "python-tmp" / "local-ai-broker" / "inspect-repo" / "job_999"
            with mock.patch.dict(os.environ, {"TMPDIR": "", "TMP": "", "TEMP": ""}, clear=False):
                with mock.patch("tempfile.gettempdir", return_value=str(base / "python-tmp")):
                    diagnostics = inspection_hotpath.cache_runtime_diagnostics(
                        {
                            "repo_inspection_cache_path": str(base / "shared-cache"),
                            "repo_inspection_use_node_local_cache": True,
                            "job_id": "job_999",
                        },
                        base / "out",
                        cache_dir,
                    )

        self.assertEqual(diagnostics["local_cache_origin"], "node_local_tmpdir")
        self.assertTrue(diagnostics["node_local_cache_selected"])
        self.assertEqual(diagnostics["tmp_env"], "python_tempdir")
        self.assertEqual(diagnostics["tmp_root"], str((base / "python-tmp").resolve()))

    def test_index_fingerprint_tracks_chunk_manifest_and_schema_version(self):
        chunk = {
            "chunk_id": "chunk-a", "path": "main.py", "language": "python", "symbol": "main",
            "line_start": 1, "line_end": 2, "content_hash": "sha256:a", "classification": "internal",
        }
        original = inspection_index.inspection_index_fingerprint("sha256:repo", [chunk])
        changed = inspection_index.inspection_index_fingerprint("sha256:repo", [dict(chunk, symbol="other")])
        upgraded = inspection_index.inspection_index_fingerprint("sha256:repo", [chunk], "next-schema")

        self.assertEqual(len({original, changed, upgraded}), 3)

    def test_index_fingerprint_changes_when_repository_state_changes_even_if_chunk_manifest_is_unchanged(self):
        chunk = {
            "chunk_id": "chunk-a",
            "path": "main.py",
            "language": "python",
            "symbol": "main",
            "line_start": 1,
            "line_end": 2,
            "content_hash": "sha256:a",
            "classification": "internal",
        }

        first = inspection_index.inspection_index_fingerprint("sha256:repo-a", [chunk])
        second = inspection_index.inspection_index_fingerprint("sha256:repo-b", [chunk])

        self.assertNotEqual(first, second)

    def test_index_fingerprint_uses_attached_file_manifest_without_iterating_chunks(self):
        class ManifestOnlyChunks(inspection_index.ChunkList):
            def __iter__(self):
                raise AssertionError("attached index manifest should avoid chunk iteration")

        chunks = ManifestOnlyChunks(
            [
                {
                    "chunk_id": "chunk-a",
                    "path": "main.py",
                    "language": "python",
                    "symbol": "main",
                    "line_start": 1,
                    "line_end": 2,
                    "content_hash": "sha256:a",
                    "classification": "internal",
                }
            ]
        )
        chunks._index_manifest = {"repo\000main.py": "sha256:file-a"}
        fingerprint = inspection_index.inspection_index_fingerprint("sha256:repo", chunks)

        self.assertTrue(fingerprint.startswith("sha256:"))

    def test_index_fingerprint_with_attached_manifest_changes_when_repository_state_changes(self):
        chunks = inspection_index.ChunkList(
            [
                {
                    "chunk_id": "chunk-a",
                    "path": "main.py",
                    "language": "python",
                    "symbol": "main",
                    "line_start": 1,
                    "line_end": 2,
                    "content_hash": "sha256:a",
                    "classification": "internal",
                }
            ]
        )
        chunks._index_manifest = {"repo\000main.py": "sha256:file-a"}

        first = inspection_index.inspection_index_fingerprint("sha256:repo-a", chunks)
        second = inspection_index.inspection_index_fingerprint("sha256:repo-b", chunks)

        self.assertNotEqual(first, second)

    def test_hotpath_index_fingerprint_changes_when_repository_state_changes_even_if_chunk_manifest_is_unchanged(self):
        chunk = {
            "chunk_id": "chunk-a",
            "path": "main.py",
            "language": "python",
            "symbol": "main",
            "line_start": 1,
            "line_end": 2,
            "content_hash": "sha256:a",
            "classification": "internal",
        }

        first = inspection_hotpath.inspection_index_fingerprint("sha256:repo-a", [chunk])
        second = inspection_hotpath.inspection_index_fingerprint("sha256:repo-b", [chunk])

        self.assertNotEqual(first, second)

    def test_hotpath_index_fingerprint_with_attached_manifest_changes_when_repository_state_changes(self):
        chunks = inspection_index.ChunkList(
            [
                {
                    "chunk_id": "chunk-a",
                    "path": "main.py",
                    "language": "python",
                    "symbol": "main",
                    "line_start": 1,
                    "line_end": 2,
                    "content_hash": "sha256:a",
                    "classification": "internal",
                }
            ]
        )
        chunks._index_manifest = {"repo\000main.py": "sha256:file-a"}

        first = inspection_hotpath.inspection_index_fingerprint("sha256:repo-a", chunks)
        second = inspection_hotpath.inspection_index_fingerprint("sha256:repo-b", chunks)

        self.assertNotEqual(first, second)

    def test_chunk_build_config_digest_ignores_content_hash_when_local_path_is_stable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            discovered_a = [{"id": "input_0", "type": "repo", "classification": "internal", "path": root, "content_hash": "git:first"}]
            discovered_b = [{"id": "input_0", "type": "repo", "classification": "internal", "path": root, "content_hash": "git:second"}]
            namespaces_a = {id(discovered_a[0]): "input_0"}
            namespaces_b = {id(discovered_b[0]): "input_0"}

            digest_a = inspection_index._chunk_build_config_digest(
                discovered_a,
                namespaces_a,
                max_lines=120,
                overlap=12,
                excluded_dir_names=None,
                excluded_paths=None,
            )
            digest_b = inspection_index._chunk_build_config_digest(
                discovered_b,
                namespaces_b,
                max_lines=120,
                overlap=12,
                excluded_dir_names=None,
                excluded_paths=None,
            )

        self.assertEqual(digest_a, digest_b)


class ServiceControlTests(unittest.TestCase):
    def test_signed_demand_spool_returns_only_fresh_authenticated_service(self):
        token = "control-secret"
        with tempfile.TemporaryDirectory() as temp_dir:
            request_dir = Path(temp_dir) / "requests"
            seen = {}

            def reconciler():
                deadline = time.time() + 2
                while time.time() < deadline:
                    paths = list(request_dir.glob("*.request.json")) if request_dir.exists() else []
                    if paths:
                        request_path = paths[0]
                        request = json.loads(request_path.read_text(encoding="utf-8"))
                        seen["request_path"] = request_path
                        seen["request"] = request
                        seen["mode"] = os.stat(request_path).st_mode & 0o777
                        now = datetime.now(timezone.utc)
                        service_record = service("v100-reasoning", ["chat"], 4)
                        service_record.update(
                            {
                                "heartbeat_at": service_control._timestamp(now),
                                "lease_expires_at": service_control._timestamp(now + timedelta(minutes=5)),
                            }
                        )
                        response = {
                            "schema": service_control.RESPONSE_SCHEMA,
                            "request_id": request["request_id"],
                            "demand_id": "demand-1",
                            "state": "ready",
                            "updated_at": service_control._timestamp(now),
                            "error": "",
                            "service": service_record,
                        }
                        response["signature"] = service_control.response_signature(token, response)
                        response_path = request_dir / f"{request['request_id']}.response.json"
                        seen["response_path"] = response_path
                        service_control._atomic_write_json(response_path, response)
                        return
                    time.sleep(0.01)

            thread = threading.Thread(target=reconciler, daemon=True)
            thread.start()
            record = service_control.request_service(
                {
                    "gpu_service_request_path": str(request_dir),
                    "gpu_service_control_token": token,
                    "gpu_service_startup_timeout_seconds": 2,
                    "gpu_service_health_interval_seconds": 0.1,
                },
                "v100-reasoning",
                "availability",
                "p40 synthesis failed",
            )
            thread.join(timeout=2)

            request = seen["request"]
            mode = seen["mode"]
            request_deleted = not seen["request_path"].exists()
            response_deleted = not seen["response_path"].exists()
        self.assertEqual(record["tier"], "v100-reasoning")
        self.assertEqual(record["state"], "ready")
        self.assertEqual(mode, 0o600)
        self.assertEqual(request["signature"], service_control.request_signature(token, request))
        self.assertTrue(request_deleted)
        self.assertTrue(response_deleted)

    def test_tampered_response_signature_is_rejected(self):
        response = {
            "schema": service_control.RESPONSE_SCHEMA,
            "request_id": "req-1",
            "demand_id": "demand-1",
            "state": "failed",
            "updated_at": "2026-01-01T00:00:00Z",
            "error": "capacity",
            "service": None,
        }
        response["signature"] = service_control.response_signature("token", response)
        response["error"] = "tampered"
        with self.assertRaisesRegex(gpu_client.GPUServiceError, "signature"):
            service_control.verify_response("token", response, "req-1")


class WorkerCLITests(unittest.TestCase):
    def test_evidence_mode_cli_writes_repo_inspection_v2(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo = root / "repo"
            repo.mkdir()
            (repo / "worker.py").write_text("def inspect_repo(query):\n    return query\n", encoding="utf-8")
            output = root / "output"
            job = {
                "job_id": "job-test",
                "task_type": "inspect_repo",
                "task_params": {"query": "Where is inspect_repo?", "mode": "evidence"},
                "constraints": {"final_pack_token_budget": 2_000},
                "output_schema": {"name": "repo_inspection_v2"},
            }
            manifest = {
                "input_refs": [
                    {"type": "repo", "uri": repo.as_uri(), "classification": "internal"}
                ]
            }
            job_path = root / "job.json"
            manifest_path = root / "manifest.json"
            job_path.write_text(json.dumps(job), encoding="utf-8")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            subprocess.run(
                [
                    sys.executable,
                    str(WORKER_DIR / "main.py"),
                    "--job-spec",
                    str(job_path),
                    "--input-manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(output),
                ],
                cwd=REPO_ROOT,
                check=True,
            )
            result = json.loads((output / "result.json").read_text(encoding="utf-8"))

        self.assertEqual(result["schema_name"], "repo_inspection_v2")
        self.assertEqual(result["schema_version"], "2.0.0")
        self.assertEqual(result["payload"]["mode"], "evidence")
        self.assertFalse(result["payload"]["quality"]["answer_ready"])
        self.assertNotIn("answer", result["payload"])


if __name__ == "__main__":
    unittest.main()

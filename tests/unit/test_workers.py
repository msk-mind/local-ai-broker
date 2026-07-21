import importlib.util
import json
import os
import select
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, rel_path: str):
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


document_summary = load_module("document_summary_worker", "workers/document-summary/main.py")
log_analysis = load_module("log_analysis_worker", "workers/log-analysis/main.py")
rag_compression = load_module("rag_compression_worker", "workers/rag-compression/main.py")
repo_summary = load_module("repo_summary_worker", "workers/repo-summary/main.py")
inspect_repo_worker = load_module("inspect_repo_worker", "workers/rag-compression/inspect_repo_worker.py")


class DocumentSummaryWorkerTests(unittest.TestCase):
    def test_build_summary_uses_opening_sentence(self):
        text = "First sentence. Second sentence.\n- bullet"
        lines = text.splitlines()
        words = text.split()

        summary = document_summary.build_summary(text, lines, words)

        self.assertIn("The document contains 6 words across 2 lines.", summary)
        self.assertIn("Opening sentence: First sentence.", summary)

    def test_derive_key_points_prefers_bullets(self):
        points = document_summary.derive_key_points(["", "- alpha", "* beta", "plain"])

        self.assertEqual(points, ["- alpha", "* beta"])


class LogAnalysisWorkerTests(unittest.TestCase):
    def test_derive_findings_classifies_and_limits(self):
        lines = [
            "fatal error: generated header missing",
            "undefined reference to demo_symbol",
            "ModuleNotFoundError: no module named demo",
            "FAILED demo_test",
        ]

        findings = log_analysis.derive_findings(lines)

        self.assertEqual(
            [item["code"] for item in findings],
            ["MISSING_GENERATED_HEADER", "UNDEFINED_REFERENCE", "MODULE_NOT_FOUND", "TEST_FAILURE"],
        )

    def test_build_excerpt_redacts_secrets(self):
        excerpt = log_analysis.build_excerpt(
            [
                "2026-01-01 fatal error token=secret-value",
                "2026-01-01 FAILED demo_test",
            ]
        )

        self.assertIn("token=[REDACTED]", excerpt)
        self.assertNotIn("secret-value", excerpt)


class RepoSummaryWorkerTests(unittest.TestCase):
    def test_build_manifest_skips_ignored_dirs_and_classifies_languages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "broker").mkdir()
            (root / "broker" / "main.go").write_text("package main\n", encoding="utf-8")
            (root / "README.md").write_text("# Demo\n", encoding="utf-8")
            (root / "node_modules").mkdir()
            (root / "node_modules" / "ignored.js").write_text("console.log('x')\n", encoding="utf-8")

            manifest = repo_summary.build_manifest(root)

        self.assertEqual(manifest["file_count"], 2)
        self.assertEqual(manifest["languages"]["go"], 1)
        self.assertEqual(manifest["languages"]["docs"], 1)
        self.assertEqual([item["path"] for item in manifest["files"]], ["README.md", "broker/main.go"])

    def test_derive_dependencies_reports_go_python_and_slurm(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "go.mod").write_text("module demo\n", encoding="utf-8")
            (root / "worker.py").write_text("print('demo')\n", encoding="utf-8")
            (root / "deploy" / "slurm").mkdir(parents=True)

            dependencies = repo_summary.derive_dependencies(root)

        self.assertEqual(
            dependencies,
            [
                {"name": "Go toolchain", "kind": "build_dependency"},
                {"name": "Python 3", "kind": "runtime_dependency"},
                {"name": "Slurm", "kind": "runtime_dependency"},
            ],
        )


class RAGCompressionWorkerTests(unittest.TestCase):
    def test_inspect_repo_requires_non_empty_query(self):
        with self.assertRaisesRegex(ValueError, "non-empty query"):
            rag_compression.validate_request("   ", "auto")

    def test_inspect_repo_rejects_unknown_mode(self):
        with self.assertRaisesRegex(ValueError, "auto, evidence, answer"):
            rag_compression.validate_request("find the call chain", "fast")

    def test_deterministic_runtime_is_never_promoted_to_real(self):
        adapter = rag_compression.build_runtime_adapter({})

        self.assertEqual(adapter["name"], "deterministic")
        self.assertEqual(adapter["backend_mode"], "heuristic")
        self.assertFalse(adapter["llm_available"])

    def test_v2_artifacts_are_compact_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            artifacts = rag_compression.write_repo_inspection_artifacts(
                output_dir,
                {
                    "evidence_pack": {"evidence": []},
                    "retrieval_result": {"selected": []},
                    "runtime_diagnostics": {"attempts": []},
                },
                "internal",
            )

        self.assertEqual(
            [item["artifact_type"] for item in artifacts],
            ["evidence_pack", "retrieval_result", "runtime_diagnostics"],
        )

    def test_inspect_repo_worker_applies_shared_cache_path_from_execution_plan(self):
        original = os.environ.get("BROKER_REPO_INSPECTION_SHARED_CACHE_DIR")
        try:
            os.environ.pop("BROKER_REPO_INSPECTION_SHARED_CACHE_DIR", None)
            inspect_repo_worker.apply_execution_plan_environment(
                {"repo_inspection_shared_cache_path": "/tmp/shared-repo-inspection-cache"}
            )
            self.assertEqual(
                os.environ.get("BROKER_REPO_INSPECTION_SHARED_CACHE_DIR"),
                "/tmp/shared-repo-inspection-cache",
            )
        finally:
            if original is None:
                os.environ.pop("BROKER_REPO_INSPECTION_SHARED_CACHE_DIR", None)
            else:
                os.environ["BROKER_REPO_INSPECTION_SHARED_CACHE_DIR"] = original

    def test_inspect_repo_worker_defaults_node_local_shared_cache_to_configured_cache_path(self):
        original = os.environ.get("BROKER_REPO_INSPECTION_SHARED_CACHE_DIR")
        try:
            os.environ.pop("BROKER_REPO_INSPECTION_SHARED_CACHE_DIR", None)
            inspect_repo_worker.apply_execution_plan_environment(
                {
                    "repo_inspection_cache_path": "/tmp/repo-inspection-cache",
                    "repo_inspection_use_node_local_cache": True,
                }
            )
            self.assertEqual(
                os.environ.get("BROKER_REPO_INSPECTION_SHARED_CACHE_DIR"),
                "/tmp/repo-inspection-cache",
            )
        finally:
            if original is None:
                os.environ.pop("BROKER_REPO_INSPECTION_SHARED_CACHE_DIR", None)
            else:
                os.environ["BROKER_REPO_INSPECTION_SHARED_CACHE_DIR"] = original

    def test_inspect_repo_worker_records_startup_phase_timings(self):
        payload = {"runtime": {}}
        inspect_repo_worker.record_phase_timings(
            payload,
            process_bootstrap_started=0.0,
            worker_started=0.010,
            after_parse_args=0.015,
            after_load_inputs=0.020,
            after_import_validate=0.025,
            after_validate=0.030,
            after_discover=0.040,
            after_import_prefetch=0.050,
            after_prefetch=0.060,
            after_cached_probe=0.070,
            after_import_pipeline=0.080,
            after_run=0.100,
            after_artifacts=0.110,
            completed_at=0.120,
            cache_hit=False,
        )

        timings = payload["runtime"]["worker_phase_timings_ms"]
        self.assertIn("process_bootstrap", timings)
        self.assertIn("parse_args", timings)
        self.assertIn("load_job_inputs", timings)
        self.assertIn("import_validate_request", timings)
        self.assertIn("import_prefetch_helpers", timings)
        self.assertIn("import_run_inspection", timings)

    def test_inspect_repo_worker_annotates_runtime_mode(self):
        payload = {"runtime": {}}

        inspect_repo_worker.annotate_runtime_mode(payload, daemon_mode=True)
        self.assertEqual(payload["runtime"]["local_backend_mode"], "warm_daemon")
        self.assertTrue(payload["runtime"]["warm_daemon_active"])

        inspect_repo_worker.annotate_runtime_mode(payload, daemon_mode=False)
        self.assertEqual(payload["runtime"]["local_backend_mode"], "direct_worker")
        self.assertFalse(payload["runtime"]["warm_daemon_active"])

    def test_process_warm_request_cleans_markers_after_cached_completion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "job"
            output_dir.mkdir()
            request_path = root / "request.working"
            heartbeat_path = output_dir / "heartbeat.json"
            (output_dir / "warm-request.marker").write_text("queued", encoding="utf-8")
            (output_dir / "cancel.request").write_text("cancel", encoding="utf-8")
            request_path.write_text(
                json.dumps(
                    {
                        "job_id": "job-1",
                        "job_spec_path": str(output_dir / "job_spec.json"),
                        "execution_plan_path": str(output_dir / "execution_plan.json"),
                        "input_manifest_path": str(output_dir / "input_manifest.json"),
                        "output_dir": str(output_dir),
                        "heartbeat_path": str(heartbeat_path),
                    }
                ),
                encoding="utf-8",
            )
            (output_dir / "job_spec.json").write_text(json.dumps({"job_id": "job-1"}), encoding="utf-8")

            with mock.patch.object(inspect_repo_worker, "run_staged_job", return_value=0):
                inspect_repo_worker.process_warm_request(request_path)

            self.assertFalse((output_dir / "warm-request.marker").exists())
            self.assertFalse((output_dir / "cancel.request").exists())

    def test_recover_warm_requests_moves_working_back_to_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            request_dir = Path(temp_dir)
            working_path = request_dir / "job_1.working"
            working_path.write_text("{}", encoding="utf-8")

            inspect_repo_worker.recover_warm_requests(request_dir)

            self.assertFalse(working_path.exists())
            self.assertTrue((request_dir / "job_1.json").exists())

    def test_write_warm_daemon_heartbeat_emits_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            spool_path = Path(temp_dir)

            inspect_repo_worker.write_warm_daemon_heartbeat(spool_path)

            heartbeat = json.loads((spool_path / "daemon-heartbeat.json").read_text(encoding="utf-8"))
            self.assertEqual(heartbeat["state"], "running")
            self.assertEqual(heartbeat["pid"], os.getpid())

    def test_warm_daemon_wakeup_socket_receives_signal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            spool_path = Path(temp_dir)
            listener = inspect_repo_worker.open_warm_daemon_wakeup_socket(spool_path)
            try:
                socket_path = inspect_repo_worker.warm_daemon_wakeup_socket_path(spool_path)
                client = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
                try:
                    client.connect(str(socket_path))
                    client.send(b"job_1.json")
                finally:
                    client.close()

                ready, _, _ = select.select([listener], [], [], 1.0)
                self.assertEqual(ready, [listener])
                messages = inspect_repo_worker.wait_for_warm_daemon_wakeup(listener, 0.0)
                self.assertEqual(messages, ["job_1.json"])
                ready, _, _ = select.select([listener], [], [], 0.0)
                self.assertEqual(ready, [])
            finally:
                inspect_repo_worker.close_warm_daemon_wakeup_socket(listener, spool_path)
                self.assertFalse(inspect_repo_worker.warm_daemon_wakeup_socket_path(spool_path).exists())

    def test_iter_warm_request_paths_prioritizes_wakeup_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            request_dir = Path(temp_dir)
            (request_dir / "b.json").write_text("{}", encoding="utf-8")
            (request_dir / "a.json").write_text("{}", encoding="utf-8")

            paths = list(
                inspect_repo_worker.iter_warm_request_paths(
                    request_dir,
                    ["b.json", "b.json", "../ignored.json", "a.json"],
                )
            )

            self.assertEqual(paths, [request_dir / "b.json", request_dir / "a.json"])

    def test_warm_daemon_poll_interval_defaults_and_accepts_override(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            self.assertEqual(inspect_repo_worker.warm_daemon_poll_interval_seconds(), 0.01)

        with mock.patch.dict(
            os.environ,
            {"BROKER_LOCAL_INSPECT_REPO_WARM_POLL_INTERVAL_SECONDS": "0.002"},
            clear=False,
        ):
            self.assertEqual(inspect_repo_worker.warm_daemon_poll_interval_seconds(), 0.002)

        with mock.patch.dict(
            os.environ,
            {"BROKER_LOCAL_INSPECT_REPO_WARM_POLL_INTERVAL_SECONDS": "invalid"},
            clear=False,
        ):
            self.assertEqual(inspect_repo_worker.warm_daemon_poll_interval_seconds(), 0.01)

    def test_preload_warm_daemon_modules_primes_cached_imports(self):
        inspect_repo_worker._VALIDATE_REQUEST = None
        inspect_repo_worker._PREPARE_PREFETCHED_STATE = None
        inspect_repo_worker._CACHED_LEXICAL_FALLBACK_FROM_CONTEXT = None
        inspect_repo_worker._RUN_INSPECTION = None

        inspect_repo_worker.preload_warm_daemon_modules()

        self.assertIsNotNone(inspect_repo_worker._VALIDATE_REQUEST)
        self.assertIsNotNone(inspect_repo_worker._PREPARE_PREFETCHED_STATE)
        self.assertIsNotNone(inspect_repo_worker._CACHED_LEXICAL_FALLBACK_FROM_CONTEXT)
        self.assertIsNotNone(inspect_repo_worker._RUN_INSPECTION)

if __name__ == "__main__":
    unittest.main()

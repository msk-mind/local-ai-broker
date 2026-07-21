import importlib.util
import json
import tempfile
import textwrap
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]


def load_evaluator():
    path = HERE / "evaluate.py"
    spec = importlib.util.spec_from_file_location("inspect_repo_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


evaluator = load_evaluator()


class InspectRepoGoldenEvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = evaluator.load_suite(HERE / "golden_queries.json")
        cls.query_map = {query["id"]: query for query in cls.suite["queries"]}

    def fixture_results(self):
        return evaluator.load_result_records(HERE / "fixtures" / "cpu_results.json", self.query_map)

    def test_suite_has_thirty_queries_and_required_gates(self):
        self.assertEqual(len(self.suite["queries"]), 30)
        self.assertEqual(
            self.suite["thresholds"],
            {
                "recall_at_10": 0.9,
                "mrr": 0.75,
                "citation_precision": 0.95,
                "missing_evidence_refs": 0,
            },
        )
        tagged = {tag for query in self.suite["queries"] for tag in query.get("tags", [])}
        self.assertIn("root-cause-regression", tagged)
        self.assertIn("mcp-call-chain-regression", tagged)
        self.assertEqual({query["mode"] for query in self.suite["queries"]}, {"auto", "evidence", "answer"})

    def test_cpu_fixture_baseline_passes_all_quality_gates(self):
        report = evaluator.evaluate(self.suite, self.fixture_results())

        self.assertTrue(report["passed"], report["failures"])
        self.assertGreaterEqual(report["metrics"]["recall_at_10"], 0.9)
        self.assertGreaterEqual(report["metrics"]["mrr"], 0.75)
        self.assertGreaterEqual(report["metrics"]["citation_precision"], 0.95)
        self.assertEqual(report["metrics"]["missing_evidence_refs"], 0)
        self.assertEqual(report["ordering_failures"], [])

    def test_unknown_finding_citation_fails_precision_and_missing_ref_gates(self):
        query = self.query_map["root_cause_false_real_local"]
        result = evaluator.compact_fixture_result(
            {
                "id": query["id"],
                "answer_ready": True,
                "ranked_paths": query["relevant_paths"],
            },
            query,
        )
        result["payload"]["findings"][0]["evidence_refs"] = ["ev_not_released"]

        report = evaluator.evaluate(self.suite, {query["id"]: result}, {query["id"]})

        self.assertFalse(report["passed"])
        self.assertEqual(report["metrics"]["citation_precision"], 0.0)
        self.assertEqual(report["metrics"]["missing_evidence_refs"], 1)

    def test_low_recall_and_rank_fail_numeric_gates(self):
        query = self.query_map["root_cause_false_real_local"]
        result = evaluator.compact_fixture_result(
            {"id": query["id"], "ranked_paths": ["README.md"]},
            query,
        )

        report = evaluator.evaluate(self.suite, {query["id"]: result}, {query["id"]})

        self.assertFalse(report["passed"])
        self.assertEqual(report["metrics"]["recall_at_10"], 0.0)
        self.assertEqual(report["metrics"]["mrr"], 0.0)
        self.assertTrue(any(failure.startswith("Recall@10") for failure in report["failures"]))
        self.assertTrue(any(failure.startswith("MRR") for failure in report["failures"]))

    def test_answer_ready_result_requires_all_three_gpu_quality_states(self):
        query = self.query_map["cpu_cannot_be_answer_ready"]
        result = evaluator.compact_fixture_result(
            {
                "id": query["id"],
                "answer_ready": True,
                "ranked_paths": query["relevant_paths"],
            },
            query,
        )
        result["payload"]["quality"]["retrieval"] = "lexical_degraded"

        report = evaluator.evaluate(self.suite, {query["id"]: result}, {query["id"]})

        self.assertFalse(report["passed"])
        self.assertTrue(any("quality.retrieval='gpu'" in error for error in report["contract_errors"]))

    def test_mcp_and_service_must_rank_above_rag_worker(self):
        query = self.query_map["mcp_inspect_repo_call_chain"]
        result = evaluator.compact_fixture_result(
            {
                "id": query["id"],
                "answer_ready": True,
                "ranked_paths": [
                    "workers/rag-compression/main.py",
                    "broker/pkg/mcp/server.go",
                    "broker/pkg/service/service_submission.go",
                    "broker/pkg/service/service_execution.go",
                    "broker/pkg/tasks/catalog.go",
                ],
            },
            query,
        )

        report = evaluator.evaluate(self.suite, {query["id"]: result}, {query["id"]})

        self.assertFalse(report["passed"])
        self.assertEqual(len(report["ordering_failures"]), 2)

    def test_mcp_ordering_does_not_require_irrelevant_rag_worker_candidate(self):
        query = self.query_map["mcp_inspect_repo_call_chain"]
        result = evaluator.compact_fixture_result(
            {
                "id": query["id"],
                "answer_ready": True,
                "ranked_paths": query["relevant_paths"],
            },
            query,
        )

        report = evaluator.evaluate(self.suite, {query["id"]: result}, {query["id"]})

        self.assertTrue(report["passed"], report["failures"])
        self.assertEqual(report["ordering_failures"], [])

    def test_answer_mode_accepts_structured_failure_with_attempt_history(self):
        query = dict(self.query_map["repo_inspection_v2_schema"])
        query["mode"] = "answer"
        envelope = {"schema_name": "repo_inspection_v2", "schema_version": "2.0.0"}
        payload = {
            "mode": "answer",
            "query": query["query"],
            "findings": [],
            "evidence": [{"id": "ev_001", "source_refs": [{"path": "broker/pkg/schemas/validate.go"}]}],
            "quality": {
                "result": "failed",
                "retrieval": "gpu",
                "reranking": "gpu",
                "synthesis": "failed",
                "answer_ready": False,
            },
            "warnings": [],
            "provenance": {},
            "retrieval": {},
            "runtime": {
                "attempts": [
                    {"operation": "synthesis", "tier": "p40-synthesis", "status": "failed", "gpu_count": 1, "failure_category": "service_failure"},
                    {"operation": "synthesis", "tier": "v100-reasoning", "status": "failed", "gpu_count": 4, "failure_category": "timeout"},
                    {"operation": "synthesis", "tier": "a100-single", "status": "failed", "gpu_count": 1, "failure_category": "service_failure"},
                ]
            },
        }

        errors, valid, invalid, missing = evaluator.contract_and_citations(envelope, payload, query)

        self.assertEqual(errors, [])
        self.assertEqual((valid, invalid, missing), (0, 0, 0))

    def test_answer_ready_requires_successful_ordered_gpu_attempts(self):
        query = self.query_map["cpu_cannot_be_answer_ready"]
        result = evaluator.compact_fixture_result(
            {"id": query["id"], "answer_ready": True, "ranked_paths": query["relevant_paths"]},
            query,
        )
        result["payload"]["runtime"]["attempts"] = []

        report = evaluator.evaluate(self.suite, {query["id"]: result}, {query["id"]})

        self.assertFalse(report["passed"])
        self.assertTrue(any("successful GPU retrieval" in error for error in report["contract_errors"]))

    def test_staged_worker_cli_adapter_runs_without_gpu(self):
        query = self.query_map["staged_worker_cli_contract"]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            root.joinpath("README.md").write_text("fixture repository\n", encoding="utf-8")
            worker = root / "fake_worker.py"
            worker.write_text(
                textwrap.dedent(
                    """
                    import argparse
                    import json
                    from pathlib import Path

                    parser = argparse.ArgumentParser()
                    parser.add_argument("--job-spec")
                    parser.add_argument("--execution-plan")
                    parser.add_argument("--input-manifest")
                    parser.add_argument("--output-dir")
                    args = parser.parse_args()
                    job = json.loads(Path(args.job_spec).read_text(encoding="utf-8"))
                    output = Path(args.output_dir)
                    output.mkdir(parents=True)
                    payload = {
                        "schema_name": "repo_inspection_v2",
                        "schema_version": "2.0.0",
                        "payload": {
                            "mode": job["task_params"]["mode"],
                            "query": job["task_params"]["query"],
                            "findings": [],
                            "evidence": [{"id": "ev_001", "source_refs": [{"path": "workers/rag-compression/main.py"}]}],
                            "quality": {"result": "evidence_only", "retrieval": "lexical_degraded", "reranking": "unavailable", "synthesis": "not_requested", "answer_ready": False},
                            "warnings": ["GPU_RETRIEVAL_UNAVAILABLE"],
                            "provenance": {},
                            "retrieval": {"ranked_candidates": [{"path": "workers/rag-compression/main.py"}]},
                            "runtime": {"attempts": []}
                        }
                    }
                    output.joinpath("result.json").write_text(json.dumps(payload), encoding="utf-8")
                    """
                ),
                encoding="utf-8",
            )

            raw = evaluator.invoke_worker_cli(worker, query, root, 10)
            envelope, payload = evaluator.unwrap_result(raw)

        self.assertEqual(envelope["schema_name"], "repo_inspection_v2")
        self.assertEqual(payload["query"], query["query"])
        self.assertEqual(payload["quality"]["result"], "evidence_only")


if __name__ == "__main__":
    unittest.main()

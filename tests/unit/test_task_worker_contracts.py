import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


TASKS = {
    "document_summary": ("workers/document-summary/main.py", "document_summary_v1", "file"),
    "log_analysis": ("workers/log-analysis/main.py", "log_analysis_v1", "file"),
    "repo_summary": ("workers/repo-summary/main.py", "repo_summary_v1", "directory"),
    "rag_compress": ("workers/rag-compression/main.py", "rag_evidence_pack_v1", "file"),
    "debug_with_local_context": ("workers/rag-compression/main.py", "debug_evidence_pack_v1", "repo"),
    "summarize_logs": ("workers/rag-compression/main.py", "log_evidence_pack_v1", "log"),
    "inspect_repo": ("workers/rag-compression/main.py", "repo_inspection_v2", "repo"),
    "propose_patch": ("workers/rag-compression/main.py", "patch_proposal_pack_v1", "repo"),
}


class TaskWorkerContractTests(unittest.TestCase):
    def run_worker(self, task_type, *, with_input=True):
        script, schema, input_type = TASKS[task_type]
        with tempfile.TemporaryDirectory(prefix="local-ai-worker-contract-") as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "output"
            output_dir.mkdir()
            input_path = root / ("repo" if input_type == "repo" or input_type == "directory" else "input.txt")
            if input_type in {"repo", "directory"}:
                input_path.mkdir()
                (input_path / "README.md").write_text("# contract fixture\n", encoding="utf-8")
                (input_path / "worker.py").write_text("def retry_job(job_id):\n    return job_id\n", encoding="utf-8")
                if task_type == "debug_with_local_context":
                    (input_path / "error.log").write_text("fatal error: fixture failure\n", encoding="utf-8")
            else:
                input_path.write_text(
                    "fatal error: fixture failure\n" if input_type == "log" else "A contract test document.\n- one\n- two\n",
                    encoding="utf-8",
                )

            refs = []
            if with_input:
                refs.append({"type": input_type, "uri": input_path.as_uri(), "classification": "internal"})
            task_params = {}
            if task_type in {"rag_compress", "inspect_repo"}:
                task_params.update({"query": "trace retry_job", "mode": "evidence"} if task_type == "inspect_repo" else {"query": "summarize the fixture"})
            elif task_type == "debug_with_local_context":
                task_params["problem"] = "Explain the fixture failure"
            elif task_type == "propose_patch":
                task_params["problem"] = "Propose a safe fix for the fixture"

            job_spec = {
                "job_id": "worker-contract",
                "task_type": task_type,
                "task_params": task_params,
                "constraints": {"final_pack_token_budget": 2048},
                "output_schema": {"name": schema},
            }
            job_spec_path = root / "job_spec.json"
            manifest_path = root / "input_manifest.json"
            execution_plan_path = root / "execution_plan.json"
            heartbeat_path = output_dir / "heartbeat.json"
            job_spec_path.write_text(json.dumps(job_spec), encoding="utf-8")
            manifest_path.write_text(json.dumps({"input_refs": refs}), encoding="utf-8")
            execution_plan_path.write_text(json.dumps({}), encoding="utf-8")

            command = [
                sys.executable,
                str(REPO_ROOT / script),
                "--job-spec", str(job_spec_path),
                "--input-manifest", str(manifest_path),
                "--output-dir", str(output_dir),
                "--heartbeat-path", str(heartbeat_path),
                "--completion-socket-path", str(root / "completion.sock"),
            ]
            if task_type not in {"document_summary", "log_analysis", "repo_summary"}:
                command.extend(["--execution-plan", str(execution_plan_path)])
            result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, timeout=45)
            outputs = {}
            if result.returncode == 0:
                outputs["result"] = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
                outputs["artifacts"] = json.loads((output_dir / "artifacts.json").read_text(encoding="utf-8"))
                outputs["heartbeat"] = json.loads((output_dir / "heartbeat.json").read_text(encoding="utf-8"))
            return result, outputs, schema

    def test_all_catalog_tasks_produce_contract_outputs(self):
        for task_type in TASKS:
            with self.subTest(task_type=task_type):
                result, outputs, schema = self.run_worker(task_type)
                self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
                result_payload = outputs["result"]
                self.assertEqual(result_payload["schema_name"], schema)
                self.assertEqual(result_payload["schema_version"], "1.0.0" if task_type != "inspect_repo" else "2.0.0")
                self.assertIsInstance(result_payload.get("payload"), dict)
                self.assertIsInstance(outputs["artifacts"], list)
                heartbeat = outputs["heartbeat"]
                self.assertEqual(heartbeat["state"], "completed")
                self.assertEqual(heartbeat["percent"], 100)

    def test_all_catalog_tasks_reject_empty_input_manifest(self):
        for task_type in TASKS:
            with self.subTest(task_type=task_type):
                result, _, _ = self.run_worker(task_type, with_input=False)
                self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()

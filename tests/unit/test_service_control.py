import json
import os
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKER_DIR = ROOT / "workers" / "rag-compression"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

import gpu_client
import service_control


class ServiceControlTests(unittest.TestCase):
    def test_terminal_oom_response_preserves_category_and_is_consumed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            request_dir = Path(temp_dir)
            token = "control-token"
            seen = {}

            def reconciler():
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    paths = list(request_dir.glob("*.request.json"))
                    if paths:
                        request = json.loads(paths[0].read_text(encoding="utf-8"))
                        seen["request_id"] = request["request_id"]
                        response = {
                            "schema": service_control.RESPONSE_SCHEMA,
                            "request_id": request["request_id"],
                            "demand_id": "demand-oom",
                            "state": "failed",
                            "failure_category": "oom",
                            "updated_at": service_control._timestamp(datetime.now(timezone.utc)),
                            "error": "Slurm OUT_OF_MEMORY",
                            "service": None,
                            "service_diagnostics": {
                                "tier": "v100-reasoning",
                                "slurm_job_id": "v100-job-42",
                                "gpu": {"type": "v100", "count": 4},
                                "model_profile": "v100-reasoning-profile",
                            },
                        }
                        response["signature"] = service_control.response_signature(token, response)
                        service_control._atomic_write_json(
                            request_dir / f"{request['request_id']}.response.json", response
                        )
                        return
                    time.sleep(0.01)

            thread = threading.Thread(target=reconciler, daemon=True)
            thread.start()
            with self.assertRaises(gpu_client.GPUServiceError) as raised:
                service_control.request_service(
                    {
                        "gpu_service_request_path": str(request_dir),
                        "gpu_service_control_token": token,
                        "gpu_service_startup_timeout_seconds": 2,
                        "gpu_service_health_interval_seconds": 0.1,
                    },
                    "v100-reasoning",
                    "service_failure",
                    "P40 failed",
                )
            thread.join(timeout=2)
            self.assertEqual(raised.exception.category, "oom")
            self.assertEqual(
                raised.exception.service_diagnostics,
                {
                    "tier": "v100-reasoning",
                    "slurm_job_id": "v100-job-42",
                    "gpu": {"type": "v100", "count": 4},
                    "model_profile": "v100-reasoning-profile",
                },
            )
            request_id = seen["request_id"]
            self.assertFalse((request_dir / f"{request_id}.request.json").exists())
            self.assertFalse((request_dir / f"{request_id}.response.json").exists())

    def test_fire_and_forget_p40_failure_report_is_signed_and_private(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            request_dir = Path(temp_dir)
            token = "control-token"
            emitted = service_control.report_p40_service_failure(
                {
                    "gpu_service_request_path": str(request_dir),
                    "gpu_service_control_token": token,
                },
                {"id": "gpu-p40-synthesis-deadbeef", "tier": "p40-synthesis"},
                "service_failure",
                "authenticated endpoint failed",
            )
            self.assertTrue(emitted)
            paths = list(request_dir.glob("*.failure.json"))
            self.assertEqual(len(paths), 1)
            payload = json.loads(paths[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], service_control.FAILURE_REPORT_SCHEMA)
            self.assertEqual(payload["service_id"], "gpu-p40-synthesis-deadbeef")
            self.assertEqual(
                payload["signature"], service_control.failure_report_signature(token, payload)
            )
            self.assertEqual(os.stat(paths[0]).st_mode & 0o777, 0o600)

            self.assertFalse(
                service_control.report_p40_service_failure(
                    {
                        "gpu_service_request_path": str(request_dir),
                        "gpu_service_control_token": token,
                    },
                    {"id": "gpu-a100-single-deadbeef", "tier": "a100-single"},
                    "service_failure",
                    "not a P40 lease",
                )
            )
            self.assertEqual(len(list(request_dir.glob("*.failure.json"))), 1)


if __name__ == "__main__":
    unittest.main()

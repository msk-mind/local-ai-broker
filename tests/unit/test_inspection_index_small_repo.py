import importlib.util
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


inspection_index = load_module("inspection_index", "workers/rag-compression/inspection_index.py")


class SmallRepoInventoryDeltaTests(unittest.TestCase):
    def test_cached_git_source_files_uses_status_path_for_small_repo(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.py").write_text("print('a')\n", encoding="utf-8")
            (root / "b.py").write_text("print('b')\n", encoding="utf-8")
            previous_record = {
                "git_top": str(root),
                "scope_rel": ".",
                "scope_oid": "head-1",
                "filter_key": "filter-1",
                "files": ["a.py", "b.py"],
            }

            with (
                mock.patch.object(inspection_index, "_git_top", return_value=root),
                mock.patch.object(inspection_index, "_git_scope_rel", return_value="."),
                mock.patch.object(inspection_index, "_discovery_filter_key", return_value="filter-1"),
                mock.patch.object(inspection_index, "_git_scope_inventory_identity", return_value="head-1"),
                mock.patch.object(inspection_index, "_git_scope_status_paths", return_value=set()) as status_paths,
                mock.patch.object(
                    inspection_index,
                    "_scoped_git_inventory_delta_paths",
                    side_effect=AssertionError("small repos should skip inventory-delta subprocess path"),
                ),
            ):
                files, record = inspection_index._cached_git_source_files(
                    root,
                    set(),
                    previous_record=previous_record,
                    prefer_inventory_delta=True,
                )

            self.assertEqual([path.name for path in files], ["a.py", "b.py"])
            self.assertEqual(record["files"], ["a.py", "b.py"])
            self.assertEqual(status_paths.call_count, 1)


if __name__ == "__main__":
    unittest.main()

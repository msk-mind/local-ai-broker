"""Pytest isolation hooks for process-local broker worker caches."""


def pytest_runtest_setup(item):
    for module_name in (
        "inspection_index",
        "inspection_hotpath",
        "inspection_pipeline",
    ):
        module = item.module.__dict__.get(module_name)
        reset = getattr(module, "reset_process_caches", None)
        if callable(reset):
            reset()

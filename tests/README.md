# Tests

Pytest tests for the fixes added on `fix/functional-bugs`.

```bash
# from the repo root, in a venv with agent-bench installed:
pip install pytest pyyaml
pytest tests/ -v
```

Each file targets one slice:

| file | what it covers |
|---|---|
| `test_agent_console.py` | Agent console output / summary formatting |
| `test_agent_sampling.py` | Agent turn-plan sampling distributions |
| `test_viewer_loader.py` | Viewer scan/load of `benchmarks/` run directories |
| `test_viewer_csv_export.py` | Viewer CSV export shape |
| `test_package_layout.py` | `agent/workloads/` is accessible after install; bundled workload YAMLs are present and non-empty |

These tests don't need network access or a real inference server — they exercise the pure-Python code paths via fakes/mocks. The wandb client integration and aiohttp HTTP loop are intentionally out of scope.

"""
Tests for the /export/<bench_id>.csv Flask route on the dashboard.

The route is wired into the Dash app's underlying Flask server so users can
pull per-benchmark CSV data without going through the live-chart UI.
Useful for offline plotting / diffing / sharing.
"""
import json
from pathlib import Path

import pytest

# Skip cleanly when the [viewer] extras aren't installed.
pytest.importorskip("dash")

from agent.viewer import create_dash_app  # noqa: E402


def _write_run(root: Path, name: str, timestamp: str, points: list) -> Path:
    run_dir = root / name / timestamp
    run_dir.mkdir(parents=True)
    metrics = run_dir / "metrics.jsonl"
    with open(metrics, "w") as f:
        for p in points:
            f.write(json.dumps(p) + "\n")
    return run_dir


@pytest.fixture
def client_with_runs(tmp_path):
    _write_run(tmp_path, "run-a", "2026-01-01-00-00-00", [
        {"elapsed_seconds": 1.0, "generation_tps": 50.0, "in_flight": 2},
        {"elapsed_seconds": 2.0, "generation_tps": 75.0, "in_flight": 4},
        {"elapsed_seconds": 3.0, "generation_tps": 100.0, "in_flight": 4},
    ])
    app = create_dash_app(tmp_path)
    flask_app = app.server
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


def test_export_returns_csv_with_default_columns(client_with_runs):
    r = client_with_runs.get("/export/run-a/2026-01-01-00-00-00.csv")
    assert r.status_code == 200
    assert r.mimetype == "text/csv"
    body = r.get_data(as_text=True)
    header, *data_rows = body.strip().split("\n")
    assert "elapsed_seconds" in header
    assert "generation_tps" in header
    # 3 metric points -> 3 data rows
    assert len(data_rows) == 3


def test_export_respects_fields_filter(client_with_runs):
    r = client_with_runs.get(
        "/export/run-a/2026-01-01-00-00-00.csv?fields=elapsed_seconds,generation_tps"
    )
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    header = body.split("\n", 1)[0]
    assert header == "elapsed_seconds,generation_tps"


def test_export_drops_unknown_fields(client_with_runs):
    """Whitelist enforcement: unknown column names are silently dropped."""
    r = client_with_runs.get(
        "/export/run-a/2026-01-01-00-00-00.csv?fields=generation_tps,evil_eval"
    )
    body = r.get_data(as_text=True)
    header = body.split("\n", 1)[0]
    assert header == "generation_tps"


def test_export_404_for_missing_benchmark(client_with_runs):
    r = client_with_runs.get("/export/no-such-run/no-such-timestamp.csv")
    assert r.status_code == 404


def test_export_filename_safe_for_disposition(client_with_runs):
    """The Content-Disposition filename has `/` replaced with `__`."""
    r = client_with_runs.get("/export/run-a/2026-01-01-00-00-00.csv")
    cd = r.headers.get("Content-Disposition", "")
    assert "run-a__2026-01-01-00-00-00.csv" in cd


def test_empty_metrics_file_returns_header_only(tmp_path):
    """A run with empty metrics.jsonl should still return a valid CSV header."""
    _write_run(tmp_path, "empty", "2026-01-01-00-00-00", [])
    app = create_dash_app(tmp_path)
    flask_app = app.server
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()
    r = client.get("/export/empty/2026-01-01-00-00-00.csv")
    assert r.status_code == 200
    body = r.get_data(as_text=True).strip().split("\n")
    assert len(body) == 1  # header only

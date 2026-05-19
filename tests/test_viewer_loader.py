"""
Tests for the viewer's data-loading layer (scan_benchmarks, load_metrics).

The viewer's mega-callback isn't exercised here (would require booting a Dash
app + driving the Plotly callback API), but the pure-Python loaders that feed
it are. These are also the parts most likely to break when a new metric field
gets added to metrics.jsonl — the loader silently dropped a real bug like that
in the past.
"""
import json
from pathlib import Path


# The loaders live in agent/viewer_data.py and have NO Dash dependency,
# so these tests can run in environments without the [viewer] extras.
from agent.viewer_data import load_metrics, scan_benchmarks


def _write_run(root: Path, name: str, timestamp: str, points: list) -> Path:
    """Build a fake benchmarks/<name>/<timestamp>/metrics.jsonl tree."""
    run_dir = root / name / timestamp
    run_dir.mkdir(parents=True)
    metrics = run_dir / "metrics.jsonl"
    with open(metrics, "w") as f:
        for p in points:
            f.write(json.dumps(p) + "\n")
    return run_dir


# ---------- scan_benchmarks ----------


class TestScanBenchmarks:

    def test_missing_root_returns_empty(self, tmp_path):
        traces = scan_benchmarks(tmp_path / "does-not-exist")
        assert traces == {}

    def test_empty_root_returns_empty(self, tmp_path):
        traces = scan_benchmarks(tmp_path)
        assert traces == {}

    def test_picks_up_run_with_metrics_jsonl(self, tmp_path):
        _write_run(tmp_path, "myrun", "2026-01-01-00-00-00",
                   [{"elapsed_seconds": 1.0}, {"elapsed_seconds": 2.0}])
        traces = scan_benchmarks(tmp_path)
        assert len(traces) == 1
        key = "myrun/2026-01-01-00-00-00"
        assert key in traces
        assert traces[key].label == "myrun"
        assert traces[key].file_path.name == "metrics.jsonl"

    def test_ignores_dirs_without_metrics_jsonl(self, tmp_path):
        # name/ with timestamp/ but no metrics.jsonl -> skipped
        (tmp_path / "noisy" / "2026-01-01-00-00-00").mkdir(parents=True)
        # a non-directory file at top level -> skipped
        (tmp_path / "stray.txt").write_text("hi")
        # a valid run
        _write_run(tmp_path, "real", "2026-01-02-00-00-00", [{"elapsed_seconds": 0.5}])
        traces = scan_benchmarks(tmp_path)
        assert set(traces) == {"real/2026-01-02-00-00-00"}

    def test_multiple_runs_under_same_name(self, tmp_path):
        _write_run(tmp_path, "sweep", "2026-01-01-00-00-00", [{"elapsed_seconds": 1}])
        _write_run(tmp_path, "sweep", "2026-01-02-00-00-00", [{"elapsed_seconds": 1}])
        traces = scan_benchmarks(tmp_path)
        assert len(traces) == 2
        assert all(t.label == "sweep" for t in traces.values())


# ---------- load_metrics ----------


class TestLoadMetrics:

    def test_missing_file_returns_empty(self, tmp_path):
        pts, pos = load_metrics(tmp_path / "missing.jsonl")
        assert pts == []
        assert pos == 0

    def test_parses_well_formed_lines(self, tmp_path):
        run = _write_run(tmp_path, "ok", "2026-01-01-00-00-00", [
            {"elapsed_seconds": 1.0, "in_flight": 2, "generation_tps": 50.5,
             "cache_hit_rate": 0.8, "requests_completed": 3},
            {"elapsed_seconds": 2.0, "in_flight": 4, "generation_tps": 75.0,
             "cache_hit_rate": 0.9, "requests_completed": 7},
        ])
        pts, pos = load_metrics(run / "metrics.jsonl")
        assert len(pts) == 2
        assert pts[0].elapsed_seconds == 1.0
        assert pts[0].in_flight == 2
        assert pts[1].generation_tps == 75.0
        assert pts[1].requests_completed == 7
        assert pos > 0  # advanced past EOF

    def test_unknown_fields_dont_break(self, tmp_path):
        """metrics.jsonl might add new fields; loader should ignore them."""
        run = _write_run(tmp_path, "future", "2026-01-01-00-00-00", [
            {"elapsed_seconds": 1.0, "in_flight": 1,
             "future_field_no_one_knows_about": 999}
        ])
        pts, _ = load_metrics(run / "metrics.jsonl")
        assert len(pts) == 1
        assert pts[0].in_flight == 1

    def test_missing_fields_fall_back_to_defaults(self, tmp_path):
        """Most fields are optional; a sparse line should still parse."""
        run = _write_run(tmp_path, "sparse", "2026-01-01-00-00-00", [
            {"elapsed_seconds": 5.0}  # only one field
        ])
        pts, _ = load_metrics(run / "metrics.jsonl")
        assert pts[0].elapsed_seconds == 5.0
        assert pts[0].in_flight == 0       # default
        assert pts[0].generation_tps == 0  # default
        assert pts[0].cache_hit_rate == 0  # default

    def test_malformed_lines_are_skipped_silently(self, tmp_path):
        """A bad line shouldn't poison the whole file."""
        metrics = tmp_path / "metrics.jsonl"
        metrics.write_text(
            '{"elapsed_seconds": 1.0}\n'
            'not-json-at-all\n'
            '{"elapsed_seconds": 2.0}\n'
        )
        pts, _ = load_metrics(metrics)
        # The bad line is dropped, the two valid ones survive.
        assert len(pts) == 2
        assert [p.elapsed_seconds for p in pts] == [1.0, 2.0]

    def test_incremental_reads_pick_up_only_new_lines(self, tmp_path):
        """The viewer tails metrics.jsonl by tracking last_position."""
        metrics = tmp_path / "metrics.jsonl"
        metrics.write_text('{"elapsed_seconds": 1.0}\n')
        first, pos1 = load_metrics(metrics)
        assert len(first) == 1

        # Append another line
        with open(metrics, "a") as f:
            f.write('{"elapsed_seconds": 2.0}\n')

        second, pos2 = load_metrics(metrics, start_position=pos1)
        assert len(second) == 1
        assert second[0].elapsed_seconds == 2.0
        assert pos2 > pos1

    def test_empty_lines_are_skipped(self, tmp_path):
        metrics = tmp_path / "metrics.jsonl"
        metrics.write_text(
            '{"elapsed_seconds": 1.0}\n'
            '\n'
            '   \n'
            '{"elapsed_seconds": 2.0}\n'
        )
        pts, _ = load_metrics(metrics)
        assert len(pts) == 2

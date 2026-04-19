#!/usr/bin/env python3
"""Sweep driver for agent_throughput.py.

Two modes:

 1. Linear sweep (default): --start-qps .. --end-qps step --step
 2. SLO-driven binary search: --auto-search, --slo-ttft-p90-ms, --slo-success

After each point completes, the driver reads the machine-readable
summary.json that agent_throughput.py writes into its run_dir and appends a
row to results.csv / summary.md for easy post-hoc comparison.
"""

import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml


# --------------------------------------------------------------------------
# Single-point driver
# --------------------------------------------------------------------------

def render_temp_config(workload_yaml: Path, dest: Path, qps: float,
                        sustain_duration: int, max_inflight: int, seed: int) -> None:
    with open(workload_yaml) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("workload", {})
    cfg["workload"]["max_qps"] = qps
    cfg["workload"]["initial_qps"] = min(0.2, qps)
    cfg["workload"]["sustain_duration"] = sustain_duration
    cfg["workload"]["max_inflight"] = max_inflight
    cfg["workload"]["random_seed"] = seed
    with open(dest, "w") as f:
        yaml.dump(cfg, f)


def locate_run_summary(data_dir: Path, benchmark_name: str) -> Path | None:
    """agent_throughput.py writes to <data_dir>/<name>/<YYYY-MM-DD-HH-MM-SS>/summary.json."""
    root = data_dir / benchmark_name
    if not root.is_dir():
        return None
    candidates = sorted([p for p in root.iterdir() if p.is_dir()])
    for d in reversed(candidates):
        sj = d / "summary.json"
        if sj.exists():
            return sj
    return None


def run_single_point(args, qps: float, seed: int, results_dir: Path) -> dict:
    """Run one QPS point. Returns a flat dict merging sweep metadata + summary.json."""
    temp_cfg = results_dir / "temp_config.yaml"
    render_temp_config(Path(args.workload_config), temp_cfg,
                       qps=qps,
                       sustain_duration=args.sustain_duration,
                       max_inflight=args.max_inflight,
                       seed=seed)

    benchmark_name = f"{args.name}-qps{qps}"
    log_file = results_dir / f"qps_{qps}_output.log"
    cmd = [
        "python3", str(Path(__file__).parent / "agent_throughput.py"),
        "--workload-config", str(temp_cfg),
        "--server", args.server,
        "--model", args.model,
        "--dashboard-mode",
        "--name", benchmark_name,
        "--data-dir", args.data_dir,
    ]

    print(f"\n{'='*60}\nRunning QPS {qps} (seed: {seed})\n{'='*60}")
    start = time.time()
    status = "success"
    return_code = 0
    killed = False
    try:
        with open(log_file, "w") as lf:
            proc = subprocess.run(cmd, cwd=str(Path(__file__).parent),
                                  stdout=lf, stderr=subprocess.STDOUT,
                                  timeout=args.sustain_duration + 300)
        return_code = proc.returncode
        if return_code != 0:
            status = "failed"
    except subprocess.TimeoutExpired:
        status, return_code, killed = "timeout", -1, True
    except Exception as exc:
        status, return_code = f"error: {exc}", -2
    elapsed = time.time() - start

    # Pull structured summary if the run wrote one
    summary = {}
    sj = locate_run_summary(Path(args.data_dir), benchmark_name)
    if sj and sj.exists():
        try:
            summary = json.loads(sj.read_text())
        except Exception as exc:
            print(f"Warning: could not parse {sj}: {exc}")

    # Prefer a path relative to results_dir for readability; fall back to
    # the absolute path when the file lives outside the sweep tree (the
    # per-run data-dir is usually separate from the sweep results-dir).
    def _nice_path(p: Path) -> str:
        try:
            return str(p.relative_to(results_dir))
        except ValueError:
            return str(p)

    row = {
        "qps":           qps,
        "random_seed":   seed,
        "status":        status,
        "return_code":   return_code,
        "elapsed_s":     elapsed,
        "killed_by_watchdog": killed,
        "log_file":      _nice_path(log_file),
        "summary_json":  _nice_path(sj) if sj else None,
    }

    if summary:
        sustain = _pick_phase(summary.get("phases", []), "sustain")
        row.update({
            "actual_avg_qps":  summary.get("actual_average_qps"),
            "requests_sent":   summary.get("requests_sent"),
            "requests_done":   summary.get("requests_completed"),
            "errors":          summary.get("errors"),
            "success_rate":    summary.get("success_rate"),
            "prompt_mean":     _g(summary, "prompt_length.mean"),
            "prompt_p90":      _g(summary, "prompt_length.p90"),
            "ttft_p50_ms":     _g(summary, "ttft_ms.p50"),
            "ttft_p90_ms":     _g(summary, "ttft_ms.p90"),
            "ttft_p99_ms":     _g(summary, "ttft_ms.p99"),
            "cache_actual":    _g(summary, "cache.actual_hit_rate"),
            "cache_ideal":     _g(summary, "cache.ideal_hit_rate"),
            "sustain_input_tpm":    sustain.get("input_tpm"),
            "sustain_cached_tpm":   sustain.get("cached_tpm"),
            "sustain_uncached_tpm": sustain.get("uncached_tpm"),
            "sustain_gen_tpm":      sustain.get("gen_tpm"),
            "sustain_visible_tpm":  sustain.get("visible_tpm"),
            "sustain_reasoning_tpm":sustain.get("reasoning_tpm"),
            "sustain_qps":          sustain.get("qps"),
            "sustain_cache_hit":    sustain.get("cache_hit_rate"),
            "sustain_ttft_p50_ms":  sustain.get("ttft_p50_ms"),
            "sustain_ttft_p90_ms":  sustain.get("ttft_p90_ms"),
        })

    print(f"QPS {qps} -> {status} in {elapsed:.1f}s  "
          f"(sustain TPM {row.get('sustain_input_tpm', 'n/a')}, "
          f"TTFT p90 {row.get('sustain_ttft_p90_ms', 'n/a')}ms, "
          f"success {row.get('success_rate', 'n/a')})")
    return row


def _g(d: dict, path: str, default=None):
    for part in path.split("."):
        if not isinstance(d, dict) or part not in d:
            return default
        d = d[part]
    return d


def _pick_phase(phases: list, name: str) -> dict:
    for p in phases:
        if p.get("phase") == name:
            return p
    return {}


# --------------------------------------------------------------------------
# Sweep-level aggregation
# --------------------------------------------------------------------------

CSV_COLUMNS = [
    "qps", "status", "actual_avg_qps", "sustain_qps",
    "requests_sent", "requests_done", "errors", "success_rate",
    "sustain_input_tpm", "sustain_cached_tpm", "sustain_uncached_tpm",
    "sustain_visible_tpm", "sustain_reasoning_tpm", "sustain_gen_tpm",
    "sustain_cache_hit", "sustain_ttft_p50_ms", "sustain_ttft_p90_ms",
    "ttft_p50_ms", "ttft_p90_ms", "ttft_p99_ms",
    "prompt_mean", "prompt_p90",
    "cache_actual", "cache_ideal",
    "elapsed_s", "random_seed", "log_file",
]


def write_sweep_artifacts(results_dir: Path, results: list, args) -> None:
    # results.json — raw
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # results.csv — flat table for pandas / Excel
    with open(results_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in CSV_COLUMNS})

    # summary.md — human-readable per-QPS comparison
    lines = [
        f"# Sweep: {args.name}",
        "",
        f"- server: `{args.server}`",
        f"- model:  `{args.model}`",
        f"- workload: `{args.workload_config}`",
        f"- sustain duration: {args.sustain_duration}s",
        f"- max_inflight: {args.max_inflight}",
        "",
        "## Per-QPS sustain metrics",
        "",
        "| target QPS | actual QPS | success | sustain input TPM | cached TPM | uncached TPM | visible gen TPM | reason TPM | cache% | TTFT p50 | TTFT p90 |",
        "|-----------:|-----------:|--------:|------------------:|-----------:|-------------:|----------------:|-----------:|-------:|---------:|---------:|",
    ]
    for r in results:
        def _fmt(x, spec=",.0f"):
            if x is None:
                return "n/a"
            return format(x, spec)
        lines.append(
            "| " + " | ".join([
                _fmt(r.get("qps"), ".2f"),
                _fmt(r.get("sustain_qps"), ".2f"),
                _fmt(r.get("success_rate"), ".1%") if r.get("success_rate") is not None else "n/a",
                _fmt(r.get("sustain_input_tpm")),
                _fmt(r.get("sustain_cached_tpm")),
                _fmt(r.get("sustain_uncached_tpm")),
                _fmt(r.get("sustain_visible_tpm")),
                _fmt(r.get("sustain_reasoning_tpm")),
                _fmt(r.get("sustain_cache_hit"), ".1%") if r.get("sustain_cache_hit") is not None else "n/a",
                _fmt(r.get("sustain_ttft_p50_ms"), ".1f") + "ms" if r.get("sustain_ttft_p50_ms") is not None else "n/a",
                _fmt(r.get("sustain_ttft_p90_ms"), ".1f") + "ms" if r.get("sustain_ttft_p90_ms") is not None else "n/a",
            ]) + " |"
        )
    lines += ["", "Files:",
              "- `results.csv` — flat per-point metrics for pandas / Excel",
              "- `results.json` — same data as nested JSON",
              "- `qps_<q>_output.log` — full console output per point",
              ""]
    (results_dir / "summary.md").write_text("\n".join(lines))

    # Legacy summary.txt (kept for anyone already grepping it)
    with open(results_dir / "summary.txt", "w") as f:
        f.write(f"QPS Sweep Results: {args.name}\n" + "=" * 60 + "\n")
        for r in results:
            f.write(f"QPS {r['qps']}: {r['status']} ({r['elapsed_s']:.1f}s)  "
                    f"sustain_input_tpm={r.get('sustain_input_tpm')}  "
                    f"ttft_p90={r.get('sustain_ttft_p90_ms')}  "
                    f"success={r.get('success_rate')}\n")


# --------------------------------------------------------------------------
# SLO-driven auto capacity search
# --------------------------------------------------------------------------

def slo_satisfied(row: dict, ttft_p90_ms_max: float, success_rate_min: float) -> tuple[bool, str]:
    if row.get("status") != "success":
        return False, f"status={row.get('status')}"
    success = row.get("success_rate")
    if success is None or success < success_rate_min:
        return False, f"success_rate={success}"
    p90 = row.get("sustain_ttft_p90_ms")
    if p90 is None:
        return False, "ttft_p90 missing"
    if p90 > ttft_p90_ms_max:
        return False, f"ttft_p90={p90:.1f}ms > {ttft_p90_ms_max:.1f}"
    return True, f"ok (p90={p90:.1f}ms, success={success:.1%})"


def auto_capacity_search(args, results_dir: Path, base_seed: int) -> list:
    """Binary-search for the highest QPS that still meets the SLOs."""
    print("\nSLO-driven auto capacity search")
    print(f"  TTFT p90 budget: {args.slo_ttft_p90_ms:.0f}ms")
    print(f"  min success rate: {args.slo_success_rate:.2%}")
    print(f"  search range: [{args.auto_min_qps:.2f}, {args.auto_max_qps:.2f}]")
    print(f"  tolerance: {args.auto_tolerance:.2f} QPS\n")

    lo = args.auto_min_qps
    hi = args.auto_max_qps
    results = []
    probe_idx = 0

    def probe(q: float) -> dict:
        nonlocal probe_idx
        seed = base_seed + probe_idx
        probe_idx += 1
        r = run_single_point(args, qps=round(q, 2), seed=seed, results_dir=results_dir)
        ok, why = slo_satisfied(r, args.slo_ttft_p90_ms, args.slo_success_rate)
        r["slo_ok"] = ok
        r["slo_reason"] = why
        results.append(r)
        write_sweep_artifacts(results_dir, results, args)
        print(f"  probe QPS {q:.2f} -> {'PASS' if ok else 'FAIL'}: {why}")
        return r

    # Check boundary: does the LOW end pass?  If not, search is pointless.
    r_lo = probe(lo)
    if not r_lo.get("slo_ok"):
        print(f"\nFloor QPS {lo} already violates SLO — server can't meet target at any rate.")
        return results

    # Check boundary: does the HIGH end pass?  Then the real capacity is
    # at least hi; bail early.
    r_hi = probe(hi)
    if r_hi.get("slo_ok"):
        print(f"\nCeiling QPS {hi} also passes — real capacity is at least {hi}.")
        return results

    # Bisect
    while (hi - lo) > args.auto_tolerance and probe_idx < args.auto_max_probes:
        mid = round((lo + hi) / 2, 2)
        r_mid = probe(mid)
        if r_mid.get("slo_ok"):
            lo = mid
        else:
            hi = mid

    # Sort by QPS for cleaner artifacts
    results.sort(key=lambda r: r["qps"])
    write_sweep_artifacts(results_dir, results, args)
    passing = [r for r in results if r.get("slo_ok")]
    if passing:
        best = max(passing, key=lambda r: r["qps"])
        print(f"\nHighest QPS that satisfied the SLOs: {best['qps']:.2f}")
    return results


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def run_sweep(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.results_dir) / f"{args.name}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    workload_yaml = Path(args.workload_config)
    if workload_yaml.exists():
        with open(workload_yaml) as f:
            original = yaml.safe_load(f)
        with open(results_dir / "original_workload.yaml", "w") as f:
            yaml.dump(original, f)

    # Seed each point differently so prefix cache doesn't get a free lunch
    # between points (and search probes don't collide).
    if args.random_seed is not None:
        base_seed = args.random_seed
    else:
        base_seed = int(datetime.now().strftime("%m%d%H%M"))
    print(f"Using base seed: {base_seed}")

    test_config = {
        "name": args.name,
        "timestamp": timestamp,
        "server": args.server,
        "model": args.model,
        "workload_config": str(args.workload_config),
        "data_dir": args.data_dir,
        "sustain_duration": args.sustain_duration,
        "random_seed_base": base_seed,
        "max_inflight": args.max_inflight,
        "auto_search": args.auto_search,
    }
    if args.auto_search:
        test_config.update({
            "auto_min_qps": args.auto_min_qps,
            "auto_max_qps": args.auto_max_qps,
            "auto_tolerance": args.auto_tolerance,
            "auto_max_probes": args.auto_max_probes,
            "slo_ttft_p90_ms": args.slo_ttft_p90_ms,
            "slo_success_rate": args.slo_success_rate,
        })
    else:
        qps_values = []
        q = args.start_qps
        while q <= args.end_qps + 0.001:
            qps_values.append(round(q, 1))
            q += args.step
        test_config["qps_values"] = qps_values
        test_config["seed_per_qps"] = {str(v): base_seed + i for i, v in enumerate(qps_values)}

    with open(results_dir / "test_config.json", "w") as f:
        json.dump(test_config, f, indent=2)

    if args.auto_search:
        results = auto_capacity_search(args, results_dir, base_seed)
    else:
        results = []
        for idx, qps in enumerate(qps_values):
            row = run_single_point(args, qps=qps, seed=base_seed + idx, results_dir=results_dir)
            results.append(row)
            write_sweep_artifacts(results_dir, results, args)

    print(f"\n{'='*60}\nSweep done. Artifacts in: {results_dir}\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="QPS sweep / SLO capacity search for agent_throughput.py")
    parser.add_argument("--name", required=True, help="Run name prefix")
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--model", default="dsv3")
    parser.add_argument("--workload-config", default="workloads/code_agent_16k.yaml")
    parser.add_argument("--data-dir", default="benchmarks")
    parser.add_argument("--results-dir", default="qps_sweep_results")
    parser.add_argument("--sustain-duration", type=int, default=600)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--max-inflight", type=int, default=24)

    # Linear sweep parameters
    parser.add_argument("--start-qps", type=float, default=0.4)
    parser.add_argument("--end-qps", type=float, default=2.0)
    parser.add_argument("--step", type=float, default=0.2)

    # SLO-driven auto search
    parser.add_argument("--auto-search", action="store_true",
                        help="Instead of a linear sweep, binary-search for "
                             "the max QPS that satisfies --slo-* bounds.")
    parser.add_argument("--slo-ttft-p90-ms", type=float, default=500.0,
                        help="Sustain-phase TTFT p90 ceiling (ms) for --auto-search")
    parser.add_argument("--slo-success-rate", type=float, default=0.99,
                        help="Success-rate floor for --auto-search (fraction)")
    parser.add_argument("--auto-min-qps", type=float, default=0.1)
    parser.add_argument("--auto-max-qps", type=float, default=10.0)
    parser.add_argument("--auto-tolerance", type=float, default=0.2,
                        help="Stop when the search range narrows below this.")
    parser.add_argument("--auto-max-probes", type=int, default=10,
                        help="Hard cap on number of benchmark runs during search.")

    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()

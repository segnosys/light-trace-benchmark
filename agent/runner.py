#!/usr/bin/env python3
"""QPS sweep script for benchmarking"""

import argparse
import subprocess
import os
import json
import time
import yaml
from datetime import datetime
from pathlib import Path

def run_sweep(args):
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.results_dir) / f"{args.name}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy original workload config
    workload_config = Path(args.workload_config)
    if workload_config.exists():
        with open(workload_config) as f:
            original_config = yaml.safe_load(f)
        with open(results_dir / "original_workload.yaml", "w") as f:
            yaml.dump(original_config, f)
    
    # Generate QPS values
    qps_values = []
    qps = args.start_qps
    while qps <= args.end_qps + 0.001:
        qps_values.append(round(qps, 1))
        qps += args.step
    
    # Generate base seed: use provided seed, or auto-generate from timestamp
    # Auto-generated seed uses timestamp to ensure different seeds across test runs
    # even without server restart (avoids cache hit between different test sessions)
    if args.random_seed is not None:
        base_seed = args.random_seed
    else:
        # Use timestamp-based seed: MMDDHHMM as integer (e.g., 02051234 for Feb 5 12:34)
        base_seed = int(datetime.now().strftime("%m%d%H%M"))
    
    print(f"Using base seed: {base_seed}")
    
    # Save test config
    # Generate seed mapping for documentation
    seed_per_qps = {}
    for idx, qps in enumerate(qps_values):
        seed_per_qps[str(qps)] = base_seed + idx
    
    test_config = {
        "name": args.name,
        "timestamp": timestamp,
        "qps_values": qps_values,
        "server": args.server,
        "model": args.model,
        "workload_config": str(args.workload_config),
        "data_dir": args.data_dir,
        "sustain_duration": args.sustain_duration,
        "random_seed_base": base_seed,
        "seed_per_qps": seed_per_qps,
        "max_inflight": args.max_inflight
    }
    with open(results_dir / "test_config.json", "w") as f:
        json.dump(test_config, f, indent=2)
    
    results = []
    
    for idx, qps in enumerate(qps_values):
        # Generate unique seed for each QPS to avoid cache effects between runs
        # seed = base_seed + qps_index (e.g., base=2051234, qps_idx=0,1,2... -> seeds 2051234,2051235,2051236...)
        current_seed = base_seed + idx
        
        print(f"\n{'='*60}")
        print(f"Running QPS {qps} (seed: {current_seed})")
        print(f"{'='*60}\n")
        
        # Create temp config with modified QPS
        temp_config_path = results_dir / "temp_config.yaml"
        with open(workload_config) as f:
            config = yaml.safe_load(f)
        
        config['workload']['max_qps'] = qps
        config['workload']['initial_qps'] = min(0.2, qps)
        config['workload']['sustain_duration'] = args.sustain_duration
        config['workload']['max_inflight'] = args.max_inflight
        config['workload']['random_seed'] = current_seed
        
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Invoke the simulator
        benchmark_name = f"{args.name}-qps{qps}"
        log_file = results_dir / f"qps_{qps}_output.log"

        cmd = [
            "python3", str(Path(__file__).parent / "agent_throughput.py"),
            "--workload-config", str(temp_config_path),
            "--server", args.server,
            "--model", args.model,
            "--dashboard-mode",
            "--name", benchmark_name,
            "--data-dir", args.data_dir
        ]
        
        start_time = time.time()
        try:
            with open(log_file, 'w') as lf:
                process = subprocess.run(
                    cmd,
                    cwd=str(Path(__file__).parent),
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    timeout=args.sustain_duration + 300  # Extra time for ramp and cleanup
                )
            elapsed = time.time() - start_time
            result = {
                "qps": qps,
                "random_seed": current_seed,
                "status": "success" if process.returncode == 0 else "failed",
                "return_code": process.returncode,
                "elapsed_seconds": elapsed,
                "log_file": str(log_file),
                "killed_by_watchdog": False
            }
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            result = {
                "qps": qps,
                "random_seed": current_seed,
                "status": "timeout",
                "return_code": -1,
                "elapsed_seconds": elapsed,
                "log_file": str(log_file),
                "killed_by_watchdog": True
            }
        except Exception as e:
            elapsed = time.time() - start_time
            result = {
                "qps": qps,
                "random_seed": current_seed,
                "status": "error",
                "error": str(e),
                "elapsed_seconds": elapsed,
                "log_file": str(log_file),
                "killed_by_watchdog": False
            }
        
        results.append(result)
        
        # Save results incrementally
        with open(results_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"QPS {qps} completed: {result['status']} in {elapsed:.1f}s")
    
    # Generate summary
    summary_lines = [f"QPS Sweep Results: {args.name}", "=" * 60]
    for r in results:
        summary_lines.append(f"QPS {r['qps']}: {r['status']} ({r['elapsed_seconds']:.1f}s)")
    
    with open(results_dir / "summary.txt", 'w') as f:
        f.write("\n".join(summary_lines))
    
    print(f"\n{'='*60}")
    print(f"Sweep completed. Results in: {results_dir}")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description="Run QPS sweep benchmark")
    parser.add_argument("--name", required=True, help="Test name")
    parser.add_argument("--start-qps", type=float, default=0.4)
    parser.add_argument("--end-qps", type=float, default=2.0)
    parser.add_argument("--step", type=float, default=0.2)
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--model", default="dsv3")
    parser.add_argument("--workload-config", default="workloads/code_agent_16k.yaml")
    parser.add_argument("--data-dir", default="benchmarks")
    parser.add_argument("--results-dir", default="qps_sweep_results")
    parser.add_argument("--sustain-duration", type=int, default=600)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--max-inflight", type=int, default=24)
    
    args = parser.parse_args()
    run_sweep(args)

if __name__ == "__main__":
    main()

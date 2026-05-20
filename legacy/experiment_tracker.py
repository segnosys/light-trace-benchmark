import os
import sys
from dataclasses import fields
from typing import Any, Dict, List, Optional

try:
    import wandb

    TRACKER_AVAILABLE = True
except ImportError:
    TRACKER_AVAILABLE = False
    wandb = None

from legacy.schema import BenchmarkReport


class ExperimentTracker:
    """
    Handles Weights & Biases logging for inference benchmark experiments.
    """

    def __init__(
        self,
        enabled: bool = False,
        project: str = "agent-bench",
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ):
        self.enabled = enabled and TRACKER_AVAILABLE
        self.project = project
        self.entity = entity
        self.tags = tags or []
        self.notes = notes
        self.run = None

        if self.enabled and not TRACKER_AVAILABLE:
            print("Warning: wandb is not installed. Install with 'pip install wandb'")
            self.enabled = False

    def start_run(self, config: Dict[str, Any], command: str, output_path: str) -> None:
        if not self.enabled:
            return

        run_name = self._derive_run_name(output_path)

        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=run_name,
            tags=self.tags,
            notes=self.notes,
            config=config,
            reinit=True,
        )

        wandb.config.update({"command": command})

        artifact = wandb.Artifact("command", type="command")
        with artifact.new_file("command.txt") as f:
            f.write(command)
        wandb.log_artifact(artifact)

    def record_metrics(self, report: BenchmarkReport, step: int) -> None:
        if not self.enabled or not self.run:
            return

        flattened = self._flatten_dataclass(report)

        wandb.log(flattened, step=step)

        if hasattr(wandb.run, "summary"):
            level_key = f"level_{report.load.level}"
            key_metrics = {
                f"{level_key}_job_level_tps": report.overview.job_level_tps,
                f"{level_key}_actual_qps": report.overview.actual_qps,
                f"{level_key}_ttft_mean": report.ttft.mean,
                f"{level_key}_ttft_p99": report.ttft.p99,
                f"{level_key}_user_tps_mean": report.user_tps.mean,
                f"{level_key}_total_requests": report.overview.total_num_requests,
                f"{level_key}_failed_requests": report.overview.num_failed_requests,
                f"{level_key}_success_rate": (
                    (report.overview.total_num_requests - report.overview.num_failed_requests)
                    / report.overview.total_num_requests
                    if report.overview.total_num_requests > 0
                    else 0
                ),
            }

            for key, value in key_metrics.items():
                wandb.run.summary[key] = value

    def end_run(self) -> None:
        if not self.enabled or not self.run:
            return

        wandb.finish()
        self.run = None

    def upload_csv_artifact(self, csv_path: str) -> None:
        if not self.enabled or not self.run or not csv_path:
            return

        if not os.path.exists(csv_path):
            print(f"Warning: CSV file not found at {csv_path}, skipping artifact upload")
            return

        artifact = wandb.Artifact("final_results", type="results")

        artifact.add_file(csv_path)

        wandb.log_artifact(artifact)
        print(f"Logged final results CSV artifact from {csv_path}")

    def _derive_run_name(self, output_path: str) -> str:
        filename = os.path.basename(output_path)

        if "." in filename:
            filename = filename.rsplit(".", 1)[0]

        sanitized = filename.replace(" ", "_").replace("/", "_").replace("\\", "_")

        if not sanitized:
            sanitized = "agent_bench_run"

        return sanitized

    def _flatten_dataclass(self, obj) -> Dict[str, Any]:
        flat_dict = {}
        for field in fields(type(obj)):
            value = getattr(obj, field.name)

            if hasattr(value, "__dataclass_fields__"):
                nested_dict = self._flatten_dataclass(value)
                flat_dict.update({f"{field.name}_{k}": v for k, v in nested_dict.items()})
            else:
                flat_dict[field.name] = value

        return flat_dict


def capture_invocation() -> str:
    return " ".join(sys.argv)


def split_tag_string(tags_str: Optional[str]) -> Optional[List[str]]:
    if not tags_str:
        return None

    return [tag.strip() for tag in tags_str.split(",") if tag.strip()]

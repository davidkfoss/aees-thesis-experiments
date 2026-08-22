#!/usr/bin/env python3
"""Aggregate AEES versus reward-free CIFAR-100 attribution controls."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
from statistics import mean, stdev
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.utils.cifar_attribution_controls import (  # noqa: E402
    LR_MULTIPLIERS,
    SOURCE_SEEDS,
    load_validated_aees_trace,
)
from scripts.tables._paired_stats import compare_paired  # noqa: E402


NOISE_REGIMES = ("asym20", "sym20", "sym40")
OPTIMIZERS = ("AdamW", "SGD")
METHODS = ("aees", "uniform-random", "frequency-matched")
METRICS = ("peak_validation_accuracy", "final_validation_accuracy", "best_to_final_drop")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aees-trace-root", type=Path, default=Path("results/cifar_noisy"))
    parser.add_argument(
        "--controls-root",
        type=Path,
        default=Path("results/cifar_attribution_controls"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reproduced_artifacts/tables/cifar_attribution_controls"),
    )
    return parser.parse_args(argv)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def control_path(root: Path, regime: str, optimizer: str, policy: str, seed: int) -> Path:
    return (
        root
        / f"cifar100_{regime}_seed{seed}"
        / f"{optimizer.lower()}_{policy.replace('-', '_')}.json"
    )


def load_result(
    path: Path,
    *,
    expected_policy: str | None = None,
    expected_optimizer: str | None = None,
    expected_seed: int | None = None,
) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Result {path} is not a JSON object.")
    required = ("best_val_accuracy", "final_val_accuracy", "episode_logs", "config")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Result {path} is incomplete; missing {missing}.")
    if expected_policy is not None:
        config = payload.get("config")
        if not isinstance(config, dict) or config.get("control_mode") != expected_policy:
            raise ValueError(f"Result {path} is not a {expected_policy} run.")
        if expected_optimizer is not None and config.get("optimizer") != expected_optimizer:
            raise ValueError(f"Result {path} has the wrong optimizer.")
        if expected_seed is not None and (
            config.get("seed") != expected_seed or payload.get("seed") != expected_seed
        ):
            raise ValueError(f"Result {path} has the wrong paired seed.")
        expected_config = {
            "epochs": 200,
            "batch_size": 128,
            "episode_length": 200,
            "lr_scheduler": "none",
            "lr_candidates": list(LR_MULTIPLIERS),
            "noise_candidates": [0.0],
        }
        mismatches = [
            key for key, expected in expected_config.items() if config.get(key) != expected
        ]
        if mismatches:
            raise ValueError(f"Result {path} has incompatible config fields {mismatches}.")
        metadata = config.get("open_loop_policy")
        logs = payload.get("episode_logs")
        if not isinstance(metadata, dict) or not isinstance(logs, dict):
            raise ValueError(f"Result {path} lacks open-loop schedule provenance.")
        planned = metadata.get("planned_lr_multipliers")
        executed = metadata.get("executed_lr_multipliers")
        logged = logs.get("selected_lr_values")
        if planned != executed or executed != logged:
            raise ValueError(f"Result {path} has inconsistent executed action traces.")
        if not isinstance(logged, list) or len(logged) != 391:
            raise ValueError(f"Result {path} has an incomplete episode action trace.")
        if (
            payload.get("total_epochs") != 200
            or payload.get("total_steps") != 78_200
            or len(payload.get("val_accuracies", [])) != 200
        ):
            raise ValueError(f"Result {path} has an incomplete training horizon.")
    return payload


def metrics_from_result(payload: dict[str, object]) -> dict[str, float]:
    peak = float(payload["best_val_accuracy"])
    final = float(payload["final_val_accuracy"])
    return {
        "peak_validation_accuracy": peak,
        "final_validation_accuracy": final,
        "best_to_final_drop": peak - final,
    }


def actions_from_result(payload: dict[str, object]) -> list[float]:
    logs = payload.get("episode_logs")
    if not isinstance(logs, dict) or not isinstance(logs.get("selected_lr_values"), list):
        raise ValueError("Result is missing episode_logs.selected_lr_values.")
    actions = [float(value) for value in logs["selected_lr_values"]]
    if any(value not in LR_MULTIPLIERS for value in actions):
        raise ValueError(f"Result contains an LR action outside {list(LR_MULTIPLIERS)}.")
    return actions


def action_distribution(actions: Sequence[float]) -> dict[str, object]:
    counts = Counter(actions)
    total = len(actions)
    return {
        "total_episodes": total,
        "counts": {str(value): counts.get(value, 0) for value in LR_MULTIPLIERS},
        "probabilities": {
            str(value): (counts.get(value, 0) / total if total else None)
            for value in LR_MULTIPLIERS
        },
    }


def descriptive(values: Sequence[float]) -> dict[str, object]:
    return {
        "n": len(values),
        "mean": mean(values) if values else None,
        "sd": stdev(values) if len(values) > 1 else (0.0 if values else None),
    }


def build_comparison(
    aees_by_seed: dict[int, dict[str, float]],
    control_by_seed: dict[int, dict[str, float]],
) -> dict[str, object]:
    shared = sorted(set(aees_by_seed) & set(control_by_seed))
    complete = shared == list(SOURCE_SEEDS)
    output: dict[str, object] = {
        "complete": complete,
        "available_pair_count": len(shared),
        "paired_seeds": shared,
        "metrics": {},
    }
    metric_output = output["metrics"]
    assert isinstance(metric_output, dict)
    for metric in METRICS:
        effects = {
            str(seed): aees_by_seed[seed][metric] - control_by_seed[seed][metric]
            for seed in shared
        }
        higher_is_better = metric != "best_to_final_drop"
        wins = sum(
            effect > 0.0 if higher_is_better else effect < 0.0
            for effect in effects.values()
        )
        summary = None
        if complete:
            comparison = compare_paired(
                [aees_by_seed[seed][metric] for seed in SOURCE_SEEDS],
                [control_by_seed[seed][metric] for seed in SOURCE_SEEDS],
                scale=100.0,
            )
            summary = asdict(comparison)
        metric_output[metric] = {
            "aees_minus_control_by_seed": effects,
            "summary_percentage_points": summary,
            "aees_win_count": wins if complete else None,
            "win_denominator": 5 if complete else None,
        }
    return output


def aggregate(
    *,
    aees_root: Path,
    controls_root: Path,
) -> dict[str, object]:
    cells: list[dict[str, object]] = []
    for regime in NOISE_REGIMES:
        for optimizer in OPTIMIZERS:
            results_by_method: dict[str, dict[int, dict[str, object]]] = {
                method: {} for method in METHODS
            }
            missing_run_ids: list[str] = []
            invalid_runs: list[dict[str, str]] = []
            for seed in SOURCE_SEEDS:
                aees_id = f"cifar100-{regime}_{optimizer.lower()}_aees_seed{seed}"
                try:
                    trace = load_validated_aees_trace(
                        trace_root=aees_root,
                        noise_regime=regime,
                        optimizer=optimizer,
                        seed=seed,
                    )
                    payload = load_result(trace.path)
                    results_by_method["aees"][seed] = {
                        "path": str(trace.path.resolve()),
                        "metrics": metrics_from_result(payload),
                        "actions": actions_from_result(payload),
                    }
                except ValueError as exc:
                    missing_run_ids.append(aees_id)
                    invalid_runs.append({"run_id": aees_id, "error": str(exc)})
                for policy in METHODS[1:]:
                    run_id = "_".join(
                        (f"cifar100-{regime}", optimizer.lower(), policy, f"seed{seed}")
                    )
                    path = control_path(controls_root, regime, optimizer, policy, seed)
                    if not path.is_file():
                        missing_run_ids.append(run_id)
                        continue
                    try:
                        payload = load_result(
                            path,
                            expected_policy=policy,
                            expected_optimizer=optimizer,
                            expected_seed=seed,
                        )
                        results_by_method[policy][seed] = {
                            "path": str(path.resolve()),
                            "metrics": metrics_from_result(payload),
                            "actions": actions_from_result(payload),
                        }
                    except ValueError as exc:
                        missing_run_ids.append(run_id)
                        invalid_runs.append({"run_id": run_id, "error": str(exc)})

            method_summaries: dict[str, object] = {}
            metric_maps: dict[str, dict[int, dict[str, float]]] = {}
            for method, seed_results in results_by_method.items():
                metric_maps[method] = {
                    seed: result["metrics"]  # type: ignore[dict-item]
                    for seed, result in seed_results.items()
                }
                per_seed_actions = {
                    str(seed): action_distribution(result["actions"])  # type: ignore[arg-type]
                    for seed, result in sorted(seed_results.items())
                }
                all_actions = [
                    action
                    for result in seed_results.values()
                    for action in result["actions"]  # type: ignore[union-attr]
                ]
                method_summaries[method] = {
                    "complete": sorted(seed_results) == list(SOURCE_SEEDS),
                    "seeds": sorted(seed_results),
                    "paths": {
                        str(seed): result["path"] for seed, result in sorted(seed_results.items())
                    },
                    "metrics": {
                        metric: descriptive(
                            [
                                metric_maps[method][seed][metric]
                                for seed in sorted(metric_maps[method])
                            ]
                        )
                        for metric in METRICS
                    },
                    "action_distribution": action_distribution(all_actions),
                    "action_distribution_by_seed": per_seed_actions,
                }
            cells.append(
                {
                    "noise_regime": regime,
                    "optimizer": optimizer,
                    "complete": not missing_run_ids,
                    "missing_run_ids": sorted(set(missing_run_ids)),
                    "invalid_runs": invalid_runs,
                    "methods": method_summaries,
                    "comparisons": {
                        "aees_minus_uniform_random": build_comparison(
                            metric_maps["aees"], metric_maps["uniform-random"]
                        ),
                        "aees_minus_frequency_matched": build_comparison(
                            metric_maps["aees"], metric_maps["frequency-matched"]
                        ),
                    },
                }
            )
    return {
        "schema_version": 1,
        "bootstrap": {
            "type": "paired percentile bootstrap of seed-level differences",
            "resamples": 10_000,
            "seed": 0,
            "interval_percentiles": [2.5, 97.5],
        },
        "complete": all(cell["complete"] for cell in cells),
        "cells": cells,
    }


def write_csv(summary: dict[str, object], path: Path) -> None:
    fields = [
        "noise_regime",
        "optimizer",
        "method",
        "complete",
        "n",
        "peak_mean_pct",
        "peak_sd_pct",
        "final_mean_pct",
        "final_sd_pct",
        "drop_mean_pp",
        "drop_sd_pp",
        "action_p_0.5",
        "action_p_1.0",
        "action_p_2.0",
        "aees_minus_method_peak_pp",
        "aees_minus_method_peak_ci_low_pp",
        "aees_minus_method_peak_ci_high_pp",
        "aees_peak_wins_out_of_5",
        "aees_minus_method_final_pp",
        "aees_minus_method_final_ci_low_pp",
        "aees_minus_method_final_ci_high_pp",
        "aees_final_wins_out_of_5",
        "missing_run_ids",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for cell in summary["cells"]:  # type: ignore[index]
            methods = cell["methods"]
            comparisons = cell["comparisons"]
            for method in METHODS:
                method_summary = methods[method]
                metrics = method_summary["metrics"]
                distribution = method_summary["action_distribution"]["probabilities"]
                row: dict[str, object] = {
                    "noise_regime": cell["noise_regime"],
                    "optimizer": cell["optimizer"],
                    "method": method,
                    "complete": method_summary["complete"],
                    "n": metrics["peak_validation_accuracy"]["n"],
                    "peak_mean_pct": scaled(metrics["peak_validation_accuracy"]["mean"]),
                    "peak_sd_pct": scaled(metrics["peak_validation_accuracy"]["sd"]),
                    "final_mean_pct": scaled(metrics["final_validation_accuracy"]["mean"]),
                    "final_sd_pct": scaled(metrics["final_validation_accuracy"]["sd"]),
                    "drop_mean_pp": scaled(metrics["best_to_final_drop"]["mean"]),
                    "drop_sd_pp": scaled(metrics["best_to_final_drop"]["sd"]),
                    "action_p_0.5": distribution["0.5"],
                    "action_p_1.0": distribution["1.0"],
                    "action_p_2.0": distribution["2.0"],
                    "missing_run_ids": ";".join(cell["missing_run_ids"]),
                }
                if method != "aees":
                    comparison_key = f"aees_minus_{method.replace('-', '_')}"
                    comparison_metrics = comparisons[comparison_key]["metrics"]
                    for short, metric in (
                        ("peak", "peak_validation_accuracy"),
                        ("final", "final_validation_accuracy"),
                    ):
                        metric_comparison = comparison_metrics[metric]
                        comparison_summary = metric_comparison["summary_percentage_points"]
                        if comparison_summary is not None:
                            row[f"aees_minus_method_{short}_pp"] = comparison_summary["diff_mean"]
                            row[f"aees_minus_method_{short}_ci_low_pp"] = comparison_summary[
                                "diff_ci_low"
                            ]
                            row[f"aees_minus_method_{short}_ci_high_pp"] = comparison_summary[
                                "diff_ci_high"
                            ]
                        row[f"aees_{short}_wins_out_of_5"] = metric_comparison[
                            "aees_win_count"
                        ]
                writer.writerow(row)


def scaled(value: object) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return None if math.isnan(converted) else 100.0 * converted


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    aees_root = resolve_path(args.aees_trace_root)
    controls_root = resolve_path(args.controls_root)
    out_dir = resolve_path(args.out_dir)
    summary = aggregate(aees_root=aees_root, controls_root=controls_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "cifar_attribution_controls_summary.json"
    csv_path = out_dir / "cifar_attribution_controls_summary.csv"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(summary, csv_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    incomplete = [
        cell for cell in summary["cells"] if not cell["complete"]  # type: ignore[index]
    ]
    if incomplete:
        print("Incomplete cells (no values were fabricated):")
        for cell in incomplete:
            print(
                f"  {cell['noise_regime']}/{cell['optimizer']}: "
                + ", ".join(cell["missing_run_ids"])
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

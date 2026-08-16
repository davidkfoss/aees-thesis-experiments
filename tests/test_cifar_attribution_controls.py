from __future__ import annotations

import json
from pathlib import Path
import random

import pytest

from experiments.utils.cifar_attribution_controls import (
    AEES_ARTIFACT_NAMES,
    LR_MULTIPLIERS,
    NOISE_REGIMES,
    OPTIMIZER_BASE_LRS,
    OpenLoopPlan,
    PrecomputedScheduleController,
    TraceValidationError,
    build_frequency_matched_plan,
    build_uniform_random_plan,
    largest_remainder_counts,
    load_validated_aees_trace,
)
from scripts import run_cifar_attribution_controls as launcher


def make_aees_payload(
    *,
    regime: str,
    optimizer: str,
    seed: int,
    actions: list[float] | None = None,
) -> dict[str, object]:
    actions = actions or [LR_MULTIPLIERS[index % 3] for index in range(391)]
    noise_type, noise_rate = NOISE_REGIMES[regime]
    return {
        "method_name": "AdaptiveScheduler",
        "seed": seed,
        "total_epochs": 200,
        "total_steps": 78_200,
        "best_val_accuracy": 0.5,
        "final_val_accuracy": 0.4,
        "val_accuracies": [0.4] * 200,
        "config": {
            "control_mode": "adaptive",
            "optimizer": optimizer,
            "label_noise_type": noise_type,
            "label_noise_rate": noise_rate,
            "episode_length": 200,
            "epochs": 200,
            "batch_size": 128,
            "lr_scheduler": "none",
            "lr": OPTIMIZER_BASE_LRS[optimizer],
            "weight_decay": 1e-4,
            "momentum": 0.9,
            "lr_candidates": list(LR_MULTIPLIERS),
            "noise_candidates": [0.0],
            "seed": seed,
        },
        "episode_logs": {
            "selected_lr_values": actions,
            "selected_lr_indices": [LR_MULTIPLIERS.index(value) for value in actions],
            "episode_start_steps": [index * 200 for index in range(391)],
            "episode_end_steps": [index * 200 + 199 for index in range(391)],
            "episode_rewards": [0.0] * 391,
        },
    }


def write_trace(
    root: Path,
    *,
    regime: str,
    optimizer: str,
    seed: int,
    actions: list[float] | None = None,
) -> Path:
    path = (
        root
        / f"cifar100_{regime}_seed{seed}"
        / AEES_ARTIFACT_NAMES[optimizer]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            make_aees_payload(
                regime=regime,
                optimizer=optimizer,
                seed=seed,
                actions=actions,
            )
        ),
        encoding="utf-8",
    )
    return path


def fake_plan(*, episode_count: int, schedule_seed: int, **_: object) -> OpenLoopPlan:
    return build_uniform_random_plan(
        episode_count=episode_count,
        schedule_seed=schedule_seed,
    )


def test_full_matrix_contains_exactly_60_unique_runs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launcher, "build_frequency_matched_plan", fake_plan)
    args = launcher.parse_args(
        [
            "--output-root",
            str(tmp_path / "out"),
            "--aees-trace-root",
            str(tmp_path / "traces"),
        ]
    )
    runs = launcher.build_run_matrix(args)
    assert len(runs) == 60
    assert len({run.run_id for run in runs}) == 60
    assert {run.seed for run in runs} == {0, 1, 2, 3, 4}
    assert all(len([item for item in runs if item.seed == seed]) == 12 for seed in range(5))


def test_leave_one_out_excludes_target_and_probabilities_sum_to_one(tmp_path: Path) -> None:
    for seed in range(5):
        actions = [0.5] * 391 if seed == 0 else [1.0] * 391
        write_trace(tmp_path, regime="sym20", optimizer="AdamW", seed=seed, actions=actions)
    plan = build_frequency_matched_plan(
        trace_root=tmp_path,
        noise_regime="sym20",
        optimizer="AdamW",
        target_seed=0,
        episode_count=391,
        schedule_seed=9,
    )
    assert plan.source_seeds == (1, 2, 3, 4)
    assert plan.probabilities == (0.0, 1.0, 0.0)
    assert all(probability >= 0.0 for probability in plan.probabilities)
    assert sum(plan.probabilities) == pytest.approx(1.0)


def test_generated_schedules_use_only_candidates_and_counts_sum() -> None:
    probabilities = (0.201, 0.398, 0.401)
    counts = largest_remainder_counts(probabilities, 391)
    assert sum(counts) == 391
    uniform = build_uniform_random_plan(episode_count=391, schedule_seed=42)
    assert set(uniform.lr_values) <= set(LR_MULTIPLIERS)
    assert sum(uniform.integer_counts) == 391


def test_schedule_generation_is_deterministic_and_does_not_advance_global_rng() -> None:
    random.seed(123)
    state_before = random.getstate()
    first = build_uniform_random_plan(episode_count=391, schedule_seed=88)
    state_after = random.getstate()
    second = build_uniform_random_plan(episode_count=391, schedule_seed=88)
    assert first == second
    assert state_before == state_after


def test_schedule_seed_is_separate_from_paired_training_seed(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launcher, "build_frequency_matched_plan", fake_plan)
    common = [
        "--policies",
        "uniform-random",
        "--seeds",
        "3",
        "--gpus",
        "0",
        "--noise-regimes",
        "sym20",
        "--optimizers",
        "AdamW",
        "--output-root",
        str(tmp_path),
    ]
    first = launcher.build_run_matrix(launcher.parse_args([*common, "--schedule-seed-base", "1"]))[0]
    second = launcher.build_run_matrix(launcher.parse_args([*common, "--schedule-seed-base", "2"]))[0]
    first_seed = first.command[first.command.index("--seed") + 1]
    second_seed = second.command[second.command.index("--seed") + 1]
    assert first_seed == second_seed == "3"
    assert first.schedule_seed != second.schedule_seed


def test_precomputed_controller_never_uses_training_feedback() -> None:
    controller = PrecomputedScheduleController([2, 0, 1], n_arms=3)
    controller.update(0, float("nan"))
    controller.update(1, float("inf"))
    assert [controller.select_mode() for _ in range(3)] == [2, 0, 1]
    state = controller.get_state()
    assert state["feedback_ignored"] is True
    assert state["bandit_statistics_updated"] is False


def test_mismatched_source_trace_is_rejected(tmp_path: Path) -> None:
    path = write_trace(tmp_path, regime="sym40", optimizer="SGD", seed=2)
    payload = json.loads(path.read_text())
    payload["config"]["episode_length"] = 100
    path.write_text(json.dumps(payload))
    with pytest.raises(TraceValidationError, match="episode_length"):
        load_validated_aees_trace(
            trace_root=tmp_path,
            noise_regime="sym40",
            optimizer="SGD",
            seed=2,
        )


def test_missing_truncated_and_duplicate_sources_are_actionable(tmp_path: Path) -> None:
    with pytest.raises(TraceValidationError, match="Missing AEES source trace"):
        load_validated_aees_trace(
            trace_root=tmp_path,
            noise_regime="asym20",
            optimizer="AdamW",
            seed=1,
        )
    path = write_trace(tmp_path, regime="asym20", optimizer="AdamW", seed=1)
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(TraceValidationError, match="Could not parse"):
        load_validated_aees_trace(
            trace_root=tmp_path,
            noise_regime="asym20",
            optimizer="AdamW",
            seed=1,
        )
    path.write_text(
        json.dumps(make_aees_payload(regime="asym20", optimizer="AdamW", seed=1)),
        encoding="utf-8",
    )
    truncated = json.loads(path.read_text())
    truncated["episode_logs"]["selected_lr_values"].pop()
    path.write_text(json.dumps(truncated), encoding="utf-8")
    with pytest.raises(TraceValidationError, match="missing or truncated"):
        load_validated_aees_trace(
            trace_root=tmp_path,
            noise_regime="asym20",
            optimizer="AdamW",
            seed=1,
        )
    path.write_text(
        json.dumps(make_aees_payload(regime="asym20", optimizer="AdamW", seed=1)),
        encoding="utf-8",
    )
    path.with_suffix(".json").write_text(path.read_text(), encoding="utf-8")
    with pytest.raises(TraceValidationError, match="Duplicate AEES source traces"):
        load_validated_aees_trace(
            trace_root=tmp_path,
            noise_regime="asym20",
            optimizer="AdamW",
            seed=1,
        )


def test_completed_result_is_safely_skipped_on_resume(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launcher, "build_frequency_matched_plan", fake_plan)
    args = launcher.parse_args(
        [
            "--policies",
            "uniform-random",
            "--seeds",
            "0",
            "--gpus",
            "0",
            "--noise-regimes",
            "sym20",
            "--optimizers",
            "AdamW",
            "--output-root",
            str(tmp_path),
        ]
    )
    run = launcher.build_run_matrix(args)[0]
    result_path = Path(run.output_path)
    result_path.parent.mkdir(parents=True)
    sequence = run.plan_metadata["planned_lr_multipliers"]
    result_path.write_text(
        json.dumps(
            {
                "total_epochs": 200,
                "total_steps": 78_200,
                "val_accuracies": [0.5] * 200,
                "config": {
                    "control_mode": run.policy,
                    "optimizer": run.optimizer,
                    "seed": run.seed,
                    "schedule_seed": run.schedule_seed,
                    "open_loop_policy": {
                        "planned_lr_multipliers": sequence,
                        "executed_lr_multipliers": sequence,
                    },
                },
                "episode_logs": {"selected_lr_values": sequence},
            }
        ),
        encoding="utf-8",
    )
    state = launcher.LaunchState(
        runs=[run],
        manifest_path=tmp_path / "manifest.json",
        dry_run=False,
        resume=True,
        overwrite=False,
    )
    launcher.prepare_statuses(state)
    assert run.status == "skipped"


def test_dry_run_starts_no_training_and_creates_no_result_or_log(monkeypatch, tmp_path: Path) -> None:
    def fail_popen(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry-run attempted to start training")

    monkeypatch.setattr(launcher.subprocess, "Popen", fail_popen)
    output_root = tmp_path / "out"
    exit_code = launcher.main(
        [
            "--policies",
            "uniform-random",
            "--seeds",
            "0",
            "--gpus",
            "0",
            "--noise-regimes",
            "sym20",
            "--optimizers",
            "AdamW",
            "--output-root",
            str(output_root),
            "--dry-run",
        ]
    )
    assert exit_code == 0
    assert (output_root / "manifest.json").is_file()
    assert not (output_root / "_logs").exists()
    assert not list(output_root.glob("cifar100_*"))

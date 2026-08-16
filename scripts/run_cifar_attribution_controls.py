#!/usr/bin/env python3
"""Launch reward-free CIFAR-100 attribution controls across seed-paired GPUs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import threading
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.utils.cifar_attribution_controls import (  # noqa: E402
    LR_MULTIPLIERS,
    NOISE_REGIMES,
    OPTIMIZER_BASE_LRS,
    SOURCE_SEEDS,
    build_frequency_matched_plan,
    build_uniform_random_plan,
    episode_count_from_training_config,
)


DEFAULT_POLICIES = ("uniform-random", "frequency-matched")
DEFAULT_OPTIMIZERS = ("AdamW", "SGD")
DEFAULT_NOISE_REGIMES = ("asym20", "sym20", "sym40")
EXPECTED_FULL_RUN_COUNT = 60


@dataclass
class LaunchRun:
    """One fully resolved launcher entry."""

    run_id: str
    policy: str
    noise_regime: str
    optimizer: str
    seed: int
    gpu_id: str
    schedule_seed: int
    epochs: int
    total_steps: int
    output_path: str
    log_path: str
    command: list[str]
    command_str: str
    environment: dict[str, str]
    plan_metadata: dict[str, object]
    status: str = "pending"
    return_code: int | None = None
    status_detail: str | None = None


@dataclass
class LaunchState:
    """Mutable state shared by worker threads."""

    runs: list[LaunchRun]
    manifest_path: Path
    dry_run: bool
    resume: bool
    overwrite: bool
    lock: threading.Lock = field(default_factory=threading.Lock)
    active_processes: set[subprocess.Popen[bytes]] = field(default_factory=set)
    stop_requested: threading.Event = field(default_factory=threading.Event)

    def persist(self) -> None:
        """Write an atomic status snapshot."""

        with self.lock:
            payload = {
                "schema_version": 1,
                "run_count": len(self.runs),
                "expected_full_run_count": EXPECTED_FULL_RUN_COUNT,
                "is_full_matrix": len(self.runs) == EXPECTED_FULL_RUN_COUNT,
                "dry_run": self.dry_run,
                "resume": self.resume,
                "overwrite": self.overwrite,
                "status_counts": count_statuses(self.runs),
                "runs": [asdict(run) for run in self.runs],
            }
            write_json_atomic(self.manifest_path, payload)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SOURCE_SEEDS))
    parser.add_argument("--noise-regimes", default=",".join(DEFAULT_NOISE_REGIMES))
    parser.add_argument("--optimizers", default=",".join(DEFAULT_OPTIMIZERS))
    parser.add_argument(
        "--gpus",
        default="0,1,2,3,4",
        help="Comma-separated physical GPU identifiers, paired in selected-seed order.",
    )
    parser.add_argument(
        "--aees-trace-root",
        type=Path,
        default=Path("results/cifar_noisy"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/cifar_attribution_controls"),
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--schedule-seed-base", type=int, default=20_260_816)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--run-ids", default=None, help="Optional comma-separated exact run IDs.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip completed valid artifacts (default: enabled).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow atomically replacing an existing artifact after a successful rerun.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Resolve one epoch per run instead of the 200-epoch production horizon.",
    )
    return parser.parse_args(argv)


def parse_csv(value: str, *, label: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError(f"--{label} must select at least one value.")
    if len(set(items)) != len(items):
        raise ValueError(f"--{label} contains duplicate values: {items}.")
    return items


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def stable_schedule_seed(base_seed: int, run_id: str) -> int:
    """Derive a stable, policy-only seed without touching a training RNG."""

    digest = hashlib.sha256(f"{base_seed}:{run_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF


def build_run_matrix(args: argparse.Namespace) -> list[LaunchRun]:
    """Build, validate, and fully resolve the selected experiment matrix."""

    policies = parse_csv(args.policies, label="policies")
    invalid_policies = sorted(set(policies) - set(DEFAULT_POLICIES))
    if invalid_policies:
        raise ValueError(f"Unknown policies: {invalid_policies}.")
    seed_tokens = parse_csv(args.seeds, label="seeds")
    try:
        seeds = tuple(int(token) for token in seed_tokens)
    except ValueError as exc:
        raise ValueError(f"--seeds must contain integers, got {seed_tokens}.") from exc
    if len(set(seeds)) != len(seeds) or any(seed not in SOURCE_SEEDS for seed in seeds):
        raise ValueError(f"--seeds must be unique members of {list(SOURCE_SEEDS)}.")
    noise_regimes = parse_csv(args.noise_regimes, label="noise-regimes")
    invalid_regimes = sorted(set(noise_regimes) - set(NOISE_REGIMES))
    if invalid_regimes:
        raise ValueError(f"Unknown noise regimes: {invalid_regimes}.")
    optimizers = parse_csv(args.optimizers, label="optimizers")
    invalid_optimizers = sorted(set(optimizers) - set(DEFAULT_OPTIMIZERS))
    if invalid_optimizers:
        raise ValueError(f"Unknown optimizers: {invalid_optimizers}.")
    gpu_ids = parse_csv(args.gpus, label="gpus")

    requested_ids = None
    if args.run_ids is not None:
        requested_ids = set(parse_csv(args.run_ids, label="run-ids"))

    epochs = 1 if args.smoke_test else 200
    episode_count = episode_count_from_training_config(
        epochs=epochs,
        batch_size=128,
        episode_length=200,
    )
    trace_root = resolve_path(args.aees_trace_root)
    output_root = resolve_path(args.output_root)
    raw: list[tuple[str, str, str, int, str]] = []
    for seed in seeds:
        for noise_regime in noise_regimes:
            for optimizer in optimizers:
                for policy in policies:
                    run_id = "_".join(
                        (
                            f"cifar100-{noise_regime}",
                            optimizer.lower(),
                            policy,
                            f"seed{seed}",
                        )
                    )
                    if requested_ids is None or run_id in requested_ids:
                        raw.append((run_id, policy, noise_regime, seed, optimizer))
    if requested_ids is not None:
        found_ids = {item[0] for item in raw}
        missing_ids = sorted(requested_ids - found_ids)
        if missing_ids:
            raise ValueError(f"Unknown or filtered --run-ids: {missing_ids}.")
    run_ids = [item[0] for item in raw]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Resolved run matrix contains duplicate run IDs.")

    selected_seeds = sorted({item[3] for item in raw})
    if len(gpu_ids) != len(selected_seeds):
        raise ValueError(
            f"Selected {len(selected_seeds)} seeds but received {len(gpu_ids)} GPU IDs; "
            "provide exactly one GPU per selected seed."
        )
    gpu_by_seed = dict(zip(selected_seeds, gpu_ids, strict=True))

    runs: list[LaunchRun] = []
    for run_id, policy, noise_regime, seed, optimizer in raw:
        schedule_seed = stable_schedule_seed(args.schedule_seed_base, run_id)
        if policy == "uniform-random":
            plan = build_uniform_random_plan(
                episode_count=episode_count,
                schedule_seed=schedule_seed,
            )
        else:
            plan = build_frequency_matched_plan(
                trace_root=trace_root,
                noise_regime=noise_regime,
                optimizer=optimizer,
                target_seed=seed,
                episode_count=episode_count,
                schedule_seed=schedule_seed,
            )
        noise_type, noise_rate = NOISE_REGIMES[noise_regime]
        optimizer_slug = optimizer.lower()
        policy_slug = policy.replace("-", "_")
        output_path = (
            output_root
            / f"cifar100_{noise_regime}_seed{seed}"
            / f"{optimizer_slug}_{policy_slug}.json"
        )
        log_path = output_root / "_logs" / f"{run_id}.log"
        command = [
            args.python_executable,
            str(PROJECT_ROOT / "experiments/task_cifar100.py"),
            "--control-mode",
            policy,
            "--epochs",
            str(epochs),
            "--batch-size",
            "128",
            "--lr",
            format(OPTIMIZER_BASE_LRS[optimizer], "g"),
            "--weight-decay",
            "0.0001",
            "--optimizer",
            optimizer,
            "--momentum",
            "0.9",
            "--lr-scheduler",
            "none",
            "--episode-length",
            "200",
            "--lr-candidates",
            "0.5,1.0,2.0",
            "--noise-candidates",
            "0.0",
            "--structured-control-mode",
            "independent",
            "--context-mode",
            "none",
            "--label-noise-type",
            noise_type,
            "--label-noise-rate",
            format(noise_rate, "g"),
            "--seed",
            str(seed),
            "--schedule-seed",
            str(schedule_seed),
            "--aees-trace-root",
            str(trace_root),
            "--run-tag",
            run_id,
            "--output",
            str(output_path),
        ]
        runs.append(
            LaunchRun(
                run_id=run_id,
                policy=policy,
                noise_regime=noise_regime,
                optimizer=optimizer,
                seed=seed,
                gpu_id=gpu_by_seed[seed],
                schedule_seed=schedule_seed,
                epochs=epochs,
                total_steps=math.ceil(50_000 / 128) * epochs,
                output_path=str(output_path),
                log_path=str(log_path),
                command=command,
                command_str=shlex.join(command),
                environment={"CUDA_VISIBLE_DEVICES": gpu_by_seed[seed]},
                plan_metadata=plan.to_metadata(),
            )
        )
    return runs


def validate_completed_result(run: LaunchRun) -> tuple[bool, str]:
    """Check that a result is complete and belongs to this exact run."""

    path = Path(run.output_path)
    if not path.exists():
        return False, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid JSON: {exc}"
    if not isinstance(payload, dict):
        return False, "top-level result is not an object"
    config = payload.get("config")
    logs = payload.get("episode_logs")
    if not isinstance(config, dict) or not isinstance(logs, dict):
        return False, "missing config or episode_logs"
    expected_pairs = {
        "control_mode": run.policy,
        "optimizer": run.optimizer,
        "seed": run.seed,
        "schedule_seed": run.schedule_seed,
    }
    for key, expected in expected_pairs.items():
        if config.get(key) != expected:
            return False, f"config.{key}={config.get(key)!r}, expected {expected!r}"
    metadata = config.get("open_loop_policy")
    if not isinstance(metadata, dict):
        return False, "missing config.open_loop_policy"
    planned = metadata.get("planned_lr_multipliers")
    executed = metadata.get("executed_lr_multipliers")
    logged = logs.get("selected_lr_values")
    expected_sequence = run.plan_metadata["planned_lr_multipliers"]
    if planned != expected_sequence or executed != expected_sequence or logged != expected_sequence:
        return False, "planned, executed, and logged schedules do not match the manifest"
    if payload.get("total_epochs") != run.epochs or payload.get("total_steps") != run.total_steps:
        return False, "training horizon is incomplete"
    if len(payload.get("val_accuracies", [])) != run.epochs:
        return False, "validation trajectory is incomplete"
    return True, "complete"


def prepare_statuses(state: LaunchState) -> None:
    """Classify existing outputs before execution without overwriting anything."""

    for run in state.runs:
        valid, detail = validate_completed_result(run)
        output_exists = Path(run.output_path).exists()
        if valid and state.resume and not state.overwrite:
            run.status = "skipped"
            run.status_detail = "completed valid result"
        elif output_exists and not state.overwrite:
            run.status = "failed"
            run.status_detail = (
                f"existing artifact is not safely reusable ({detail}); pass --overwrite to replace it"
            )
        else:
            run.status = "pending"
            run.status_detail = "dry-run only" if state.dry_run else None


def run_seed_worker(seed_runs: list[LaunchRun], state: LaunchState) -> None:
    """Execute one seed's runs sequentially on its paired GPU."""

    for run in seed_runs:
        if state.stop_requested.is_set():
            return
        if run.status != "pending":
            continue
        try:
            log_path = Path(run.log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update(run.environment)
            with log_path.open("w", encoding="utf-8") as log_file:
                log_file.write(f"command: {run.command_str}\n")
                log_file.write(f"CUDA_VISIBLE_DEVICES={run.gpu_id}\n\n")
                log_file.flush()
                process = subprocess.Popen(
                    run.command,
                    cwd=PROJECT_ROOT,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
                with state.lock:
                    state.active_processes.add(process)
                try:
                    return_code = process.wait()
                finally:
                    with state.lock:
                        state.active_processes.discard(process)
        except Exception as exc:
            run.status = "failed"
            run.status_detail = f"launcher could not execute run: {exc}"
            state.persist()
            continue
        run.return_code = return_code
        if return_code != 0:
            run.status = "failed"
            run.status_detail = f"training command exited {return_code}; see {run.log_path}"
        else:
            valid, detail = validate_completed_result(run)
            if valid:
                run.status = "completed"
                run.status_detail = "completed and validated"
            else:
                run.status = "failed"
                run.status_detail = f"command exited zero but result validation failed: {detail}"
        state.persist()


def terminate_active_processes(state: LaunchState) -> None:
    """Terminate all active children, escalating to kill only if necessary."""

    state.stop_requested.set()
    with state.lock:
        processes = list(state.active_processes)
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is not None:
            continue
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def count_statuses(runs: Sequence[LaunchRun]) -> dict[str, int]:
    statuses = ("completed", "skipped", "failed", "pending")
    return {status: sum(run.status == status for run in runs) for status in statuses}


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            json.dump(payload, temporary_file, indent=2, sort_keys=True)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def print_matrix(runs: Sequence[LaunchRun]) -> None:
    print(f"Resolved {len(runs)} unique runs:")
    for index, run in enumerate(runs, start=1):
        print(
            f"[{index:02d}/{len(runs):02d}] gpu={run.gpu_id} seed={run.seed} "
            f"regime={run.noise_regime} optimizer={run.optimizer} policy={run.policy}"
        )
        print(f"  output: {run.output_path}")
        print(f"  command: CUDA_VISIBLE_DEVICES={shlex.quote(run.gpu_id)} {run.command_str}")
    if len(runs) == EXPECTED_FULL_RUN_COUNT:
        print("Full-matrix check: exactly 60 unique runs (PASS).")
    else:
        print(f"Subset check: {len(runs)} unique selected runs (full matrix is 60).")


def print_summary(state: LaunchState, args: argparse.Namespace) -> int:
    counts = count_statuses(state.runs)
    print(
        "Final summary: "
        + ", ".join(f"{status}={count}" for status, count in counts.items())
    )
    print(f"Manifest: {state.manifest_path}")
    failed = [run for run in state.runs if run.status == "failed"]
    if failed:
        print("Retry failed runs:")
        for run in failed:
            retry = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--run-ids",
                run.run_id,
                "--gpus",
                run.gpu_id,
                "--aees-trace-root",
                str(resolve_path(args.aees_trace_root)),
                "--output-root",
                str(resolve_path(args.output_root)),
                "--schedule-seed-base",
                str(args.schedule_seed_base),
                "--overwrite",
            ]
            if args.smoke_test:
                retry.append("--smoke-test")
            print(f"  {shlex.join(retry)}")
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        runs = build_run_matrix(args)
    except (ValueError, OSError) as exc:
        print(f"Launcher validation failed: {exc}", file=sys.stderr)
        return 2
    print_matrix(runs)
    manifest_path = resolve_path(args.manifest) if args.manifest else resolve_path(
        args.output_root
    ) / "manifest.json"
    state = LaunchState(
        runs=runs,
        manifest_path=manifest_path,
        dry_run=args.dry_run,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    prepare_statuses(state)
    state.persist()
    if args.dry_run:
        print("Dry run: no training processes were started.")
        return print_summary(state, args)

    seed_groups: dict[int, list[LaunchRun]] = {}
    for run in runs:
        seed_groups.setdefault(run.seed, []).append(run)
    executor = ThreadPoolExecutor(max_workers=len(seed_groups))
    futures = [
        executor.submit(run_seed_worker, seed_runs, state)
        for _, seed_runs in sorted(seed_groups.items())
    ]
    try:
        for future in as_completed(futures):
            future.result()
    except KeyboardInterrupt:
        print("Interrupt received; terminating active training processes...", file=sys.stderr)
        terminate_active_processes(state)
        executor.shutdown(wait=True, cancel_futures=True)
        for run in state.runs:
            if run.status == "pending":
                run.status_detail = "interrupted before completion"
        state.persist()
        return 130
    except Exception as exc:
        print(f"Launcher worker failed unexpectedly: {exc}", file=sys.stderr)
        terminate_active_processes(state)
        executor.shutdown(wait=True, cancel_futures=True)
        for run in state.runs:
            if run.status == "pending":
                run.status_detail = "cancelled after an unexpected launcher failure"
        state.persist()
        return 1
    else:
        executor.shutdown(wait=True)
    state.persist()
    return print_summary(state, args)


if __name__ == "__main__":
    raise SystemExit(main())

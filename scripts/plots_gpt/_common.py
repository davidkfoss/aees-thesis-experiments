"""Shared result-loading helpers for thesis plot scripts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import pathlib
from typing import Any, Callable, Generator, Iterable

from scripts.plots_gpt._style import PALETTE


Record = dict[str, Any]


@dataclass(frozen=True)
class RunInfo:
    """Normalized identity for one archived training run.

    ``variant_key`` is deliberately limited to the shared plotting palette so
    every figure uses stable colors. ``variant_id`` keeps a more specific label
    for smoke tests and sanity summaries where fixed-LR or optimizer-specific
    rows must not collapse together.
    """

    task: str
    optimizer: str
    variant_key: str
    scheduler: str
    seed: int | None
    noise_setting: str
    is_aees: bool
    lr_active: bool = False
    noise_active: bool = False
    is_fixed: bool = False
    variant_id: str | None = None


def walk_runs(root: str | pathlib.Path) -> Generator[tuple[pathlib.Path, Record], None, None]:
    """Yield ``(path, record)`` for every JSON-decodable file below ``root``.

    Result archives include extensionless CIFAR-noisy files, so this intentionally
    does not filter on ``.json``. Logs, unreadable files, and non-JSON files are
    skipped silently per ``TASKS.md``.
    """

    root = pathlib.Path(root)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix == ".log":
            continue
        try:
            with path.open() as handle:
                record = json.load(handle)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(record, dict):
            yield path, record


def identify(record: Record) -> RunInfo:
    """Identify a run using the decision tree in ``scripts/plots/TASKS.md``."""

    config = record.get("config") or {}
    scheduler = _normalize_scheduler(config.get("lr_scheduler"))
    task = str(config.get("task_name") or record.get(
        "task_name") or "unknown").lower()
    optimizer = _normalize_optimizer(config, record)
    seed = _as_int(record.get("seed", config.get("seed")))
    noise_setting = _noise_setting(config)

    lr_candidates = _candidate_list(config.get("lr_candidates"))
    noise_candidates = _candidate_list(config.get("noise_candidates"))
    lr_active = len(lr_candidates) > 1
    noise_active = len(noise_candidates) > 1

    controller_logs = record.get("controller_logs")
    fixed_controller = _has_only_fixed_controllers(controller_logs)
    is_fixed = bool(config.get("fixed_mode_name")) or (
        controller_logs is not None and fixed_controller)
    is_aees = controller_logs is not None and not is_fixed and (
        lr_active or noise_active)

    if not is_aees:
        variant_key = scheduler if scheduler != "none" else "flat"
    elif scheduler != "none" and (lr_active or noise_active):
        variant_key = f"{scheduler}_aees"
        if variant_key not in PALETTE:
            variant_key = _axis_variant_key(lr_active, noise_active)
    else:
        variant_key = _axis_variant_key(
            lr_active, noise_active, plain_lr_as_aees=True)

    if variant_key not in PALETTE:
        raise ValueError(f"Unrecognized plot variant key: {variant_key}")

    variant_id = _variant_id(
        optimizer=optimizer,
        variant_key=variant_key,
        scheduler=scheduler,
        lr_active=lr_active,
        noise_active=noise_active,
        is_aees=is_aees,
        is_fixed=is_fixed,
        lr_candidates=lr_candidates,
        noise_candidates=noise_candidates,
    )

    return RunInfo(
        task=task,
        optimizer=optimizer,
        variant_key=variant_key,
        scheduler=scheduler,
        seed=seed,
        noise_setting=noise_setting,
        is_aees=is_aees,
        lr_active=lr_active,
        noise_active=noise_active,
        is_fixed=is_fixed,
        variant_id=variant_id,
    )


def episode_to_epoch(record: Record, episode_index: int) -> int:
    """Map a 0-based episode index to the nearest 0-based epoch index."""

    episode_logs = record.get("episode_logs") or {}
    end_steps = episode_logs.get("episode_end_steps") or []
    total_steps = int(record.get("total_steps") or 0)
    total_epochs = int(record.get("total_epochs") or 0)
    if total_steps <= 0 or total_epochs <= 0:
        raise ValueError(
            "record must define positive total_steps and total_epochs")
    if episode_index < 0 or episode_index >= len(end_steps):
        raise IndexError(f"episode_index {episode_index} out of range")
    epoch_float = float(end_steps[episode_index]) * total_epochs / total_steps
    return max(0, min(total_epochs - 1, round(epoch_float) - 1))


def epoch_step_range(record: Record, epoch_index: int) -> tuple[float, float]:
    """Return the half-open optimizer-step range for a 0-based epoch index."""

    total_steps = int(record.get("total_steps") or 0)
    total_epochs = int(record.get("total_epochs") or 0)
    if total_steps <= 0 or total_epochs <= 0:
        raise ValueError(
            "record must define positive total_steps and total_epochs")
    if epoch_index < 0 or epoch_index >= total_epochs:
        raise IndexError(f"epoch_index {epoch_index} out of range")
    steps_per_epoch = total_steps / total_epochs
    return epoch_index * steps_per_epoch, (epoch_index + 1) * steps_per_epoch


def load_set(
    root: str | pathlib.Path,
    predicate: Callable[[pathlib.Path, Record, RunInfo], bool] | Callable[[Record, RunInfo], bool],
) -> dict[str, dict[int, list[tuple[pathlib.Path, Record, RunInfo]]]]:
    """Load runs under ``root`` grouped as ``variant_key -> seed -> runs``.

    The predicate may accept either ``(record, info)`` or ``(path, record, info)``.
    """

    grouped: dict[str, dict[int, list[tuple[pathlib.Path, Record, RunInfo]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for path, record in walk_runs(root):
        try:
            info = identify(record)
        except (KeyError, TypeError, ValueError):
            continue
        if not _predicate_accepts(predicate, path, record, info):
            continue
        if info.seed is None:
            continue
        grouped[info.variant_key][info.seed].append((path, record, info))
    return {variant: dict(seeds) for variant, seeds in grouped.items()}


def _predicate_accepts(
    predicate: Callable[..., bool], path: pathlib.Path, record: Record, info: RunInfo
) -> bool:
    try:
        return bool(predicate(path, record, info))
    except TypeError:
        return bool(predicate(record, info))


def _candidate_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        return [float(v) for v in value]
    return [float(value)]


def _normalize_scheduler(value: Any) -> str:
    scheduler = str(value or "none").lower()
    return "none" if scheduler in {"", "null", "none"} else scheduler


def _normalize_optimizer(config: Record, record: Record) -> str:
    raw = (
        config.get("optimizer")
        or config.get("optimizer_family")
        or config.get("base_optimizer")
        or config.get("method")
        or record.get("optimizer")
        or record.get("method_name")
        or ""
    )
    opt = str(raw).lower()
    if "sgd" in opt:
        return "sgd"
    if "adam" in opt:
        return "adamw"

    task = str(config.get("task_name")
               or record.get("task_name") or "").lower()
    base_lr = config.get("base_lr", config.get("lr"))
    try:
        lr = float(base_lr)
    except (TypeError, ValueError):
        lr = None
    if task == "cifar100" and lr is not None and lr >= 0.01:
        return "sgd"
    return "adamw"


def _noise_setting(config: Record) -> str:
    rate = config.get("label_noise_rate")
    noise_type = config.get("label_noise_type")
    try:
        rate_float = float(rate or 0.0)
    except (TypeError, ValueError):
        rate_float = 0.0
    if not noise_type or str(noise_type).lower() in {"none", "clean"} or rate_float == 0:
        return "clean"
    prefix = "sym" if str(noise_type).lower().startswith("sym") else "asym"
    return f"{prefix}{int(round(rate_float * 100))}"


def _has_only_fixed_controllers(controller_logs: Any) -> bool:
    if not isinstance(controller_logs, dict):
        return False
    controllers = [
        controller_logs.get("lr_controller_logs") or {},
        controller_logs.get("noise_controller_logs") or {},
    ]
    controller_types = [str(c.get("controller_type") or "")
                        for c in controllers if c]
    return bool(controller_types) and all(t == "FixedModeController" for t in controller_types)


def _axis_variant_key(
    lr_active: bool, noise_active: bool, *, plain_lr_as_aees: bool = False
) -> str:
    if lr_active and noise_active:
        return "aees_dual"
    if lr_active:
        return "aees" if plain_lr_as_aees else "aees_lr"
    if noise_active:
        return "aees_noise"
    return "aees"


def _variant_id(
    *,
    optimizer: str,
    variant_key: str,
    scheduler: str,
    lr_active: bool,
    noise_active: bool,
    is_aees: bool,
    is_fixed: bool,
    lr_candidates: list[float],
    noise_candidates: list[float],
) -> str:
    if is_fixed:
        lr = lr_candidates[0] if lr_candidates else 1.0
        noise = noise_candidates[0] if noise_candidates else 0.0
        return f"{optimizer}_fixed_lr{_compact_float(lr)}_n{_compact_float(noise)}_{scheduler}"
    if is_aees:
        if lr_active and noise_active:
            axis = "aees_dual"
        elif lr_active:
            axis = "aees_lr"
        elif noise_active:
            axis = "aees_noise"
        else:
            axis = "aees"
        return f"{optimizer}_{axis}_{scheduler}"
    return f"{optimizer}_{variant_key}"


def _compact_float(value: float) -> str:
    text = f"{value:g}".replace(".", "p").replace("-", "m")
    if text.startswith("0p"):
        text = text[1:]
    return text


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

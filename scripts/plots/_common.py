"""Shared run-discovery and identification helpers for thesis figure scripts.

Every figure script under scripts/plots/<category>/ imports `walk_runs`,
`identify`, `load_set`, and the small episode/epoch math helpers from here.
The contract is: one decision tree for identifying a run lives in `identify()`
so every plot agrees on what variant a JSON file represents.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any, Callable, Iterator


# ---------------------------------------------------------------------------
# Run-info dataclass returned by identify().
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunInfo:
    task: str | None              # "cifar100", "agnews", "sst2"
    optimizer: str | None         # "AdamW", "SGD", ...
    variant_key: str              # key into PALETTE/LABEL, e.g. "aees_dual"
    scheduler: str                # "none" | "cosine" | "linear" | "warmup_linear"
    seed: int | None
    noise_setting: str            # "clean" | "sym20" | "sym40" | "asym20" | ...
    is_aees: bool
    fixed_multiplier: bool        # True for ctrl_logs-present-but-no-arms ablations
    path: pathlib.Path | None = None


# ---------------------------------------------------------------------------
# Filesystem walker. JSON-decodes anything that isn't a .log, ignores files
# that fail to parse. Intentionally not filtered on suffix — CIFAR-noisy
# files are extensionless.
# ---------------------------------------------------------------------------


def walk_runs(root: str | pathlib.Path) -> Iterator[tuple[pathlib.Path, dict[str, Any]]]:
    """Yield (path, record_dict) for every JSON-decodable file under `root`."""
    root = pathlib.Path(root)
    if not root.exists():
        return
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix == ".log":
            continue
        try:
            with open(p) as f:
                r = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
        if not isinstance(r, dict):
            continue
        yield p, r


# ---------------------------------------------------------------------------
# Variant-identification decision tree. Matches the "Run identification"
# section of TASKS.md.
# ---------------------------------------------------------------------------


_OPTIMIZER_TOKENS = ("AdamW", "SGD", "SGDM")


def _detect_optimizer(r: dict, path: pathlib.Path | None) -> str | None:
    cfg = r.get("config", {})
    for key in ("optimizer_family", "optimizer"):
        v = cfg.get(key)
        if v:
            return v
    method = cfg.get("method")
    if method in _OPTIMIZER_TOKENS:
        return method
    mname = r.get("method_name")
    if mname in _OPTIMIZER_TOKENS:
        return mname
    if path is not None:
        s = str(path).lower()
        if "sgd" in s:
            return "SGD"
        if "adamw" in s:
            return "AdamW"
        # cifar_clean AEES runs encode the optimizer only by directory
        if "cifar_clean" in s or "cifar_clean_aees" in s:
            return "AdamW"
    # AG News and SST-2 are AdamW-only across this thesis; the field is just
    # not always populated in the AEES JSONs (method_name == "AdaptiveScheduler").
    task = cfg.get("task_name")
    if task in ("agnews", "sst2"):
        return "AdamW"
    return None


def _detect_noise_setting(r: dict) -> str:
    cfg = r.get("config", {})
    nt = cfg.get("label_noise_type")
    nr = cfg.get("label_noise_rate")
    if nt is None or (isinstance(nt, str) and nt.lower() == "none"):
        return "clean"
    if not nr:
        return "clean"
    pct = int(round(float(nr) * 100))
    if nt == "symmetric":
        return f"sym{pct}"
    if nt == "asymmetric":
        return f"asym{pct}"
    return f"{nt}{pct}"


def identify(r: dict, path: pathlib.Path | None = None) -> RunInfo:
    """Map a single run-record to a RunInfo.

    See TASKS.md "Run identification" — this is the single source of truth.
    `variant_key` is one of the keys in PALETTE/LABEL when the run fits one of
    the main variants; fixed-multiplier ablations get variant_key="fixed_multiplier"
    and is_aees=False so callers can filter them out.
    """
    cfg = r.get("config", {})
    task = cfg.get("task_name")
    sched = cfg.get("lr_scheduler") or "none"
    if isinstance(sched, str) and sched == "":
        sched = "none"
    lr_cands = cfg.get("lr_candidates") or []
    noise_cands = cfg.get("noise_candidates") or []
    lr_active = len(lr_cands) > 1
    noise_active = len(noise_cands) > 1
    has_ctrl = r.get("controller_logs") is not None
    seed = cfg.get("seed")
    if seed is None:
        seed = r.get("seed")
    noise_setting = _detect_noise_setting(r)
    optimizer = _detect_optimizer(r, path)

    fixed_multiplier = False
    if not has_ctrl:
        is_aees = False
        variant_key = "flat" if sched == "none" else sched
    else:
        if not lr_active and not noise_active:
            # fixed-multiplier ablation: a controller exists but every axis
            # has a single arm, so there's no online adaptation
            is_aees = False
            fixed_multiplier = True
            variant_key = "fixed_multiplier"
        else:
            is_aees = True
            if lr_active and noise_active:
                aees_type = "aees_dual"
            elif lr_active:
                aees_type = "aees_lr"
            else:
                aees_type = "aees_noise"
            if sched == "none":
                variant_key = aees_type
            else:
                variant_key = f"{sched}_aees"

    return RunInfo(
        task=task,
        optimizer=optimizer,
        variant_key=variant_key,
        scheduler=sched,
        seed=seed,
        noise_setting=noise_setting,
        is_aees=is_aees,
        fixed_multiplier=fixed_multiplier,
        path=path,
    )


# ---------------------------------------------------------------------------
# Episode <-> epoch helpers. Episodes and epochs are independently indexed
# in the schema; align them via episode_end_steps / total_steps / total_epochs.
# ---------------------------------------------------------------------------


def episode_to_epoch(r: dict, episode_index: int) -> int:
    """Return the epoch (0-indexed) the i-th completed episode ends in."""
    end_steps = r["episode_logs"]["episode_end_steps"][episode_index]
    total_steps = r["total_steps"]
    total_epochs = r["total_epochs"]
    return int(round(end_steps * total_epochs / total_steps))


def epoch_step_range(r: dict, epoch_index: int) -> tuple[int, int]:
    """Step range [lo, hi) belonging to epoch `epoch_index` (0-indexed)."""
    total_steps = r["total_steps"]
    total_epochs = r["total_epochs"]
    steps_per_epoch = total_steps / total_epochs
    lo = int(round(epoch_index * steps_per_epoch))
    hi = int(round((epoch_index + 1) * steps_per_epoch))
    return lo, hi


# ---------------------------------------------------------------------------
# Aggregating helper. Walks `root`, applies `predicate(info, record)`, and
# groups surviving runs by variant_key. Returns {variant_key: [(info, record), ...]}.
# ---------------------------------------------------------------------------


PredicateFn = Callable[[RunInfo, dict[str, Any]], bool]


def load_set(
    root: str | pathlib.Path,
    predicate: PredicateFn | None = None,
) -> dict[str, list[tuple[RunInfo, dict[str, Any]]]]:
    """Return runs under `root` keyed by variant_key.

    `predicate(info, record)` returning False filters the run out. When
    omitted, every successfully-parsed run is included.
    """
    out: dict[str, list[tuple[RunInfo, dict[str, Any]]]] = {}
    for path, r in walk_runs(root):
        info = identify(r, path)
        if predicate is not None and not predicate(info, r):
            continue
        out.setdefault(info.variant_key, []).append((info, r))
    return out

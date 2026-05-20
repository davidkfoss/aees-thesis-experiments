#!/usr/bin/env python3
"""Generate the LaTeX table for the noisy AG News σ-axis ablation.

Loads the 7 conditions (σ=0 baselines + 5 σ>0 conditions) from results/,
aggregates best and final validation accuracies (mean ± sample std) across
5 seeds, and emits a LaTeX `tabular` matching the styling of
make_nlp_latex_tables.py (bold best, italic second-best, drop in pp).

CLI:
    python scripts/tables/make_nlp_noise_ablation_table.py \\
        --runs-root results \\
        --out results/tables/nlp_noise_ablation_table.tex
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Any, Callable


# Conditions in display order (σ=0 baselines first, then σ>0 cluster).
@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    predicate: Callable[[dict, pathlib.Path], bool]


def _cfg(r: dict) -> dict:
    c = r.get("config", {})
    return c if isinstance(c, dict) else {}


def _is_noisy_agnews(r: dict) -> bool:
    c = _cfg(r)
    return (
        c.get("task_name") == "agnews"
        and c.get("label_noise_type") == "symmetric"
        and float(c.get("label_noise_rate") or 0.0) == 0.2
    )


def _close_list(a, b, atol: float = 1e-9) -> bool:
    if a is None or b is None:
        return False
    if len(a) != len(b):
        return False
    return all(abs(float(x) - float(y)) <= atol for x, y in zip(a, b))


def _predicate(method_name: str, lr_scheduler: str, lr_cands, noise_cands):
    def fn(r: dict, _p: pathlib.Path) -> bool:
        if not _is_noisy_agnews(r):
            return False
        if r.get("method_name") != method_name:
            return False
        c = _cfg(r)
        if c.get("lr_scheduler") != lr_scheduler:
            return False
        if not _close_list(c.get("lr_candidates"), lr_cands):
            return False
        return _close_list(c.get("noise_candidates"), noise_cands)
    return fn


CONDITIONS: list[Condition] = [
    Condition(
        "adamw_flat", "AdamW (Flat, $\\sigma=0$)",
        _predicate("AdamW", "none", [1.0], [0.0]),
    ),
    Condition(
        "adamw_wl", "AdamW + WL ($\\sigma=0$)",
        _predicate("AdamW", "warmup_linear", [1.0], [0.0]),
    ),
    Condition(
        "fixed_005", "Fixed $\\sigma=0.005$ + WL",
        _predicate("RandomScheduler", "warmup_linear", [1.0], [0.005]),
    ),
    Condition(
        "fixed_01", "Fixed $\\sigma=0.01$ + WL",
        _predicate("RandomScheduler", "warmup_linear", [1.0], [0.01]),
    ),
    Condition(
        "random_sigma", "Random $\\sigma$ + WL",
        _predicate("RandomScheduler", "warmup_linear", [1.0], [0.0, 0.005, 0.01]),
    ),
    Condition(
        "aees_noise_wl", "AEES-Noise + WL",
        _predicate("AdaptiveScheduler", "warmup_linear", [1.0], [0.0, 0.005, 0.01]),
    ),
    Condition(
        "aees_dual_wl", "AEES-Dual + WL",
        _predicate(
            "AdaptiveScheduler", "warmup_linear",
            [0.5, 1.0, 2.0], [0.0, 0.005, 0.01],
        ),
    ),
]


def _walk_runs(root: pathlib.Path):
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


@dataclass
class CellStats:
    cond: Condition
    seeds: list[int]
    bests: list[float]
    finals: list[float]
    drops: list[float]
    peak_epochs: list[int]

    @property
    def best_mean(self) -> float:
        return mean(self.bests) * 100.0

    @property
    def best_std(self) -> float:
        return stdev(self.bests) * 100.0 if len(self.bests) > 1 else 0.0

    @property
    def final_mean(self) -> float:
        return mean(self.finals) * 100.0

    @property
    def final_std(self) -> float:
        return stdev(self.finals) * 100.0 if len(self.finals) > 1 else 0.0

    @property
    def drop_mean(self) -> float:
        return mean(self.drops) * 100.0

    @property
    def drop_std(self) -> float:
        return stdev(self.drops) * 100.0 if len(self.drops) > 1 else 0.0

    @property
    def peak_epoch_mean(self) -> float:
        return mean(self.peak_epochs)

    @property
    def peak_epoch_std(self) -> float:
        return stdev(self.peak_epochs) if len(self.peak_epochs) > 1 else 0.0


def _collect(runs_root: pathlib.Path) -> dict[str, CellStats]:
    buckets: dict[str, dict[int, dict[str, float]]] = {c.key: {} for c in CONDITIONS}
    for path, record in _walk_runs(runs_root):
        for cond in CONDITIONS:
            if cond.predicate(record, path):
                seed = record.get("seed")
                if seed is None:
                    seed = _cfg(record).get("seed")
                if seed is None:
                    continue
                seed = int(seed)
                best = float(record["best_val_accuracy"])
                final = float(record["final_val_accuracy"])
                vals = record.get("val_accuracies") or []
                if vals:
                    peak_epoch = int(max(range(len(vals)), key=lambda i: vals[i])) + 1
                else:
                    peak_epoch = -1
                buckets[cond.key][seed] = {
                    "best": best,
                    "final": final,
                    "drop": best - final,
                    "peak_epoch": peak_epoch,
                }
                break

    out: dict[str, CellStats] = {}
    for cond in CONDITIONS:
        per_seed = buckets[cond.key]
        if not per_seed:
            raise SystemExit(f"No runs matched condition: {cond.key} ({cond.label})")
        seeds = sorted(per_seed.keys())
        out[cond.key] = CellStats(
            cond=cond,
            seeds=seeds,
            bests=[per_seed[s]["best"] for s in seeds],
            finals=[per_seed[s]["final"] for s in seeds],
            drops=[per_seed[s]["drop"] for s in seeds],
            peak_epochs=[per_seed[s]["peak_epoch"] for s in seeds],
        )
    return out


# ---------------------------------------------------------------------------
# Styling.
# ---------------------------------------------------------------------------


def _rank(values: dict[str, float], higher_is_better: bool) -> dict[str, str]:
    """Returns {cond_key: style} where style is "best", "second", or ""."""
    ordered = sorted(values.items(), key=lambda kv: kv[1], reverse=higher_is_better)
    style = {k: "" for k in values}
    if len(ordered) >= 1:
        style[ordered[0][0]] = "best"
    if len(ordered) >= 2:
        style[ordered[1][0]] = "second"
    return style


def _fmt_pm(m: float, s: float, digits: int = 2) -> str:
    return f"{m:.{digits}f} $\\pm$ {s:.{digits}f}"


def _styled(text: str, style: str) -> str:
    if style == "best":
        return f"\\textbf{{{text}}}"
    if style == "second":
        return f"\\emph{{{text}}}"
    return text


def _build_table(cells: dict[str, CellStats]) -> str:
    best_rank = _rank({k: c.best_mean for k, c in cells.items()}, higher_is_better=True)
    final_rank = _rank({k: c.final_mean for k, c in cells.items()}, higher_is_better=True)
    drop_rank = _rank({k: c.drop_mean for k, c in cells.items()}, higher_is_better=False)

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Noisy AG News $\sigma$-axis falsification ablation. "
        r"All seven conditions use AdamW (base $\eta = 5\times10^{-5}$) on AG News "
        r"with 20\% symmetric training-label noise; clean validation labels. "
        r"`Fixed $\sigma$' injects a constant gradient-noise standard deviation. "
        r"`Random $\sigma$' samples $\sigma \in \{0, 0.005, 0.01\}$ uniformly at "
        r"random per episode (no reward feedback). AEES variants use the discounted-UCB "
        r"controller. Best and final validation accuracy are mean $\pm$ standard "
        r"deviation across $n$ seeds. Drop is best$-$final in percentage points (lower is better). "
        r"Bold and italics mark the best and second-best values per column.}"
    )
    lines.append(r"\label{tab:nlp_noise_ablation}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Condition & $n$ & Best (\%) & Final (\%) & Drop (pp) & Peak epoch \\"
    )
    lines.append(r"\midrule")

    n_baselines = 2  # adamw_flat, adamw_wl
    for idx, cond in enumerate(CONDITIONS):
        if idx == n_baselines:
            lines.append(r"\midrule")
        c = cells[cond.key]
        best_cell = _styled(_fmt_pm(c.best_mean, c.best_std), best_rank[cond.key])
        final_cell = _styled(_fmt_pm(c.final_mean, c.final_std), final_rank[cond.key])
        drop_cell = _styled(_fmt_pm(c.drop_mean, c.drop_std), drop_rank[cond.key])

        peak_cell = _fmt_pm(c.peak_epoch_mean, c.peak_epoch_std)
        lines.append(
            f"{cond.label} & {len(c.seeds)} & {best_cell} & {final_cell} & "
            f"{drop_cell} & {peak_cell} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root", type=pathlib.Path, default=pathlib.Path("results"),
        help="Parent directory to walk recursively for noisy AG News runs.",
    )
    parser.add_argument(
        "--out", type=pathlib.Path,
        default=pathlib.Path("results/tables/nlp_noise_ablation_table.tex"),
        help="Output .tex path.",
    )
    args = parser.parse_args()

    if not args.runs_root.exists():
        raise SystemExit(f"--runs-root not found: {args.runs_root}")

    cells = _collect(args.runs_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(_build_table(cells))
    print(f"wrote {args.out}")

    # Print summary table for sanity.
    print()
    print(f"{'condition':<26}  {'n':>2}  {'best':>15}  {'final':>15}  {'drop':>10}")
    print("-" * 80)
    for cond in CONDITIONS:
        c = cells[cond.key]
        print(
            f"{cond.key:<26}  {len(c.seeds):>2}  "
            f"{c.best_mean:>7.3f} ± {c.best_std:>5.3f}  "
            f"{c.final_mean:>7.3f} ± {c.final_std:>5.3f}  "
            f"{c.drop_mean:>+7.3f} pp"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

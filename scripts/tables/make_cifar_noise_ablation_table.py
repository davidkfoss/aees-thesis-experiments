#!/usr/bin/env python3
"""Generate the LaTeX table for the noisy CIFAR-100 small-noise-injection ablation.

Loads two matched AEES controller configurations on noisy CIFAR-100 (symmetric
20% label noise) from ``archived_results/cifar_noise_ablation/seed_*/``:

  - No injection (``lronly``):              noise_candidates = [0.0]
  - Small-noise injection (``smallnoise``): noise_candidates = [0.0, 0.0025, 0.005]

Both share episode length 200, the ``trend`` context mode, and a 100-epoch
budget; n = 5 seeds. Aggregates peak (best) and final validation accuracy
(mean +/- sample SD), and the peak-to-final gap, and emits a LaTeX tabular
(bold marks the higher/better value per accuracy column) plus a CSV companion.

Caveat: both conditions use the LR candidate grid {0.5, 1.0, 2.0} except the
no-injection seed 0, which used {0.5, 1.0, 1.5}. The gradient-noise injection
axis is the systematic difference. This is noted in the table footnote.

CLI:
    uv run python scripts/tables/make_cifar_noise_ablation_table.py \\
        --runs-root archived_results/cifar_noise_ablation \\
        --out-dir reproduced_artifacts/tables/cifar_ablation
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys
from collections import Counter
from dataclasses import dataclass
from statistics import mean, stdev


# Two conditions in display order: no-injection baseline first.
TAGS = ["lronly", "smallnoise"]
LABEL = {
    "lronly": "No gradient noise",
    "smallnoise": "Gradient-noise injection",
}
_SEED_RE = re.compile(r"_seed(\d+)$")


@dataclass
class CellStats:
    tag: str
    label: str
    seeds: list[int]
    bests: list[float]   # fractions
    finals: list[float]  # fractions
    lr_candidates: list  # modal grid across seeds
    noise_candidates: list
    lr_grids: list       # all distinct LR grids seen across seeds

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
    def gap(self) -> float:
        return self.best_mean - self.final_mean


def _collect(runs_root: pathlib.Path) -> dict[str, CellStats]:
    cells: dict[str, CellStats] = {}
    for tag in TAGS:
        pattern = f"seed_*/noisy_coarse_{tag}_ep200_trend_seed*.json"
        files = sorted(runs_root.glob(pattern))
        if not files:
            raise SystemExit(
                f"No runs matched condition '{tag}': {pattern} under {runs_root}")
        per_seed: dict[int, dict] = {}
        for f in files:
            record = json.loads(f.read_text())
            m = _SEED_RE.search(f.stem)
            seed = int(m.group(1)) if m else record.get("seed")
            cfg = record.get("config", {})
            per_seed[int(seed)] = {
                "best": float(record["best_val_accuracy"]),
                "final": float(record["final_val_accuracy"]),
                "lr": cfg.get("lr_candidates"),
                "noise": cfg.get("noise_candidates"),
            }
        seeds = sorted(per_seed)
        lr_modal, lr_grids = _grid_stats(per_seed[s]["lr"] for s in seeds)
        noise_modal, _ = _grid_stats(per_seed[s]["noise"] for s in seeds)
        cells[tag] = CellStats(
            tag=tag,
            label=LABEL[tag],
            seeds=seeds,
            bests=[per_seed[s]["best"] for s in seeds],
            finals=[per_seed[s]["final"] for s in seeds],
            lr_candidates=lr_modal,
            noise_candidates=noise_modal,
            lr_grids=lr_grids,
        )
    return cells


def _grid_stats(grids) -> tuple[list, list]:
    """Return (modal grid, sorted distinct grids) from per-seed candidate grids."""
    counter = Counter(tuple(g) for g in grids if g is not None)
    if not counter:
        return [], []
    return list(counter.most_common(1)[0][0]), [list(g) for g in sorted(counter)]


def _fmt_pm(m: float, s: float) -> str:
    return f"{m:.2f} $\\pm$ {s:.2f}"


def _bold_higher(cells: dict[str, CellStats], attr: str) -> dict[str, bool]:
    """Mark the condition with the higher value of ``attr`` (better accuracy)."""
    best_tag = max(TAGS, key=lambda t: getattr(cells[t], attr))
    return {t: (t == best_tag) for t in TAGS}


def _styled(text: str, bold: bool) -> str:
    return f"\\textbf{{{text}}}" if bold else text


def _build_table(cells: dict[str, CellStats]) -> str:
    bold_best = _bold_higher(cells, "best_mean")
    bold_final = _bold_higher(cells, "final_mean")
    n = len(cells[TAGS[0]].seeds)

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Small-noise gradient-injection ablation on noisy CIFAR-100 "
        r"(symmetric 20\% training-label noise). Both conditions use the AEES "
        r"controller with the \texttt{trend} context, episode length 200, over a "
        r"100-epoch budget; $n$ seeds. `No injection' restricts the controller to "
        r"$\sigma=0$ (learning-rate multiplier axis only); `Small-noise injection' "
        r"adds $\sigma \in \{0.0, 0.0025, 0.005\}$ gradient-noise candidate arms. Peak and "
        r"final validation accuracy are mean $\pm$ sample standard deviation; gap is "
        r"peak$-$final in percentage points. Bold marks the higher (better) value per "
        r"accuracy column. Note: both conditions use the learning-rate candidate grid "
        r"$\{0.5,1.0,2.0\}$ (the no-injection seed~0 used $\{0.5,1.0,1.5\}$), so the "
        r"gradient-noise injection axis is the systematic difference between conditions.}"
    )
    lines.append(r"\label{tab:cifar_noise_ablation}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{6pt}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Condition & $n$ & Peak (\%) & Final (\%) & Gap (pp) \\")
    lines.append(r"\midrule")
    for tag in TAGS:
        c = cells[tag]
        best_cell = _styled(_fmt_pm(c.best_mean, c.best_std), bold_best[tag])
        final_cell = _styled(
            _fmt_pm(c.final_mean, c.final_std), bold_final[tag])
        lines.append(
            f"{c.label} & {len(c.seeds)} & {best_cell} & {final_cell} & {c.gap:.2f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def _write_csv(cells: dict[str, CellStats], path: pathlib.Path) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "condition", "tag", "n", "seeds", "noise_candidates", "lr_candidates",
            "peak_mean_pct", "peak_std_pct", "final_mean_pct", "final_std_pct", "gap_pp",
        ])
        for tag in TAGS:
            c = cells[tag]
            w.writerow([
                c.label, c.tag, len(c.seeds),
                " ".join(str(s) for s in c.seeds),
                " ".join(str(v) for v in (c.noise_candidates or [])),
                " ".join(str(v) for v in (c.lr_candidates or [])),
                f"{c.best_mean:.4f}", f"{c.best_std:.4f}",
                f"{c.final_mean:.4f}", f"{c.final_std:.4f}", f"{c.gap:.4f}",
            ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root", type=pathlib.Path,
        default=pathlib.Path("archived_results/cifar_noise_ablation"),
        help="Directory containing the cifar_noise_ablation seed_*/ result JSONs.",
    )
    parser.add_argument(
        "--out-dir", type=pathlib.Path,
        default=pathlib.Path("reproduced_artifacts/tables/cifar_ablation"),
        help="Output directory for the .tex and .csv files.",
    )
    args = parser.parse_args()

    if not args.runs_root.exists():
        raise SystemExit(f"--runs-root not found: {args.runs_root}")

    cells = _collect(args.runs_root)

    # Seed-coverage summary (matches the other table scripts).
    print()
    print("=" * 38)
    print("CIFAR NOISE-INJECTION ABLATION COVERAGE")
    print("=" * 38)
    for tag in TAGS:
        c = cells[tag]
        print(f"  {c.label:<22} ({tag}) n={len(c.seeds)} seeds={c.seeds}")
        print(
            f"      noise_candidates(modal)={c.noise_candidates}  lr_candidates(modal)={c.lr_candidates}")
        if len(c.lr_grids) > 1:
            print(f"      NOTE: LR grid varies across seeds: {c.lr_grids}")
        print(f"      peak={c.best_mean:.2f} ± {c.best_std:.2f}   "
              f"final={c.final_mean:.2f} ± {c.final_std:.2f}   gap={c.gap:.2f} pp")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tex_path = args.out_dir / "cifar_noise_ablation_table.tex"
    csv_path = args.out_dir / "cifar_noise_ablation_summary.csv"
    tex_path.write_text(_build_table(cells))
    _write_csv(cells, csv_path)
    print()
    print(f"Wrote {tex_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

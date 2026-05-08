#!/usr/bin/env python3
"""
Create LaTeX tables for CIFAR-100 peak-checkpoint diagnostics.

Expected input layout:
  results/checkpointing/cifar100_asym20/*.json
  results/checkpointing/cifar100_sym40/*.json

Expected output:
  results/tables/checkpointing/checkpoint_diagnostics_main.tex
  results/tables/checkpointing/checkpoint_diagnostics_appendix.tex
  results/tables/checkpointing/checkpoint_diagnostics_main.csv
  results/tables/checkpointing/checkpoint_diagnostics_detailed.csv

The rerun files are assumed to stop at the selected/peak epoch, so the final
diagnostics in each JSON file are interpreted as peak-checkpoint diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any


@dataclass(frozen=True)
class Row:
    setting_key: str
    setting_label: str
    method_key: str
    optimizer_key: str
    optimizer_label: str
    variant_key: str
    variant_label: str
    seed: int
    selected_epoch: int
    val_acc: float
    corr_noisy: float
    corr_clean: float
    train_clean: float


SETTING_ORDER = {
    "asym20": 0,
    "sym40": 1,
}

SETTING_LABELS = {
    "asym20": r"Asym. 20\%",
    "sym40": r"Sym. 40\%",
}

OPTIMIZER_ORDER = {
    "adamw": 0,
    "sgd": 1,
}

OPTIMIZER_LABELS = {
    "adamw": r"\adamw{}",
    "sgd": r"\sgdm{}",
}

VARIANT_ORDER = {
    "baseline": 0,
    "aees": 1,
    "cosine": 2,
    "cosine_aees": 3,
}

VARIANT_LABELS = {
    "baseline": r"Baseline",
    "aees": r"\method{}",
    "cosine": r"Cosine",
    "cosine_aees": r"Cosine + \method{}",
}

METHOD_ORDER = {
    "adamw": 0,
    "adamw_aees": 1,
    "adamw_cosine": 2,
    "adamw_cosine_aees": 3,
    "sgd": 4,
    "sgd_aees": 5,
    "sgd_cosine": 6,
    "sgd_cosine_aees": 7,
}

METHOD_TO_OPT_VARIANT = {
    "adamw": ("adamw", "baseline"),
    "adamw_aees": ("adamw", "aees"),
    "adamw_cosine": ("adamw", "cosine"),
    "adamw_cosine_aees": ("adamw", "cosine_aees"),
    "sgd": ("sgd", "baseline"),
    "sgd_aees": ("sgd", "aees"),
    "sgd_cosine": ("sgd", "cosine"),
    "sgd_cosine_aees": ("sgd", "cosine_aees"),
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_setting(path: Path, data: dict[str, Any]) -> tuple[str, str]:
    text = str(path).lower()
    cfg = data.get("config", {})
    noise_type = str(cfg.get("label_noise_type", "")).lower()
    noise_rate = float(cfg.get("label_noise_rate", 0.0))

    if "asym20" in text or (noise_type == "asymmetric" and abs(noise_rate - 0.2) < 1e-9):
        return "asym20", SETTING_LABELS["asym20"]

    if "sym40" in text or (noise_type == "symmetric" and abs(noise_rate - 0.4) < 1e-9):
        return "sym40", SETTING_LABELS["sym40"]

    raise ValueError(f"Could not infer setting for {path}")


def infer_method_key(data: dict[str, Any], path: Path) -> str:
    cfg = data.get("config", {})
    optimizer = str(cfg.get("optimizer", "")).lower()
    control_mode = str(cfg.get("control_mode", "")).lower()
    scheduler = str(
        cfg.get("lr_scheduler", cfg.get("scheduler", "none"))).lower()
    scheduler_active = bool(
        cfg.get("scheduler_active", scheduler not in {"", "none"}))

    # Fallbacks from filename, useful if config fields differ.
    name = path.stem.lower()

    is_sgd = (
        optimizer == "sgd"
        or name.startswith("sgd_")
        or "_sgd_" in name
        or "sgd_" in name
    )
    is_adamw = (
        optimizer == "adamw"
        or name.startswith("adamw_")
        or "_adamw_" in name
        or "adamw_" in name
    )
    is_aees = control_mode == "adaptive" or "_aees" in name or "aees_" in name
    is_cosine = (
        scheduler == "cosine"
        or scheduler_active
        or "_cosine" in name
        or "cosine_" in name
    )

    if is_adamw:
        if is_cosine and is_aees:
            return "adamw_cosine_aees"
        if is_cosine:
            return "adamw_cosine"
        if is_aees:
            return "adamw_aees"
        return "adamw"

    if is_sgd:
        if is_cosine and is_aees:
            return "sgd_cosine_aees"
        if is_cosine:
            return "sgd_cosine"
        if is_aees:
            return "sgd_aees"
        return "sgd"

    raise ValueError(f"Could not infer optimizer/method for {path}")


def get_seed(data: dict[str, Any], path: Path) -> int:
    if "seed" in data:
        return int(data["seed"])

    cfg = data.get("config", {})
    if "seed" in cfg:
        return int(cfg["seed"])

    m = re.search(r"seed(\d+)", path.stem)
    if m:
        return int(m.group(1))

    raise ValueError(f"Could not infer seed for {path}")


def get_selected_epoch(data: dict[str, Any]) -> int:
    # These reruns should stop at the selected/peak epoch.
    if "total_epochs" in data:
        return int(data["total_epochs"])

    cfg = data.get("config", {})
    if "epochs" in cfg:
        return int(cfg["epochs"])

    if "val_accuracies" in data:
        return len(data["val_accuracies"])

    raise ValueError("Could not infer selected epoch")


def get_val_acc(data: dict[str, Any]) -> float:
    # Because reruns stop at peak, final_val_accuracy should equal selected checkpoint accuracy.
    if "final_val_accuracy" in data:
        return float(data["final_val_accuracy"])

    if "best_val_accuracy" in data:
        return float(data["best_val_accuracy"])

    vals = data.get("val_accuracies", [])
    if vals:
        return float(vals[-1])

    raise ValueError("Could not infer validation accuracy")


def get_diag(data: dict[str, Any], key: str) -> float:
    cfg = data.get("config", {})
    diagnostics = cfg.get("diagnostics", {})
    subset = diagnostics.get("final_subset_accuracies", {})

    if key not in subset:
        raise ValueError(f"Missing diagnostic key: {key}")

    return float(subset[key])


def parse_result(path: Path) -> Row:
    data = load_json(path)

    setting_key, setting_label = infer_setting(path, data)
    method_key = infer_method_key(data, path)
    optimizer_key, variant_key = METHOD_TO_OPT_VARIANT[method_key]

    return Row(
        setting_key=setting_key,
        setting_label=setting_label,
        method_key=method_key,
        optimizer_key=optimizer_key,
        optimizer_label=OPTIMIZER_LABELS[optimizer_key],
        variant_key=variant_key,
        variant_label=VARIANT_LABELS[variant_key],
        seed=get_seed(data, path),
        selected_epoch=get_selected_epoch(data),
        val_acc=get_val_acc(data),
        corr_noisy=get_diag(data, "train_corrupted_accuracy_vs_noisy_labels"),
        corr_clean=get_diag(data, "train_corrupted_accuracy_vs_clean_labels"),
        train_clean=get_diag(data, "train_clean_accuracy"),
    )


def sd(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def fmt_mean_sd(values: list[float], *, percent: bool = False, decimals: int = 1) -> str:
    if not values:
        return "--"

    vals = [v * 100 for v in values] if percent else values
    m = mean(vals)
    s = sd(vals)
    return rf"{m:.{decimals}f} $\pm$ {s:.{decimals}f}"


def fmt_single(value: float, *, percent: bool = False, decimals: int = 1) -> str:
    value = value * 100 if percent else value
    return f"{value:.{decimals}f}"


def aggregate(rows: list[Row]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Row]] = {}

    for row in rows:
        grouped.setdefault((row.setting_key, row.method_key), []).append(row)

    out: list[dict[str, Any]] = []

    for (setting_key, method_key), group in grouped.items():
        group = sorted(group, key=lambda r: r.seed)
        optimizer_key, variant_key = METHOD_TO_OPT_VARIANT[method_key]

        selected_epochs = [r.selected_epoch for r in group]
        val_accs = [r.val_acc for r in group]
        corr_noisy = [r.corr_noisy for r in group]
        corr_clean = [r.corr_clean for r in group]
        train_clean = [r.train_clean for r in group]

        out.append({
            "setting_key": setting_key,
            "setting_label": group[0].setting_label,
            "method_key": method_key,
            "optimizer_key": optimizer_key,
            "optimizer_label": OPTIMIZER_LABELS[optimizer_key],
            "variant_key": variant_key,
            "variant_label": VARIANT_LABELS[variant_key],
            "n": len(group),

            "selected_epoch": fmt_mean_sd(selected_epochs, percent=False, decimals=1),
            "val_acc": fmt_mean_sd(val_accs, percent=True, decimals=1),
            "corr_noisy": fmt_mean_sd(corr_noisy, percent=True, decimals=1),
            "corr_clean": fmt_mean_sd(corr_clean, percent=True, decimals=1),
            "train_clean": fmt_mean_sd(train_clean, percent=True, decimals=1),

            "selected_epoch_mean": mean(selected_epochs),
            "val_acc_mean": mean(val_accs),
            "corr_noisy_mean": mean(corr_noisy),
            "corr_clean_mean": mean(corr_clean),
            "train_clean_mean": mean(train_clean),

            "selected_epoch_sd": sd(selected_epochs),
            "val_acc_sd": sd(val_accs),
            "corr_noisy_sd": sd(corr_noisy),
            "corr_clean_sd": sd(corr_clean),
            "train_clean_sd": sd(train_clean),
        })

    return sorted(
        out,
        key=lambda r: (
            SETTING_ORDER[r["setting_key"]],
            OPTIMIZER_ORDER[r["optimizer_key"]],
            VARIANT_ORDER[r["variant_key"]],
        ),
    )


def rank_values(
    rows: list[dict[str, Any]],
    metric_key: str,
    *,
    higher_is_better: bool,
) -> tuple[set[str], set[str]]:
    """
    Return one best and one second-best method key within a block.

    Ranking uses:
      1. metric mean, with direction given by higher_is_better
      2. lower standard deviation as tie-breaker
      3. variant order as deterministic final tie-breaker

    This intentionally avoids multiple bold/italic entries in compact diagnostic tables.
    """
    if not rows:
        return set(), set()

    sd_key = metric_key.replace("_mean", "_sd")

    def sort_key(row: dict[str, Any]) -> tuple[float, float, int]:
        mean_value = float(row[metric_key])
        sd_value = float(row.get(sd_key, 0.0))
        primary = -mean_value if higher_is_better else mean_value
        return (
            primary,
            sd_value,
            VARIANT_ORDER[row["variant_key"]],
        )

    ranked = sorted(rows, key=sort_key)

    best = {ranked[0]["method_key"]}
    second = {ranked[1]["method_key"]} if len(ranked) > 1 else set()

    return best, second


def decorate_cell(
    cell: str,
    method_key: str,
    best_methods: set[str],
    second_methods: set[str],
) -> str:
    if method_key in best_methods:
        return rf"\textbf{{{cell}}}"
    if method_key in second_methods:
        return rf"\emph{{{cell}}}"
    return cell


def diagnostic_rank_maps(block: list[dict[str, Any]]) -> dict[str, tuple[set[str], set[str]]]:
    return {
        "corr_noisy": rank_values(block, "corr_noisy_mean", higher_is_better=False),
        "corr_clean": rank_values(block, "corr_clean_mean", higher_is_better=True),
    }


def write_main_csv(path: Path, aggregated: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "setting_label",
        "optimizer_label",
        "variant_label",
        "n",
        "selected_epoch",
        "val_acc",
        "corr_noisy",
        "corr_clean",
        "train_clean",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in aggregated:
            writer.writerow({k: row[k] for k in fields})


def write_detailed_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "setting_label",
        "seed",
        "optimizer_label",
        "variant_label",
        "selected_epoch",
        "val_acc",
        "corr_noisy",
        "corr_clean",
        "train_clean",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for r in rows:
            writer.writerow({
                "setting_label": r.setting_label,
                "seed": r.seed,
                "optimizer_label": r.optimizer_label,
                "variant_label": r.variant_label,
                "selected_epoch": r.selected_epoch,
                "val_acc": r.val_acc,
                "corr_noisy": r.corr_noisy,
                "corr_clean": r.corr_clean,
                "train_clean": r.train_clean,
            })


def write_main_latex(path: Path, aggregated: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    setting_keys = sorted(
        {r["setting_key"] for r in aggregated},
        key=lambda k: SETTING_ORDER[k],
    )

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption[Peak-checkpoint diagnostics on noisy \cifar{}]{"
        r"Peak-checkpoint diagnostics on noisy \cifar{}. "
        r"Rows are grouped by noise setting and optimizer. "
        r"The selected epoch is the epoch at which the checkpoint was evaluated. "
        r"Accuracy columns are reported as mean $\pm$ standard deviation over five seeds. "
        r"``Corr. noisy'' is accuracy on corrupted training examples measured against corrupted labels, "
        r"where lower values indicate less memorization of corrupted targets. "
        r"``Corr. clean'' evaluates the same corrupted examples against their original clean labels. "
        r"``Train clean'' is accuracy on the uncorrupted training examples and is included as a sanity check. "
        r"Arrows indicate the preferred direction for the corrupted-subset diagnostics. "
        r"Within each noise-setting and optimizer block, bold and italics mark the best and second-best corrupted-subset diagnostic values, respectively; "
        r"ties in the mean are resolved by lower standard deviation."
        r"}"
    )
    lines.append(r"\label{tab:checkpoint-diagnostics-main}")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lllccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Setting & Opt. & Variant & Epoch & Val acc. & Corr. noisy $\downarrow$ & Corr. clean $\uparrow$ & Train clean \\"
    )
    lines.append(r"\midrule")

    for s_idx, setting_key in enumerate(setting_keys):
        setting_block = [
            r for r in aggregated if r["setting_key"] == setting_key]
        setting_block = sorted(
            setting_block,
            key=lambda r: (
                OPTIMIZER_ORDER[r["optimizer_key"]],
                VARIANT_ORDER[r["variant_key"]],
            ),
        )

        first_setting_row = True
        opt_keys = sorted(
            {r["optimizer_key"] for r in setting_block},
            key=lambda k: OPTIMIZER_ORDER[k],
        )

        for o_idx, opt_key in enumerate(opt_keys):
            opt_block = [
                r for r in setting_block if r["optimizer_key"] == opt_key]
            opt_block = sorted(
                opt_block, key=lambda r: VARIANT_ORDER[r["variant_key"]])

            ranks = diagnostic_rank_maps(opt_block)
            corr_noisy_best, corr_noisy_second = ranks["corr_noisy"]
            corr_clean_best, corr_clean_second = ranks["corr_clean"]

            for j, row in enumerate(opt_block):
                setting_cell = (
                    rf"\multirow{{{len(setting_block)}}}{{*}}{{{row['setting_label']}}}"
                    if first_setting_row else ""
                )
                opt_cell = (
                    rf"\multirow{{{len(opt_block)}}}{{*}}{{{row['optimizer_label']}}}"
                    if j == 0 else ""
                )

                corr_noisy_cell = decorate_cell(
                    row["corr_noisy"],
                    row["method_key"],
                    corr_noisy_best,
                    corr_noisy_second,
                )
                corr_clean_cell = decorate_cell(
                    row["corr_clean"],
                    row["method_key"],
                    corr_clean_best,
                    corr_clean_second,
                )

                lines.append(
                    f"{setting_cell} & {opt_cell} & {row['variant_label']} & "
                    f"{row['selected_epoch']} & {row['val_acc']} & "
                    f"{corr_noisy_cell} & {corr_clean_cell} & {row['train_clean']} \\\\"
                )

                first_setting_row = False

            if o_idx != len(opt_keys) - 1:
                lines.append(r"\cmidrule(lr){2-8}")

        if s_idx != len(setting_keys) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_appendix_latex(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = sorted(
        rows,
        key=lambda r: (
            SETTING_ORDER[r.setting_key],
            r.seed,
            OPTIMIZER_ORDER[r.optimizer_key],
            VARIANT_ORDER[r.variant_key],
        ),
    )

    lines: list[str] = []
    lines.append(r"\begingroup")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{longtable}{lll l c c c c c}")
    lines.append(
        r"\caption[Detailed peak-checkpoint diagnostics on noisy \cifar{}]{"
        r"Detailed peak-checkpoint diagnostics on noisy \cifar{}. "
        r"Accuracy values are reported in percent. "
        r"The diagnostics are computed at the selected checkpoint because each rerun was stopped at the corresponding peak epoch. "
        r"Arrows indicate the preferred direction for the corrupted-subset diagnostics. "
        r"Train clean is included as a sanity check."
        r"}"
    )
    lines.append(r"\label{tab:checkpoint-diagnostics-detailed}\\")
    lines.append(r"\toprule")
    lines.append(
        r"Setting & Seed & Opt. & Variant & Epoch & Val acc. & Corr. noisy $\downarrow$ & Corr. clean $\uparrow$ & Train clean \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(
        r"\caption[]{Detailed peak-checkpoint diagnostics on noisy \cifar{} continued.}\\"
    )
    lines.append(r"\toprule")
    lines.append(
        r"Setting & Seed & Opt. & Variant & Epoch & Val acc. & Corr. noisy $\downarrow$ & Corr. clean $\uparrow$ & Train clean \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{9}{r}{Continued on next page}\\")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    setting_keys = sorted({r.setting_key for r in rows},
                          key=lambda k: SETTING_ORDER[k])

    for s_idx, setting_key in enumerate(setting_keys):
        setting_rows = [r for r in rows if r.setting_key == setting_key]
        setting_rows = sorted(
            setting_rows,
            key=lambda r: (
                r.seed,
                OPTIMIZER_ORDER[r.optimizer_key],
                VARIANT_ORDER[r.variant_key],
            ),
        )
        setting_len = len(setting_rows)
        first_setting_row = True

        seed_values = sorted({r.seed for r in setting_rows})

        for seed_idx, seed in enumerate(seed_values):
            seed_rows = [r for r in setting_rows if r.seed == seed]
            seed_rows = sorted(
                seed_rows,
                key=lambda r: (
                    OPTIMIZER_ORDER[r.optimizer_key],
                    VARIANT_ORDER[r.variant_key],
                ),
            )
            seed_len = len(seed_rows)
            first_seed_row = True

            opt_keys = sorted({r.optimizer_key for r in seed_rows},
                              key=lambda k: OPTIMIZER_ORDER[k])

            for opt_idx, opt_key in enumerate(opt_keys):
                opt_rows = [r for r in seed_rows if r.optimizer_key == opt_key]
                opt_rows = sorted(
                    opt_rows, key=lambda r: VARIANT_ORDER[r.variant_key])
                opt_len = len(opt_rows)
                first_opt_row = True

                for r in opt_rows:
                    setting_cell = (
                        rf"\multirow{{{setting_len}}}{{*}}{{{r.setting_label}}}"
                        if first_setting_row else ""
                    )
                    seed_cell = (
                        rf"\multirow{{{seed_len}}}{{*}}{{{seed}}}"
                        if first_seed_row else ""
                    )
                    opt_cell = (
                        rf"\multirow{{{opt_len}}}{{*}}{{{r.optimizer_label}}}"
                        if first_opt_row else ""
                    )

                    lines.append(
                        f"{setting_cell} & {seed_cell} & {opt_cell} & {r.variant_label} & "
                        f"{r.selected_epoch} & "
                        f"{fmt_single(r.val_acc, percent=True, decimals=1)} & "
                        f"{fmt_single(r.corr_noisy, percent=True, decimals=1)} & "
                        f"{fmt_single(r.corr_clean, percent=True, decimals=1)} & "
                        f"{fmt_single(r.train_clean, percent=True, decimals=1)} \\\\"
                    )

                    first_setting_row = False
                    first_seed_row = False
                    first_opt_row = False

                if opt_idx != len(opt_keys) - 1:
                    lines.append(r"\cmidrule(lr){3-9}")

            if seed_idx != len(seed_values) - 1:
                lines.append(r"\cmidrule(lr){2-9}")

        if s_idx != len(setting_keys) - 1:
            lines.append(r"\midrule")

    lines.append(r"\end{longtable}")
    lines.append(r"\endgroup")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_completeness(rows: list[Row]) -> list[str]:
    warnings: list[str] = []
    by_setting_method: dict[tuple[str, str], list[int]] = {}

    for r in rows:
        by_setting_method.setdefault(
            (r.setting_key, r.method_key), []).append(r.seed)

    expected_settings = ["asym20", "sym40"]
    expected_methods = list(METHOD_ORDER.keys())

    for setting in expected_settings:
        for method in expected_methods:
            seeds = sorted(by_setting_method.get((setting, method), []))
            if seeds != [0, 1, 2, 3, 4]:
                warnings.append(
                    f"Missing or unexpected seeds for {setting}/{method}: "
                    f"found {seeds}, expected [0, 1, 2, 3, 4]"
                )

    return warnings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root results directory containing checkpointing/cifar100_asym20 and checkpointing/cifar100_sym40.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tables/checkpointing"),
        help="Directory where LaTeX and CSV tables are written.",
    )
    args = parser.parse_args()

    json_paths = sorted(
        list((args.results_root / "checkpointing" / "cifar100_asym20").glob("*.json"))
        + list((args.results_root / "checkpointing" /
               "cifar100_sym40").glob("*.json"))
    )

    if not json_paths:
        raise SystemExit(
            f"No JSON files found under "
            f"{args.results_root / 'checkpointing' / 'cifar100_asym20'} or "
            f"{args.results_root / 'checkpointing' / 'cifar100_sym40'}"
        )

    rows: list[Row] = []
    errors: list[str] = []

    for path in json_paths:
        try:
            rows.append(parse_result(path))
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    if errors:
        print("Skipped files with parsing errors:")
        for e in errors:
            print(f"  - {e}")

    if not rows:
        raise SystemExit("No valid result rows parsed.")

    rows = sorted(
        rows,
        key=lambda r: (
            SETTING_ORDER[r.setting_key],
            r.seed,
            OPTIMIZER_ORDER[r.optimizer_key],
            VARIANT_ORDER[r.variant_key],
        ),
    )
    aggregated = aggregate(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_main_latex(
        args.output_dir / "checkpoint_diagnostics_main.tex",
        aggregated,
    )
    write_appendix_latex(
        args.output_dir / "checkpoint_diagnostics_appendix.tex",
        rows,
    )
    write_main_csv(
        args.output_dir / "checkpoint_diagnostics_main.csv",
        aggregated,
    )
    write_detailed_csv(
        args.output_dir / "checkpoint_diagnostics_detailed.csv",
        rows,
    )

    warnings = validate_completeness(rows)

    print(f"Parsed {len(rows)} result files.")
    print(f"Wrote {args.output_dir / 'checkpoint_diagnostics_main.tex'}")
    print(f"Wrote {args.output_dir / 'checkpoint_diagnostics_appendix.tex'}")
    print(f"Wrote {args.output_dir / 'checkpoint_diagnostics_main.csv'}")
    print(f"Wrote {args.output_dir / 'checkpoint_diagnostics_detailed.csv'}")

    if warnings:
        print("\nCompleteness warnings:")
        for w in warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main()

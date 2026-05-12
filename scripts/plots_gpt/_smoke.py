"""Inventory smoke test for archived result folders used by thesis plots."""

from __future__ import annotations

from collections import Counter, defaultdict
import pathlib
import re
import sys

from scripts.plots_gpt._common import identify, walk_runs


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"


def main() -> int:
    failures: list[str] = []

    failures.extend(_check_simple_setting("cifar_clean", 40, 8, 5))
    failures.extend(_check_cifar_noisy())
    failures.extend(_check_simple_setting("noisy_agnews", 30, 6, 5))
    failures.extend(_check_simple_setting("clean_agnews", 20, 4, 5))
    failures.extend(_check_simple_setting("sst2", 30, 6, 5))
    failures.extend(_check_compute_overhead())
    failures.extend(_check_checkpointing())

    if failures:
        print("\nFAILURES")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("\nAll smoke checks passed.")
    return 0


def _check_simple_setting(
    setting: str, expected_total: int, expected_variants: int, expected_per_variant: int
) -> list[str]:
    root = RESULTS / setting
    failures: list[str] = []
    counts: Counter[str] = Counter()
    unidentified: list[str] = []

    for path, record in walk_runs(root):
        try:
            identify(record)
        except Exception as exc:  # noqa: BLE001 - smoke should diagnose all bad files.
            unidentified.append(f"{path.relative_to(REPO_ROOT)} ({exc})")
            continue
        counts[_strip_seed(path.stem)] += 1

    _print_counts(setting, counts)
    total = sum(counts.values())
    if total != expected_total:
        failures.append(
            f"{setting}: expected {expected_total} files, found {total}")
    if len(counts) != expected_variants:
        failures.append(
            f"{setting}: expected {expected_variants} variants, found {len(counts)}")
    bad = {name: n for name, n in counts.items() if n != expected_per_variant}
    if bad:
        failures.append(
            f"{setting}: expected {expected_per_variant} seeds per variant, off cells={bad}")
    if unidentified:
        failures.append(f"{setting}: unidentified files: {unidentified}")
    return failures


def _check_cifar_noisy() -> list[str]:
    setting = "cifar_noisy"
    root = RESULTS / setting
    expected_variants = {"sym20": 14, "sym40": 14, "asym20": 16}
    failures: list[str] = []
    by_noise: dict[str, Counter[str]] = defaultdict(Counter)
    unidentified: list[str] = []

    for path, record in walk_runs(root):
        try:
            info = identify(record)
        except Exception as exc:  # noqa: BLE001
            unidentified.append(f"{path.relative_to(REPO_ROOT)} ({exc})")
            continue
        by_noise[info.noise_setting][path.name] += 1

    print(f"\n{setting}")
    for noise in sorted(expected_variants):
        counts = by_noise.get(noise, Counter())
        print(f"  {noise}: {sum(counts.values())} files, {len(counts)} variants")
        for name, count in sorted(counts.items()):
            print(f"    {name}: {count}")
        if len(counts) != expected_variants[noise]:
            failures.append(
                f"{setting}/{noise}: expected {expected_variants[noise]} variants, found {len(counts)}"
            )
        bad = {name: n for name, n in counts.items() if n != 5}
        if bad:
            failures.append(
                f"{setting}/{noise}: expected 5 seeds per variant, off cells={bad}")
    extra_noise = sorted(set(by_noise) - set(expected_variants))
    if extra_noise:
        failures.append(f"{setting}: unexpected noise groups {extra_noise}")
    if unidentified:
        failures.append(f"{setting}: unidentified files: {unidentified}")
    return failures


def _check_compute_overhead() -> list[str]:
    setting = "compute_overhead"
    root = RESULTS / setting
    failures: list[str] = []
    counts: Counter[tuple[str, str, str, str]] = Counter()
    unidentified: list[str] = []

    pattern = re.compile(
        r"^(?P<task>cifar_clean|agnews_clean|sst2_clean)_"
        r"(?P<optimizer>adamw|sgd)_"
        r"(?P<variant>baseline|aees_lr_only|aees_noise_only|aees_both)_"
        r"(?P<scheduler>none|cosine|warmup_linear)_seed0(?:_run\d+)?$"
    )
    for path, record in walk_runs(root):
        try:
            identify(record)
        except Exception as exc:  # noqa: BLE001
            unidentified.append(f"{path.relative_to(REPO_ROOT)} ({exc})")
            continue
        match = pattern.match(path.stem)
        if not match:
            unidentified.append(
                f"{path.relative_to(REPO_ROOT)} (filename did not match)")
            continue
        groups = match.groupdict()
        counts[
            (
                groups["task"],
                groups["optimizer"],
                groups["variant"],
                groups["scheduler"],
            )
        ] += 1

    cells = {(task, opt) for task, opt, _, _ in counts}
    print(
        f"\n{setting}: {sum(counts.values())} files, {len(cells)} task/optimizer cells")
    for cell in sorted(cells):
        cell_counts = {
            f"{variant}:{scheduler}": n
            for task, opt, variant, scheduler in sorted(counts)
            if (task, opt) == cell
            for n in [counts[(task, opt, variant, scheduler)]]
        }
        print(
            f"  {cell[0]}/{cell[1]}: {sum(cell_counts.values())} files, {len(cell_counts)} variants")
        for name, count in cell_counts.items():
            print(f"    {name}: {count}")
        if len(cell_counts) != 8:
            failures.append(
                f"{setting}/{cell}: expected 8 variant+scheduler groups, found {len(cell_counts)}")
        bad = {name: n for name, n in cell_counts.items() if n != 3}
        if bad:
            failures.append(
                f"{setting}/{cell}: expected 3 repeats per group, off cells={bad}")
    if sum(counts.values()) != 96:
        failures.append(
            f"{setting}: expected 96 files, found {sum(counts.values())}")
    if len(cells) != 4:
        failures.append(
            f"{setting}: expected 4 task/optimizer cells, found {len(cells)}")
    if unidentified:
        failures.append(f"{setting}: unidentified files: {unidentified}")
    return failures


def _check_checkpointing() -> list[str]:
    setting = "checkpointing"
    root = RESULTS / setting
    failures: list[str] = []
    by_noise: dict[str, Counter[str]] = defaultdict(Counter)
    unidentified: list[str] = []

    for path, record in walk_runs(root):
        try:
            info = identify(record)
        except Exception as exc:  # noqa: BLE001
            unidentified.append(f"{path.relative_to(REPO_ROOT)} ({exc})")
            continue
        by_noise[info.noise_setting][_checkpoint_variant(path.stem)] += 1

    print(f"\n{setting}: {sum(sum(c.values()) for c in by_noise.values())} files")
    for noise, counts in sorted(by_noise.items()):
        print(f"  {noise}: {sum(counts.values())} files, {len(counts)} variants")
        for name, count in sorted(counts.items()):
            print(f"    {name}: {count}")
        if len(counts) != 8:
            failures.append(
                f"{setting}/{noise}: expected 8 variants, found {len(counts)}")
        bad = {name: n for name, n in counts.items() if n != 5}
        if bad:
            failures.append(
                f"{setting}/{noise}: expected 5 seeds per variant, off cells={bad}")
    if set(by_noise) != {"asym20", "sym40"}:
        failures.append(
            f"{setting}: expected noise groups asym20/sym40, found {sorted(by_noise)}")
    total = sum(sum(c.values()) for c in by_noise.values())
    if total != 80:
        failures.append(f"{setting}: expected 80 files, found {total}")
    if unidentified:
        failures.append(f"{setting}: unidentified files: {unidentified}")
    return failures


def _print_counts(setting: str, counts: Counter[str]) -> None:
    print(f"\n{setting}: {sum(counts.values())} files, {len(counts)} variants")
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count}")


def _strip_seed(stem: str) -> str:
    stem = re.sub(r"_seed\d+(?:_run\d+)?$", "", stem)
    stem = re.sub(r"_run\d+$", "", stem)
    return stem


def _checkpoint_variant(stem: str) -> str:
    return re.sub(r"^cifar100_(?:asym20|sym40)_seed\d+_", "", stem)


if __name__ == "__main__":
    sys.exit(main())

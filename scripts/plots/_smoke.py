"""Walk every results/<setting>/ and print variant x seed counts.

Run before the figure scripts to confirm the loader sees what TASKS.md expects.
Prints one block per setting, plus a final OK / MISMATCH summary. Files that
don't fit any expected variant are listed at the bottom.
"""

from __future__ import annotations

import pathlib
import sys
from collections import defaultdict

from scripts.plots._common import identify, walk_runs


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"


# Expected variant-key x seed totals per setting. Numbers come from TASKS.md.
EXPECTED: dict[str, int] = {
    "cifar_clean": 40,            # 8 variants x 5 seeds
    "cifar_noisy": 14 * 5 * 2 + 16 * 5,  # sym20+sym40 (14 each) + asym20 (16)
    "noisy_agnews": 6 * 5,        # 30
    "clean_agnews": 4 * 5,        # 20 (5 .log files skipped)
    "sst2": 6 * 5,                # 30
    "compute_overhead": 8 * 4 * 3,  # 96
    "checkpointing": 8 * 5 * 2,   # 80
}


_SEED_RE = __import__("re").compile(
    r"_?seed\d+", flags=__import__("re").IGNORECASE)
_RUN_RE = __import__("re").compile(
    r"_run\d+$", flags=__import__("re").IGNORECASE)


def _file_variant_token(path: pathlib.Path) -> str:
    """Filename stem with seed/run/.json stripped, so sibling seeds collapse together."""
    stem = path.name
    if stem.endswith(".json"):
        stem = stem[:-5]
    stem = _SEED_RE.sub("", stem)
    stem = _RUN_RE.sub("", stem)
    return stem.rstrip("_")


def _bucket_key(info, path: pathlib.Path) -> str:
    """Disambiguating key (variant_key + optimizer + noise + filename token)."""
    return (
        f"{info.optimizer or '?'}/{info.variant_key}/{info.noise_setting}/"
        f"{_file_variant_token(path)}"
    )


def _audit_setting(setting: str) -> tuple[int, dict[str, list[int]], list[pathlib.Path]]:
    setting_dir = RESULTS / setting
    n = 0
    counts: dict[str, list[int]] = defaultdict(list)
    bad: list[pathlib.Path] = []
    for path, r in walk_runs(setting_dir):
        n += 1
        info = identify(r, path)
        key = _bucket_key(info, path)
        if info.seed is None:
            bad.append(path)
            continue
        counts[key].append(info.seed)
    return n, counts, bad


def main() -> int:
    rc = 0
    print(f"Smoke audit at {RESULTS}\n")
    grand = {}
    for setting, expected in EXPECTED.items():
        n, counts, bad = _audit_setting(setting)
        grand[setting] = n
        status = "OK" if n == expected else "MISMATCH"
        print(f"=== {setting} ({n} files, expect {expected}) — {status} ===")
        for key in sorted(counts):
            seeds = sorted(counts[key])
            print(f"  {key:90s}  n={len(seeds):2d}  seeds={seeds}")
        if bad:
            print(f"  WARN: {len(bad)} files with seed=None:")
            for p in bad[:10]:
                print(f"    {p.relative_to(REPO_ROOT)}")
            if len(bad) > 10:
                print(f"    ... ({len(bad) - 10} more)")
        if n != expected:
            rc = 1
        print()

    print("Totals:")
    for k, v in grand.items():
        marker = " " if v == EXPECTED[k] else "!"
        print(f"  {marker} {k:20s} {v:4d}  (expect {EXPECTED[k]})")
    print()
    print("OK" if rc == 0 else "MISMATCH — fix before continuing to Phase 1.")
    return rc


if __name__ == "__main__":
    sys.exit(main())

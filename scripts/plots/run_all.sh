#!/usr/bin/env bash
#
# Regenerate every figure using archived-results defaults.
#
# Each script reads from archived_results/ and writes to
# reproduced_artifacts/figures/<category>/. Run from anywhere; this script
# cd's to the repo root first. Continues past a failing script and exits
# non-zero if any script failed.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODULES=(
  cifar.plot_cifar_noisy_curves
  cifar.plot_cifar_symmetric40_curves
  cifar.clean_cifar_trajectories
  cifar.plot_clean_cifar_barplot
  nlp.plot_agnews_noisy_curves
  nlp.plot_sst2_curves
  nlp_ablation.plot_agnews_noise_ablation_curves
  nlp_ablation.plot_agnews_noise_ablation_barplot
  cifar_ablation.plot_cifar_noise_ablation_curves
  cifar_ablation.plot_cifar_noise_ablation_barplot
  diagnostics.plot_peak_checkpoint_diagnostics
  controllers.plot_arm_selection
  controllers.plot_reward_per_arm
  controllers.plot_update_norm_trajectory
)

fail=0
for m in "${MODULES[@]}"; do
  echo "=== scripts.plots.$m ==="
  if ! uv run python -m "scripts.plots.$m"; then
    echo "FAILED: scripts.plots.$m" >&2
    fail=1
  fi
done

exit "$fail"

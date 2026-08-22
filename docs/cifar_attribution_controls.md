# CIFAR-100 attribution controls

This launcher adds two reward-free, open-loop controls to the existing
`experiments/task_cifar100.py` training path:

- **Uniform random** tests whether AEES gains can be explained by exposing the
  optimizer to arbitrary multiplier changes. Every episode action is an
  independent uniform sample from `0.5, 1.0, 2.0`; the realized counts are not
  forced to be balanced.
- **Frequency matched** tests whether gains can be explained only by AEES's
  marginal preference for each multiplier. For target seed `s`, it pools the
  executed AEES actions from the other four seeds in the same corruption and
  optimizer cell, converts those probabilities to exact integer counts with
  largest-remainder allocation, and shuffles the complete multiset before
  training.

Both controls use the same `StructuredEpisodeManager`, wrapped AdamW/SGD
optimizer, 200-step boundaries, constant base LR, and zero explicit gradient
noise as AEES. Their controller accepts the episode manager's common callback
but ignores its reward argument and stores no reward-dependent or bandit
statistics.

## Required AEES traces

The default trace root is `results/cifar_noisy`. It must contain the five
completed constant-base AEES artifacts for every corruption/optimizer cell:

```text
results/cifar_noisy/
  cifar100_{asym20,sym20,sym40}_seed{0,1,2,3,4}/
    adamw_aees_ep200_lr05102
    sgd_aees_ep200_lr05102_base01
```

The launcher reads `episode_logs.selected_lr_values` as the authoritative
executed action trace. Before using a source it checks seed, optimizer,
corruption, candidate set, 200-step episode length, 200-epoch horizon,
constant schedule, base LR, weight decay, momentum, zero noise, total steps,
trace lengths, action/index agreement, and contiguous episode boundaries.
Missing, duplicate, truncated, or incompatible sources fail before training.

For target seed `s`, the frequency estimate is

```text
q_-s(a) = count of action a across source seeds != s
          / total episode actions across source seeds != s
```

The result records `q_-s`, source seed IDs, absolute source paths, allocated
counts, schedule seed, planned sequence, executed sequence, and the unchanged
model/data/corruption seed. Schedule construction uses a private
`random.Random` instance and cannot advance any training RNG.

## Five-GPU RunPod launch

Run these commands from the repository root. First resolve and validate all 60
runs without starting training:

```bash
uv run python scripts/run_cifar_attribution_controls.py \
  --gpus 0,1,2,3,4 \
  --aees-trace-root results/cifar_noisy \
  --output-root results/cifar_attribution_controls \
  --dry-run
```

The dry run prints all commands, verifies exactly 60 unique run IDs, validates
all AEES inputs, precomputes all schedules, and writes
`results/cifar_attribution_controls/manifest.json`. It does not start training
or create run logs/results.

Launch the production workload later with:

```bash
uv run python scripts/run_cifar_attribution_controls.py \
  --gpus 0,1,2,3,4 \
  --aees-trace-root results/cifar_noisy \
  --output-root results/cifar_attribution_controls
```

GPU 0 receives seed 0, and so on through GPU 4/seed 4. Each GPU runs its 12
jobs sequentially; the five seed workers run concurrently. Every run has an
isolated log under `_logs/`. Results are written atomically, so interruption
cannot expose a partially serialized result at the final path.

Resume with the same command. Resume is enabled by default and skips only
artifacts that pass identity, horizon, and schedule validation. Invalid
existing artifacts are left untouched and require explicit `--overwrite`.
On failure the launcher exits nonzero and prints exact one-run retry commands.
Ctrl-C terminates active child processes cleanly; rerun the production command
to resume pending work.

An optional one-epoch orchestration smoke configuration is available without
changing the validated 200-epoch AEES source requirements:

```bash
uv run python scripts/run_cifar_attribution_controls.py \
  --seeds 0 --gpus 0 \
  --noise-regimes sym20 --optimizers AdamW \
  --policies uniform-random \
  --output-root results/cifar_attribution_smoke \
  --smoke-test --dry-run
```

Remove `--dry-run` only when you intentionally want that one-epoch smoke run.

## Selecting a subset

The selection flags take comma-separated values:

```bash
# One policy, optimizer, regime, and seed on physical GPU 3
uv run python scripts/run_cifar_attribution_controls.py \
  --policies frequency-matched \
  --optimizers SGD \
  --noise-regimes asym20 \
  --seeds 2 --gpus 3 \
  --aees-trace-root results/cifar_noisy \
  --output-root results/cifar_attribution_controls
```

Accepted values are:

- policies: `uniform-random`, `frequency-matched`
- optimizers: `AdamW`, `SGD`
- noise regimes: `asym20`, `sym20`, `sym40`
- seeds: `0,1,2,3,4`

When selecting multiple seeds, provide exactly one GPU ID per selected seed;
mapping is in ascending seed order. `--run-ids` accepts exact IDs printed in
the manifest and is used by the generated retry commands.

## Aggregation

After runs complete:

```bash
uv run python -m scripts.tables.aggregate_cifar_attribution_controls \
  --aees-trace-root results/cifar_noisy \
  --controls-root results/cifar_attribution_controls \
  --out-dir reproduced_artifacts/tables/cifar_attribution_controls
```

This writes a compact CSV and a detailed JSON. Each corruption/optimizer cell
contains mean ± sample SD for peak, final, and best-to-final drop; seed-paired
AEES-minus-control effects; 10,000-resample paired percentile-bootstrap 95%
intervals with RNG seed 0; AEES win counts; and pooled/per-seed action
distributions. Incomplete cells are marked and list their missing run IDs;
paired paper summaries remain null until all five pairs exist.

## Output structure

```text
results/cifar_attribution_controls/
  manifest.json
  _logs/<run-id>.log
  cifar100_<regime>_seed<seed>/
    adamw_uniform_random.json
    adamw_frequency_matched.json
    sgd_uniform_random.json
    sgd_frequency_matched.json
```


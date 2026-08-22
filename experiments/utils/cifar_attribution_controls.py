"""Open-loop controls and AEES trace validation for CIFAR-100 attribution.

The controls in this module deliberately keep schedule randomness in a local
``random.Random`` instance.  They never touch Python's module-level RNG or a
Torch generator, so constructing a schedule cannot perturb model, data,
augmentation, corruption, dropout, or optimizer randomness.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Literal, Sequence


OpenLoopPolicy = Literal["uniform-random", "frequency-matched"]

LR_MULTIPLIERS: tuple[float, ...] = (0.5, 1.0, 2.0)
NOISE_CANDIDATES: tuple[float, ...] = (0.0,)
SOURCE_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
SOURCE_EPOCHS = 200
SOURCE_BATCH_SIZE = 128
SOURCE_EPISODE_LENGTH = 200
SOURCE_WEIGHT_DECAY = 1e-4
SOURCE_MOMENTUM = 0.9
CIFAR100_TRAIN_EXAMPLES = 50_000

NOISE_REGIMES: dict[str, tuple[str, float]] = {
    "asym20": ("asymmetric", 0.2),
    "sym20": ("symmetric", 0.2),
    "sym40": ("symmetric", 0.4),
}

OPTIMIZER_BASE_LRS: dict[str, float] = {
    "AdamW": 1e-3,
    "SGD": 1e-1,
}

AEES_ARTIFACT_NAMES: dict[str, str] = {
    "AdamW": "adamw_aees_ep200_lr05102",
    "SGD": "sgd_aees_ep200_lr05102_base01",
}


class TraceValidationError(ValueError):
    """Raised when an AEES source trace is missing, ambiguous, or incompatible."""


@dataclass(frozen=True)
class ValidatedAeesTrace:
    """The executed LR actions and provenance from one validated AEES artifact."""

    seed: int
    path: Path
    actions: tuple[float, ...]
    total_steps: int


@dataclass(frozen=True)
class OpenLoopPlan:
    """A complete LR schedule generated before optimization starts."""

    policy: OpenLoopPolicy
    schedule_seed: int
    episode_count: int
    lr_indices: tuple[int, ...]
    lr_values: tuple[float, ...]
    probabilities: tuple[float, ...]
    integer_counts: tuple[int, ...]
    source_seeds: tuple[int, ...] = ()
    source_artifacts: tuple[str, ...] = ()

    def to_metadata(self) -> dict[str, object]:
        """Return result/manifest metadata sufficient to reproduce the schedule."""

        metadata: dict[str, object] = {
            "policy": self.policy,
            "reward_free": True,
            "training_feedback_consulted": False,
            "sequence_fully_precomputed": True,
            "schedule_rng": "python.random.Random(local instance)",
            "schedule_seed": self.schedule_seed,
            "episode_count": self.episode_count,
            "lr_candidates": list(LR_MULTIPLIERS),
            "probabilities": {
                str(value): probability
                for value, probability in zip(
                    LR_MULTIPLIERS, self.probabilities, strict=True
                )
            },
            "integer_counts": {
                str(value): count
                for value, count in zip(
                    LR_MULTIPLIERS, self.integer_counts, strict=True
                )
            },
            "source_seeds": list(self.source_seeds),
            "source_artifacts": list(self.source_artifacts),
            "planned_lr_indices": list(self.lr_indices),
            "planned_lr_multipliers": list(self.lr_values),
        }
        if self.policy == "uniform-random":
            metadata["sampling_seed"] = self.schedule_seed
        else:
            metadata["shuffle_seed"] = self.schedule_seed
            metadata["q_minus_target_seed"] = dict(metadata["probabilities"])
        return metadata


class PrecomputedScheduleController:
    """Controller adapter that replays a fixed arm sequence without feedback.

    ``StructuredEpisodeManager`` calls ``update`` through its common controller
    API.  This implementation intentionally ignores the supplied reward and
    retains no reward or bandit statistics.
    """

    def __init__(self, arm_indices: Sequence[int], n_arms: int) -> None:
        if n_arms <= 0:
            raise ValueError("n_arms must be positive.")
        schedule = tuple(int(index) for index in arm_indices)
        if not schedule:
            raise ValueError("A precomputed schedule must contain at least one arm.")
        invalid = [index for index in schedule if not 0 <= index < n_arms]
        if invalid:
            raise ValueError(
                f"Schedule contains invalid arm indices {invalid}; expected [0, {n_arms})."
            )
        self.n_arms = n_arms
        self._schedule = schedule
        self._cursor = 0
        self._update_call_count = 0

    def select_mode(self) -> int:
        """Return the next precomputed arm, failing if training overruns the plan."""

        if self._cursor >= len(self._schedule):
            raise RuntimeError(
                "Precomputed LR schedule exhausted; the training step count does "
                "not match the schedule's episode count."
            )
        selected = self._schedule[self._cursor]
        self._cursor += 1
        return selected

    def update(self, mode_id: int, reward: float) -> None:
        """Accept the manager callback without reading feedback or updating state."""

        if not 0 <= mode_id < self.n_arms:
            raise ValueError(f"mode_id must be in [0, {self.n_arms}), got {mode_id}.")
        # Do not inspect, validate, store, or otherwise use ``reward``.
        self._update_call_count += 1

    def get_state(self) -> dict[str, object]:
        """Return schedule progress only; no reward-dependent statistics exist."""

        return {
            "n_arms": self.n_arms,
            "schedule_length": len(self._schedule),
            "selection_count": self._cursor,
            "feedback_ignored": True,
            "bandit_statistics_updated": False,
            "update_call_count": self._update_call_count,
        }


class FeedbackFreeReward:
    """Reward adapter for open-loop controls that never reads episode feedback."""

    def compute(self, summary: object) -> float:
        """Return a constant placeholder required by the episode-manager API."""

        return 0.0


def episode_count_from_training_config(
    *,
    epochs: int,
    batch_size: int,
    episode_length: int,
    train_examples: int = CIFAR100_TRAIN_EXAMPLES,
) -> int:
    """Resolve episode count from the configured training geometry."""

    if epochs <= 0 or batch_size <= 0 or episode_length <= 0 or train_examples <= 0:
        raise ValueError("Training geometry values must all be positive.")
    steps_per_epoch = math.ceil(train_examples / batch_size)
    return math.ceil((steps_per_epoch * epochs) / episode_length)


def episode_count_from_total_steps(total_steps: int, episode_length: int) -> int:
    """Resolve episode count when the instantiated loader length is known."""

    if total_steps <= 0 or episode_length <= 0:
        raise ValueError("total_steps and episode_length must be positive.")
    return math.ceil(total_steps / episode_length)


def largest_remainder_counts(
    probabilities: Sequence[float],
    total_count: int,
) -> tuple[int, ...]:
    """Allocate integer counts deterministically with Hamilton apportionment."""

    if total_count <= 0:
        raise ValueError("total_count must be positive.")
    values = tuple(float(probability) for probability in probabilities)
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("probabilities must be finite and nonnegative.")
    probability_sum = sum(values)
    if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"probabilities must sum to one, got {probability_sum}.")

    quotas = [value * total_count for value in values]
    counts = [math.floor(quota) for quota in quotas]
    remainder = total_count - sum(counts)
    order = sorted(
        range(len(values)),
        key=lambda index: (-(quotas[index] - counts[index]), index),
    )
    for index in order[:remainder]:
        counts[index] += 1
    if sum(counts) != total_count:
        raise RuntimeError("Largest-remainder allocation did not preserve total count.")
    return tuple(counts)


def build_uniform_random_plan(
    *,
    episode_count: int,
    schedule_seed: int,
) -> OpenLoopPlan:
    """Pre-sample independent uniform episode actions with a private RNG."""

    if episode_count <= 0:
        raise ValueError("episode_count must be positive.")
    rng = random.Random(schedule_seed)
    indices = tuple(rng.randrange(len(LR_MULTIPLIERS)) for _ in range(episode_count))
    counts = Counter(indices)
    return OpenLoopPlan(
        policy="uniform-random",
        schedule_seed=schedule_seed,
        episode_count=episode_count,
        lr_indices=indices,
        lr_values=tuple(LR_MULTIPLIERS[index] for index in indices),
        probabilities=tuple(1.0 / len(LR_MULTIPLIERS) for _ in LR_MULTIPLIERS),
        integer_counts=tuple(counts.get(index, 0) for index in range(len(LR_MULTIPLIERS))),
    )


def build_frequency_matched_plan(
    *,
    trace_root: str | Path,
    noise_regime: str,
    optimizer: str,
    target_seed: int,
    episode_count: int,
    schedule_seed: int,
) -> OpenLoopPlan:
    """Build a shuffled, leave-one-seed-out frequency-matched schedule."""

    source_seeds = tuple(seed for seed in SOURCE_SEEDS if seed != target_seed)
    if len(source_seeds) != 4:
        raise ValueError(
            f"target_seed must be one of {list(SOURCE_SEEDS)}, got {target_seed}."
        )
    traces = [
        load_validated_aees_trace(
            trace_root=trace_root,
            noise_regime=noise_regime,
            optimizer=optimizer,
            seed=source_seed,
        )
        for source_seed in source_seeds
    ]
    action_counts = Counter(action for trace in traces for action in trace.actions)
    total_actions = sum(action_counts.values())
    if total_actions <= 0:
        raise TraceValidationError("Validated AEES traces contain no executed actions.")
    probabilities = tuple(action_counts.get(value, 0) / total_actions for value in LR_MULTIPLIERS)
    if any(value < 0.0 for value in probabilities) or not math.isclose(
        sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise TraceValidationError(
            f"Invalid leave-one-out probabilities for target seed {target_seed}: {probabilities}."
        )
    counts = largest_remainder_counts(probabilities, episode_count)
    indices = [
        index
        for index, count in enumerate(counts)
        for _ in range(count)
    ]
    rng = random.Random(schedule_seed)
    rng.shuffle(indices)
    return OpenLoopPlan(
        policy="frequency-matched",
        schedule_seed=schedule_seed,
        episode_count=episode_count,
        lr_indices=tuple(indices),
        lr_values=tuple(LR_MULTIPLIERS[index] for index in indices),
        probabilities=probabilities,
        integer_counts=counts,
        source_seeds=source_seeds,
        source_artifacts=tuple(str(trace.path.resolve()) for trace in traces),
    )


def load_validated_aees_trace(
    *,
    trace_root: str | Path,
    noise_regime: str,
    optimizer: str,
    seed: int,
) -> ValidatedAeesTrace:
    """Find exactly one source artifact and validate its executed action trace."""

    if noise_regime not in NOISE_REGIMES:
        raise TraceValidationError(
            f"Unknown noise regime {noise_regime!r}; expected {sorted(NOISE_REGIMES)}."
        )
    if optimizer not in AEES_ARTIFACT_NAMES:
        raise TraceValidationError(
            f"Unknown optimizer {optimizer!r}; expected {sorted(AEES_ARTIFACT_NAMES)}."
        )
    if seed not in SOURCE_SEEDS:
        raise TraceValidationError(f"Source seed must be in {list(SOURCE_SEEDS)}, got {seed}.")

    root = Path(trace_root)
    run_dir = root / f"cifar100_{noise_regime}_seed{seed}"
    base_name = AEES_ARTIFACT_NAMES[optimizer]
    candidates = sorted(path for path in run_dir.glob(f"{base_name}*") if path.is_file())
    if not candidates:
        raise TraceValidationError(
            "Missing AEES source trace for "
            f"regime={noise_regime}, optimizer={optimizer}, seed={seed}. "
            f"Expected {run_dir / base_name} (or the same path with .json)."
        )
    if len(candidates) != 1:
        raise TraceValidationError(
            "Duplicate AEES source traces for "
            f"regime={noise_regime}, optimizer={optimizer}, seed={seed}: "
            + ", ".join(str(path) for path in candidates)
        )
    path = candidates[0]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TraceValidationError(f"Could not parse AEES source trace {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TraceValidationError(f"AEES source trace {path} must contain a JSON object.")
    return _validate_aees_payload(
        payload=payload,
        path=path,
        noise_regime=noise_regime,
        optimizer=optimizer,
        seed=seed,
    )


def _validate_aees_payload(
    *,
    payload: dict[str, object],
    path: Path,
    noise_regime: str,
    optimizer: str,
    seed: int,
) -> ValidatedAeesTrace:
    config = payload.get("config")
    episode_logs = payload.get("episode_logs")
    if not isinstance(config, dict) or not isinstance(episode_logs, dict):
        raise TraceValidationError(
            f"AEES source trace {path} is truncated: config and episode_logs are required."
        )

    expected_noise_type, expected_noise_rate = NOISE_REGIMES[noise_regime]
    expected = {
        "control_mode": "adaptive",
        "optimizer": optimizer,
        "label_noise_type": expected_noise_type,
        "label_noise_rate": expected_noise_rate,
        "episode_length": SOURCE_EPISODE_LENGTH,
        "epochs": SOURCE_EPOCHS,
        "batch_size": SOURCE_BATCH_SIZE,
        "lr_scheduler": "none",
        "lr": OPTIMIZER_BASE_LRS[optimizer],
        "weight_decay": SOURCE_WEIGHT_DECAY,
        "lr_candidates": list(LR_MULTIPLIERS),
        "noise_candidates": list(NOISE_CANDIDATES),
    }
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        actual_value = config.get(key)
        if not _equivalent(actual_value, expected_value):
            mismatches.append(f"config.{key}={actual_value!r} (expected {expected_value!r})")
    for location, actual_seed in (("seed", payload.get("seed")), ("config.seed", config.get("seed"))):
        if actual_seed != seed:
            mismatches.append(f"{location}={actual_seed!r} (expected {seed})")
    if payload.get("method_name") != "AdaptiveScheduler":
        mismatches.append(
            f"method_name={payload.get('method_name')!r} (expected 'AdaptiveScheduler')"
        )
    if payload.get("total_epochs") != SOURCE_EPOCHS:
        mismatches.append(
            f"total_epochs={payload.get('total_epochs')!r} (expected {SOURCE_EPOCHS})"
        )
    if optimizer == "SGD" and not _equivalent(config.get("momentum"), SOURCE_MOMENTUM):
        mismatches.append(
            f"config.momentum={config.get('momentum')!r} (expected {SOURCE_MOMENTUM})"
        )
    if mismatches:
        raise TraceValidationError(
            f"AEES source trace {path} has incompatible configuration: "
            + "; ".join(mismatches)
        )

    actions_raw = episode_logs.get("selected_lr_values")
    indices_raw = episode_logs.get("selected_lr_indices")
    start_steps = episode_logs.get("episode_start_steps")
    end_steps = episode_logs.get("episode_end_steps")
    rewards = episode_logs.get("episode_rewards")
    required_lists = {
        "selected_lr_values": actions_raw,
        "selected_lr_indices": indices_raw,
        "episode_start_steps": start_steps,
        "episode_end_steps": end_steps,
        "episode_rewards": rewards,
    }
    invalid_fields = [key for key, value in required_lists.items() if not isinstance(value, list)]
    if invalid_fields:
        raise TraceValidationError(
            f"AEES source trace {path} is truncated: expected list fields {invalid_fields}."
        )

    total_steps = payload.get("total_steps")
    if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps <= 0:
        raise TraceValidationError(f"AEES source trace {path} has invalid total_steps={total_steps!r}.")
    expected_total_steps = math.ceil(CIFAR100_TRAIN_EXAMPLES / SOURCE_BATCH_SIZE) * SOURCE_EPOCHS
    if total_steps != expected_total_steps:
        raise TraceValidationError(
            f"AEES source trace {path} has total_steps={total_steps}; expected {expected_total_steps}."
        )
    expected_episodes = episode_count_from_total_steps(total_steps, SOURCE_EPISODE_LENGTH)
    lengths = {key: len(value) for key, value in required_lists.items() if isinstance(value, list)}
    if set(lengths.values()) != {expected_episodes}:
        raise TraceValidationError(
            f"AEES source trace {path} is missing or truncated: expected {expected_episodes} "
            f"episodes in every action/boundary field, got {lengths}."
        )

    actions = tuple(float(value) for value in actions_raw)
    invalid_actions = [value for value in actions if value not in LR_MULTIPLIERS]
    if invalid_actions:
        raise TraceValidationError(
            f"AEES source trace {path} contains actions outside {list(LR_MULTIPLIERS)}: "
            f"{invalid_actions[:5]}."
        )
    expected_indices = [LR_MULTIPLIERS.index(value) for value in actions]
    if list(indices_raw) != expected_indices:
        raise TraceValidationError(
            f"AEES source trace {path} has inconsistent LR action values and indices."
        )
    expected_starts = [index * SOURCE_EPISODE_LENGTH for index in range(expected_episodes)]
    expected_ends = [min(start + SOURCE_EPISODE_LENGTH - 1, total_steps - 1) for start in expected_starts]
    if start_steps != expected_starts or end_steps != expected_ends:
        raise TraceValidationError(
            f"AEES source trace {path} has duplicate, missing, or non-contiguous episode boundaries."
        )
    return ValidatedAeesTrace(seed=seed, path=path, actions=actions, total_steps=total_steps)


def _equivalent(actual: object, expected: object) -> bool:
    """Compare configuration scalars/lists with strict-enough float tolerance."""

    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isclose(
            float(actual), expected, rel_tol=0.0, abs_tol=1e-12
        )
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return False
        return all(_equivalent(left, right) for left, right in zip(actual, expected, strict=True))
    return actual == expected

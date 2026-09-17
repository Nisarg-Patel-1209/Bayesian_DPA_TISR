"""Trial generation for the three sequence conditions.

CONDITION 1 (same_finger):      F F F F, D D D D, J J J J, K K K K
CONDITION 2 (one_per_finger):   a small, hand-picked pool of permutations of
                                 all four characters (e.g. F D J K, K J D F, ...)
CONDITION 3 (randomized):       a permutation of all four characters drawn from
                                 the FULL 24-permutation space, so it is a
                                 methodologically distinct source of order
                                 variability from condition 2's curated pool.

Every one_per_finger / randomized trial uses each of F/D/J/K exactly once by
construction, and same_finger trials are assigned round-robin across the four
characters, so character frequency is balanced across the whole run without
any extra bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import random

from .config import TrialConfig

SAME_FINGER = "same_finger"
ONE_PER_FINGER = "one_per_finger"
RANDOMIZED = "randomized"
CONDITIONS = (SAME_FINGER, ONE_PER_FINGER, RANDOMIZED)

PRACTICE = "practice"
MAIN = "main"


@dataclass(frozen=True)
class Trial:
    trial_index: int
    block: str  # "practice" | "main"
    condition: str  # one of CONDITIONS
    sequence: tuple[str, ...]

    @property
    def sequence_text(self) -> str:
        return " ".join(self.sequence)

    @property
    def sequence_compact(self) -> str:
        return "".join(self.sequence)


def build_trial_plan(
    characters: tuple[str, ...],
    trial_cfg: TrialConfig,
    rng: random.Random,
) -> tuple[list[Trial], list[Trial]]:
    """Return ``(practice_trials, main_trials)``, each already order-randomized
    (interleaved across conditions, per the paradigm's randomization request)."""
    practice = _build_block(
        characters, trial_cfg, rng, block=PRACTICE, n_per_condition=trial_cfg.practice_trials_per_condition
    )
    main = _build_block(
        characters, trial_cfg, rng, block=MAIN, n_per_condition=trial_cfg.trials_per_condition
    )
    return practice, main


def _build_block(
    characters: tuple[str, ...],
    trial_cfg: TrialConfig,
    rng: random.Random,
    block: str,
    n_per_condition: int,
) -> list[Trial]:
    if n_per_condition <= 0:
        return []

    same_finger_seqs = generate_same_finger_sequences(characters, n_per_condition, trial_cfg.sequence_length, rng)
    one_per_finger_seqs = generate_pool_sequences(
        parse_pool(trial_cfg.one_per_finger_pool, characters), n_per_condition, rng
    )
    randomized_seqs = generate_randomized_sequences(characters, n_per_condition, rng)

    planned: list[tuple[str, tuple[str, ...]]] = (
        [(SAME_FINGER, seq) for seq in same_finger_seqs]
        + [(ONE_PER_FINGER, seq) for seq in one_per_finger_seqs]
        + [(RANDOMIZED, seq) for seq in randomized_seqs]
    )

    if trial_cfg.shuffle:
        rng.shuffle(planned)

    return [
        Trial(trial_index=idx, block=block, condition=condition, sequence=seq)
        for idx, (condition, seq) in enumerate(planned, start=1)
    ]


def generate_same_finger_sequences(
    characters: tuple[str, ...], n: int, sequence_length: int, rng: random.Random
) -> list[tuple[str, ...]]:
    letters = balanced_choices(characters, n, rng)
    return [tuple([letter] * sequence_length) for letter in letters]


def generate_randomized_sequences(characters: tuple[str, ...], n: int, rng: random.Random) -> list[tuple[str, ...]]:
    full_pool = list(itertools.permutations(characters))
    return generate_pool_sequences(full_pool, n, rng)


def generate_pool_sequences(pool: list[tuple[str, ...]], n: int, rng: random.Random) -> list[tuple[str, ...]]:
    """Draw ``n`` sequences from ``pool`` by cycling shuffled copies of it, so
    that within any single pass through the pool nothing repeats, and no
    sequence is ever immediately followed by itself across a reshuffle
    boundary. When ``n <= len(pool)`` this guarantees zero duplicates for the
    whole run, matching the "no accidental duplicate sequences" requirement."""
    if not pool:
        raise ValueError("Sequence pool must not be empty.")

    result: list[tuple[str, ...]] = []
    last = None
    while len(result) < n:
        shuffled = list(pool)
        rng.shuffle(shuffled)
        if last is not None and shuffled[0] == last and len(shuffled) > 1:
            # Avoid a same-sequence seam at the reshuffle boundary.
            swap_at = rng.randrange(1, len(shuffled))
            shuffled[0], shuffled[swap_at] = shuffled[swap_at], shuffled[0]
        result.extend(shuffled)
        last = shuffled[-1]
    return result[:n]


def balanced_choices(items: tuple[str, ...], n: int, rng: random.Random) -> list[str]:
    """Assign ``n`` picks across ``items`` as evenly as possible; the leftover
    ``n % len(items)`` picks are drawn without replacement so no item is
    systematically favoured, then the whole list is shuffled."""
    k = len(items)
    assigned = list(items) * (n // k)
    remainder = n % k
    if remainder:
        assigned.extend(rng.sample(items, remainder))
    rng.shuffle(assigned)
    return assigned


def parse_pool(pool_strings: tuple[str, ...], characters: tuple[str, ...]) -> list[tuple[str, ...]]:
    """Validate the configured one_per_finger pool: every entry must be a
    permutation of exactly ``characters`` (no repeats, no unknown letters)."""
    expected = set(characters)
    parsed: list[tuple[str, ...]] = []
    for entry in pool_strings:
        letters = tuple(entry.upper())
        if set(letters) != expected or len(letters) != len(characters):
            raise ValueError(
                f"one_per_finger_pool entry {entry!r} is not a permutation of {characters}."
            )
        parsed.append(letters)
    return parsed

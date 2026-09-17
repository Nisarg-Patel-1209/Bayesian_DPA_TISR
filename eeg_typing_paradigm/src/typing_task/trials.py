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
    (interleaved across conditions, per the paradigm's randomization request).

    The main block's ``total_trials`` need not be divisible by 3 (e.g. a
    140-trial session): counts are split across the three conditions as
    evenly as possible (46/47/47), with ``rng`` deciding which condition(s)
    absorb the remainder so the choice is still seed-reproducible."""
    practice = _build_block(
        characters,
        trial_cfg,
        rng,
        block=PRACTICE,
        condition_counts=dict.fromkeys(CONDITIONS, trial_cfg.practice_trials_per_condition),
    )
    main_counts = resolve_main_condition_counts(trial_cfg.total_trials, trial_cfg.sequence_length, rng)
    main = _build_block(characters, trial_cfg, rng, block=MAIN, condition_counts=main_counts)
    return practice, main


def resolve_main_condition_counts(
    total_trials: int, sequence_length: int, rng: random.Random
) -> dict[str, int]:
    """Split ``total_trials`` across the three conditions, biased to keep
    character frequency balanced even when ``total_trials`` isn't a multiple
    of 3.

    one_per_finger / randomized trials each contribute exactly one of every
    character regardless of how many trials they get, so any split between
    them is character-neutral. same_finger trials contribute
    ``sequence_length`` copies of a single character per trial, so its count
    is snapped to the nearest multiple of ``sequence_length`` (e.g. 48, not
    46, out of 140) -- that alone gets full character balance from
    same_finger, and the remaining trials split evenly between the other two.
    """
    ideal = total_trials / 3
    nearest_multiple = round(ideal / sequence_length) * sequence_length
    same_finger_n = min(max(nearest_multiple, 0), total_trials)

    remaining = total_trials - same_finger_n
    one_per_finger_n, randomized_n = split_evenly(remaining, 2, rng)
    if rng.random() < 0.5:  # avoid systematically favouring one condition with the odd trial
        one_per_finger_n, randomized_n = randomized_n, one_per_finger_n

    return {SAME_FINGER: same_finger_n, ONE_PER_FINGER: one_per_finger_n, RANDOMIZED: randomized_n}


def _build_block(
    characters: tuple[str, ...],
    trial_cfg: TrialConfig,
    rng: random.Random,
    block: str,
    condition_counts: dict[str, int],
) -> list[Trial]:
    if all(n <= 0 for n in condition_counts.values()):
        return []

    same_finger_seqs = generate_same_finger_sequences(
        characters, condition_counts[SAME_FINGER], trial_cfg.sequence_length, rng
    )
    one_per_finger_seqs = generate_pool_sequences(
        parse_pool(trial_cfg.one_per_finger_pool, characters), condition_counts[ONE_PER_FINGER], rng
    )
    randomized_seqs = generate_randomized_sequences(characters, condition_counts[RANDOMIZED], rng)

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


def split_evenly(total: int, k: int, rng: random.Random) -> list[int]:
    """Split ``total`` into ``k`` non-negative counts as evenly as possible;
    any remainder is handed to a random subset of ``k`` (reproducible via
    ``rng``) so no one condition is systematically favoured across studies
    that don't divide evenly by ``k`` (e.g. 140 trials / 3 conditions)."""
    if total < 0:
        raise ValueError("total must be non-negative.")
    base, remainder = divmod(total, k)
    counts = [base] * k
    for i in rng.sample(range(k), remainder):
        counts[i] += 1
    return counts


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

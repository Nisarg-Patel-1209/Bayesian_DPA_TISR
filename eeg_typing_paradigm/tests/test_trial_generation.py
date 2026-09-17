from __future__ import annotations

from collections import Counter
import itertools
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from typing_task.config import load_config
from typing_task.trials import (
    ONE_PER_FINGER,
    RANDOMIZED,
    SAME_FINGER,
    balanced_choices,
    build_trial_plan,
    generate_pool_sequences,
    generate_randomized_sequences,
    generate_same_finger_sequences,
    parse_pool,
)

CHARACTERS = ("F", "D", "J", "K")


class ConfigTests(unittest.TestCase):
    def test_demo_config_loads(self) -> None:
        cfg = load_config(ROOT / "config" / "demo.toml")
        self.assertEqual(cfg.keys.characters, CHARACTERS)
        self.assertEqual(cfg.trials.sequence_length, 4)
        self.assertGreater(cfg.trials.trials_per_condition, 0)


class BalancedChoicesTests(unittest.TestCase):
    def test_exact_multiple_is_perfectly_balanced(self) -> None:
        rng = random.Random(0)
        picks = balanced_choices(CHARACTERS, 8, rng)
        self.assertEqual(Counter(picks), Counter({c: 2 for c in CHARACTERS}))

    def test_remainder_is_still_within_one(self) -> None:
        rng = random.Random(0)
        picks = balanced_choices(CHARACTERS, 9, rng)
        counts = Counter(picks)
        self.assertEqual(sum(counts.values()), 9)
        self.assertTrue(all(v in (2, 3) for v in counts.values()))


class PoolSequenceTests(unittest.TestCase):
    def test_no_duplicates_within_one_pass(self) -> None:
        rng = random.Random(1)
        pool = list(itertools.permutations(CHARACTERS))
        seqs = generate_pool_sequences(pool, len(pool), rng)
        self.assertEqual(len(set(seqs)), len(pool))

    def test_no_adjacent_repeat_across_reshuffle_boundary(self) -> None:
        rng = random.Random(2)
        pool = [("A",), ("B",)]
        seqs = generate_pool_sequences(pool, 20, rng)
        for a, b in zip(seqs, seqs[1:]):
            self.assertNotEqual(a, b)


class SameFingerTests(unittest.TestCase):
    def test_each_sequence_is_one_character_repeated(self) -> None:
        rng = random.Random(3)
        seqs = generate_same_finger_sequences(CHARACTERS, 8, 4, rng)
        for seq in seqs:
            self.assertEqual(len(set(seq)), 1)
            self.assertEqual(len(seq), 4)

    def test_characters_balanced_across_trials(self) -> None:
        rng = random.Random(3)
        seqs = generate_same_finger_sequences(CHARACTERS, 8, 4, rng)
        counts = Counter(seq[0] for seq in seqs)
        self.assertEqual(counts, Counter({c: 2 for c in CHARACTERS}))


class RandomizedTests(unittest.TestCase):
    def test_every_sequence_uses_all_four_characters_once(self) -> None:
        rng = random.Random(4)
        seqs = generate_randomized_sequences(CHARACTERS, 10, rng)
        for seq in seqs:
            self.assertEqual(set(seq), set(CHARACTERS))
            self.assertEqual(len(seq), 4)

    def test_no_duplicates_when_within_full_permutation_space(self) -> None:
        rng = random.Random(4)
        seqs = generate_randomized_sequences(CHARACTERS, 24, rng)
        self.assertEqual(len(set(seqs)), 24)


class PoolValidationTests(unittest.TestCase):
    def test_valid_pool_parses(self) -> None:
        parsed = parse_pool(("FDJK", "KJDF"), CHARACTERS)
        self.assertEqual(parsed, [("F", "D", "J", "K"), ("K", "J", "D", "F")])

    def test_pool_entry_with_repeated_letter_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_pool(("FFJK",), CHARACTERS)

    def test_pool_entry_with_unknown_letter_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_pool(("FDJX",), CHARACTERS)


class TrialPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config(ROOT / "config" / "demo.toml")

    def test_condition_counts_match_config(self) -> None:
        rng = random.Random(self.cfg.experiment.random_seed)
        practice, main = build_trial_plan(self.cfg.keys.characters, self.cfg.trials, rng)
        n = self.cfg.trials.trials_per_condition
        self.assertEqual(len(main), n * 3)
        counts = Counter(t.condition for t in main)
        self.assertEqual(counts, Counter({SAME_FINGER: n, ONE_PER_FINGER: n, RANDOMIZED: n}))

    def test_condition_order_is_interleaved_not_blocked(self) -> None:
        rng = random.Random(self.cfg.experiment.random_seed)
        _, main = build_trial_plan(self.cfg.keys.characters, self.cfg.trials, rng)
        condition_sequence = [t.condition for t in main]
        self.assertGreater(len(set(condition_sequence)), 1)
        # A fully blocked order would put every same_finger trial first.
        self.assertNotEqual(condition_sequence[: self.cfg.trials.trials_per_condition], [SAME_FINGER] * self.cfg.trials.trials_per_condition)

    def test_trial_indices_are_contiguous(self) -> None:
        rng = random.Random(self.cfg.experiment.random_seed)
        _, main = build_trial_plan(self.cfg.keys.characters, self.cfg.trials, rng)
        self.assertEqual([t.trial_index for t in main], list(range(1, len(main) + 1)))

    def test_same_seed_reproduces_same_plan(self) -> None:
        rng_a = random.Random(self.cfg.experiment.random_seed)
        rng_b = random.Random(self.cfg.experiment.random_seed)
        _, main_a = build_trial_plan(self.cfg.keys.characters, self.cfg.trials, rng_a)
        _, main_b = build_trial_plan(self.cfg.keys.characters, self.cfg.trials, rng_b)
        self.assertEqual([t.sequence for t in main_a], [t.sequence for t in main_b])

    def test_character_frequency_is_balanced_across_whole_run(self) -> None:
        rng = random.Random(self.cfg.experiment.random_seed)
        _, main = build_trial_plan(self.cfg.keys.characters, self.cfg.trials, rng)
        char_counts = Counter(ch for t in main for ch in t.sequence)
        counts = list(char_counts.values())
        self.assertLessEqual(max(counts) - min(counts), 1)


class CsvSchemaTests(unittest.TestCase):
    def test_fieldnames_scale_with_sequence_length(self) -> None:
        sys.path.insert(0, str(ROOT / "src"))
        from typing_task.runner import csv_fieldnames

        fields = csv_fieldnames(4)
        for i in range(1, 5):
            self.assertIn(f"key{i}_expected", fields)
            self.assertIn(f"key{i}_actual", fields)
            self.assertIn(f"key{i}_time_sec", fields)
        for i in range(1, 4):
            self.assertIn(f"iki_{i}_{i + 1}_sec", fields)
        self.assertIn("sequence_correct", fields)


if __name__ == "__main__":
    unittest.main()

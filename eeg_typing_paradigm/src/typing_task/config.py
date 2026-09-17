"""TOML-driven configuration for the EEG typing-sequence pilot.

All experimenter-tunable numbers (timings, trial counts, key mapping) live in
one `config/*.toml` file and are loaded into frozen dataclasses here, so the
task script itself never needs to be edited to change parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    participant: str
    session: str
    random_seed: int | None


@dataclass(frozen=True)
class TimingConfig:
    fixation_duration_sec: float
    sequence_cue_duration_sec: float
    preparation_duration_sec: float
    go_duration_sec: float
    rest_duration_sec: float
    # Safety net only: the trial otherwise ends purely on keypress count, with
    # no imposed inter-key interval (per the paradigm's "natural typing" rule).
    typing_timeout_sec: float


@dataclass(frozen=True)
class TrialConfig:
    # Total trials in the main block, split as evenly as possible across the
    # three conditions (need not be a multiple of 3).
    total_trials: int
    practice_trials_per_condition: int
    sequence_length: int
    # Curated example sequences for the one-letter-per-finger condition
    # (each uses all four characters exactly once, in a fixed hand-picked order).
    one_per_finger_pool: tuple[str, ...]
    shuffle: bool
    # A break screen is inserted after every this-many main-block trials
    # (e.g. 35 -> breaks after trials 35/70/105 of a 140-trial session).
    # 0 or None disables breaks.
    break_every_n_trials: int | None = None


@dataclass(frozen=True)
class KeyConfig:
    characters: tuple[str, ...]
    finger_map: dict[str, str]
    quit_key: str


@dataclass(frozen=True)
class DisplayConfig:
    fullscreen: bool
    screen_size: tuple[int, int]
    monitor_name: str
    background_color: tuple[float, float, float]
    text_color: tuple[float, float, float]
    units: str
    screen: int = 0


@dataclass(frozen=True)
class EegConfig:
    enabled: bool
    lsl_stream_name: str
    source_id: str


@dataclass(frozen=True)
class OutputConfig:
    data_dir: Path
    log_dir: Path


@dataclass(frozen=True)
class TypingTaskConfig:
    config_path: Path
    experiment: ExperimentConfig
    timing: TimingConfig
    trials: TrialConfig
    keys: KeyConfig
    display: DisplayConfig
    eeg: EegConfig
    output: OutputConfig


def load_config(path: str | Path) -> TypingTaskConfig:
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)

    experiment = ExperimentConfig(**raw["experiment"])
    timing = TimingConfig(**raw["timing"])

    trials_raw = dict(raw["trials"])
    trials_raw["one_per_finger_pool"] = tuple(trials_raw["one_per_finger_pool"])
    trials = TrialConfig(**trials_raw)

    keys_raw = dict(raw["keys"])
    keys_raw["characters"] = tuple(c.upper() for c in keys_raw["characters"])
    keys = KeyConfig(**keys_raw)

    display_raw = dict(raw["display"])
    display_raw["screen_size"] = tuple(display_raw["screen_size"])
    display_raw["background_color"] = tuple(display_raw["background_color"])
    display_raw["text_color"] = tuple(display_raw["text_color"])
    display = DisplayConfig(**display_raw)

    eeg = EegConfig(**raw["eeg"])

    output_raw = dict(raw["output"])
    output_raw["data_dir"] = Path(output_raw["data_dir"])
    output_raw["log_dir"] = Path(output_raw["log_dir"])
    output = OutputConfig(**output_raw)

    return TypingTaskConfig(
        config_path=path,
        experiment=experiment,
        timing=timing,
        trials=trials,
        keys=keys,
        display=display,
        eeg=eeg,
        output=output,
    )


def as_nested_dict(cfg: TypingTaskConfig) -> dict[str, Any]:
    from dataclasses import asdict

    out = asdict(cfg)
    out.pop("config_path", None)
    return out

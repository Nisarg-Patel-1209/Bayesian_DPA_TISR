"""PsychoPy runner for the EEG typing-sequence pilot.

Per-trial flow (see config/demo.toml for the timing values):

    fixation (+)  ->  sequence cue (e.g. "F D J K")  ->  PREPARE (blank sequence)
    ->  GO  ->  natural typing (no imposed inter-key interval, one safety
    timeout) ->  rest (+)

PsychoPy is imported lazily inside the functions that need it so that trial
generation, config loading, and the CSV schema can be unit-tested (and the
``--dry-run`` preview used) on a machine without PsychoPy installed.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import TypingTaskConfig
from .trials import ONE_PER_FINGER, RANDOMIZED, Trial, build_trial_plan
from .triggers import (
    BREAK_END,
    BREAK_ONSET,
    FIXATION_ONSET,
    GO_ONSET,
    PREPARATION_ONSET,
    REST_ONSET,
    SEQUENCE_CUE_ONSET,
    TYPING_COMPLETE,
    key_marker,
    make_marker_sender,
    trial_end_marker,
    trial_start_marker,
)

INSTRUCTIONS_TEXT = """\
FINGER MAPPING

LEFT HAND                          RIGHT HAND
F  =  LEFT INDEX                   J  =  RIGHT INDEX
D  =  LEFT MIDDLE                  K  =  RIGHT MIDDLE

You will see a short sequence of letters on the screen.
Read the entire sequence and remember its order.

During the preparation period, mentally prepare the complete sequence.

When GO appears, type the entire sequence naturally.
Do not wait for another GO cue between letters.
Do not intentionally slow down or speed up between letters.
Use the assigned finger for each character.

After completing the sequence, relax during the rest period.
Try to avoid unnecessary movements during fixation and preparation.

Press any key to begin the practice trials.
"""

PRACTICE_DONE_TEXT = """\
Practice complete.

The real trials work exactly the same way.
Press any key to begin.
"""


def run_task(cfg: TypingTaskConfig, practice_only: bool = False) -> dict[str, Path]:
    try:
        from psychopy import core, visual
    except ImportError as exc:  # pragma: no cover - depends on GUI environment
        raise RuntimeError(
            "PsychoPy is not installed in this Python environment. "
            "Install it with `pip install -r requirements.txt`."
        ) from exc

    cfg.output.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.output.log_dir.mkdir(parents=True, exist_ok=True)

    import random

    rng = random.Random(cfg.experiment.random_seed)
    practice_trials, main_trials = build_trial_plan(cfg.keys.characters, cfg.trials, rng)

    eeg_enabled = cfg.eeg.enabled
    markers = make_marker_sender(
        enabled=eeg_enabled, stream_name=cfg.eeg.lsl_stream_name, source_id=cfg.eeg.source_id
    )

    win = visual.Window(
        size=cfg.display.screen_size,
        fullscr=cfg.display.fullscreen,
        monitor=cfg.display.monitor_name,
        units=cfg.display.units,
        color=cfg.display.background_color,
        screen=cfg.display.screen,
        allowGUI=not cfg.display.fullscreen,
    )
    text_kwargs = dict(color=cfg.display.text_color, wrapWidth=1.6)
    fixation = visual.TextStim(win, text="+", height=0.12, **text_kwargs)
    sequence_stim = visual.TextStim(win, text="", height=0.12, **text_kwargs)
    prepare_stim = visual.TextStim(win, text="PREPARE", height=0.09, **text_kwargs)
    go_stim = visual.TextStim(win, text="GO", height=0.14, **text_kwargs)
    message_stim = visual.TextStim(win, text="", height=0.06, **text_kwargs)

    output_paths: dict[str, Path] = {}
    try:
        show_message(win, message_stim, INSTRUCTIONS_TEXT)

        quit_requested = False
        if practice_trials:
            # Practice is short (per-condition, not the full session) -- no breaks needed.
            practice_rows, quit_requested = run_block(
                win, fixation, sequence_stim, prepare_stim, go_stim, message_stim, markers, cfg, practice_trials
            )
            output_paths["practice"] = write_rows(
                make_output_path(cfg, suffix="practice"), practice_rows, cfg.trials.sequence_length
            )
            if practice_only or quit_requested:
                show_message(win, message_stim, "Practice complete. Ending here.")
                return output_paths
            show_message(win, message_stim, PRACTICE_DONE_TEXT)

        if not quit_requested:
            main_rows, _ = run_block(
                win,
                fixation,
                sequence_stim,
                prepare_stim,
                go_stim,
                message_stim,
                markers,
                cfg,
                main_trials,
                break_every_n_trials=cfg.trials.break_every_n_trials,
            )
            output_paths["main"] = write_rows(
                make_output_path(cfg, suffix="main"), main_rows, cfg.trials.sequence_length
            )
            show_finish(win, message_stim, main_rows)
    finally:
        markers.close()
        win.close()

    return output_paths


def run_block(
    win: Any,
    fixation: Any,
    sequence_stim: Any,
    prepare_stim: Any,
    go_stim: Any,
    message_stim: Any,
    markers: Any,
    cfg: TypingTaskConfig,
    trials: list[Trial],
    break_every_n_trials: int | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Run every trial in order; stop early (without dropping the row already
    collected) if the participant presses the quit key. Returns
    ``(rows, quit_requested)``.

    If ``break_every_n_trials`` is set, a self-paced break screen is shown
    after every that-many completed trials (but never after the last trial,
    since the block is simply over at that point)."""
    total = len(trials)
    rows: list[dict[str, Any]] = []
    for trial in trials:
        row = run_trial(win, fixation, sequence_stim, prepare_stim, go_stim, markers, cfg, trial)
        rows.append(row)
        if row["quit"]:
            return rows, True
        if (
            break_every_n_trials
            and trial.trial_index % break_every_n_trials == 0
            and trial.trial_index < total
        ):
            show_break(win, message_stim, markers, trial.trial_index, total)
    return rows, False


def show_break(win: Any, message_stim: Any, markers: Any, completed: int, total: int) -> None:
    from psychopy import event

    markers.send(BREAK_ONSET)
    text = (
        f"Break\n\n{completed} / {total} trials complete.\n\n"
        "Relax for a bit.\nPress any key when you are ready to continue."
    )
    message_stim.text = text
    message_stim.draw()
    win.flip()
    event.waitKeys()
    markers.send(BREAK_END)


def run_trial(
    win: Any,
    fixation: Any,
    sequence_stim: Any,
    prepare_stim: Any,
    go_stim: Any,
    markers: Any,
    cfg: TypingTaskConfig,
    trial: Trial,
) -> dict[str, Any]:
    from psychopy import core, event

    timing = cfg.timing
    characters = set(cfg.keys.characters)
    key_list = [c.lower() for c in cfg.keys.characters] + [cfg.keys.quit_key]

    markers.send(trial_start_marker(trial.trial_index, trial.condition, trial.sequence_compact))

    # --- Fixation --------------------------------------------------------
    fixation.draw()
    win.callOnFlip(markers.send, FIXATION_ONSET)
    fixation_onset = win.flip()
    core.wait(timing.fixation_duration_sec)

    # --- Sequence cue: participant reads and memorizes, does NOT type ----
    sequence_stim.text = trial.sequence_text
    sequence_stim.draw()
    win.callOnFlip(markers.send, SEQUENCE_CUE_ONSET)
    sequence_cue_onset = win.flip()
    event.clearEvents(eventType="keyboard")  # discard any early keys during memorization
    core.wait(timing.sequence_cue_duration_sec)

    # --- Preparation: sequence hidden, no countdown, no new cues ----------
    prepare_stim.draw()
    win.callOnFlip(markers.send, PREPARATION_ONSET)
    preparation_onset = win.flip()
    event.clearEvents(eventType="keyboard")
    core.wait(timing.preparation_duration_sec)

    # --- GO: exactly one cue for the whole sequence -----------------------
    go_stim.draw()
    go_clock = core.Clock()
    win.callOnFlip(event.clearEvents, eventType="keyboard")
    win.callOnFlip(go_clock.reset)
    win.callOnFlip(markers.send, GO_ONSET)
    go_onset = win.flip()
    core.wait(timing.go_duration_sec)
    win.flip()  # clear GO; nothing else is shown during typing (no on-screen feedback)

    # --- Natural typing: accept only the four assigned characters ---------
    expected = trial.sequence
    n_expected = len(expected)
    keypresses: list[tuple[str, float]] = []  # (character, time-since-GO)
    quit_requested = False
    timed_out = False

    while len(keypresses) < n_expected:
        if go_clock.getTime() > timing.typing_timeout_sec:
            timed_out = True
            break
        for key_name, rt_sec in event.getKeys(keyList=key_list, timeStamped=go_clock):
            if key_name == cfg.keys.quit_key:
                quit_requested = True
                break
            character = key_name.upper()
            if character in characters:
                keypresses.append((character, rt_sec))
                markers.send(key_marker(character))
        if quit_requested:
            break

    typing_complete_time = go_clock.getTime()
    if len(keypresses) == n_expected:
        markers.send(TYPING_COMPLETE)

    # --- Rest --------------------------------------------------------------
    fixation.draw()
    win.callOnFlip(markers.send, REST_ONSET)
    rest_onset = win.flip()
    if not quit_requested:
        core.wait(timing.rest_duration_sec)
    markers.send(trial_end_marker(trial.trial_index))

    return build_row(
        trial=trial,
        fixation_onset=fixation_onset,
        sequence_cue_onset=sequence_cue_onset,
        preparation_onset=preparation_onset,
        go_onset=go_onset,
        keypresses=keypresses,
        typing_complete_time=typing_complete_time,
        rest_onset=rest_onset,
        timed_out=timed_out,
        quit_requested=quit_requested,
    )


def build_row(
    trial: Trial,
    fixation_onset: float,
    sequence_cue_onset: float,
    preparation_onset: float,
    go_onset: float,
    keypresses: list[tuple[str, float]],
    typing_complete_time: float,
    rest_onset: float,
    timed_out: bool,
    quit_requested: bool,
) -> dict[str, Any]:
    n_expected = len(trial.sequence)
    row: dict[str, Any] = {
        "trial_index": trial.trial_index,
        "block": trial.block,
        "condition": trial.condition,
        "sequence": trial.sequence_text,
        "sequence_length": n_expected,
        "fixation_onset": f"{fixation_onset:.6f}",
        "sequence_cue_onset": f"{sequence_cue_onset:.6f}",
        "preparation_onset": f"{preparation_onset:.6f}",
        "go_onset": f"{go_onset:.6f}",
    }

    n_correct = 0
    prev_time = None
    for position in range(1, n_expected + 1):
        expected_char = trial.sequence[position - 1]
        if position <= len(keypresses):
            actual_char, rt = keypresses[position - 1]
            correct = int(actual_char == expected_char)
            n_correct += correct
            row[f"key{position}_expected"] = expected_char
            row[f"key{position}_actual"] = actual_char
            row[f"key{position}_time_sec"] = f"{rt:.6f}"
            row[f"key{position}_correct"] = correct
            if prev_time is not None and position > 1:
                row[f"iki_{position - 1}_{position}_sec"] = f"{rt - prev_time:.6f}"
            prev_time = rt
        else:
            row[f"key{position}_expected"] = expected_char
            row[f"key{position}_actual"] = ""
            row[f"key{position}_time_sec"] = ""
            row[f"key{position}_correct"] = 0
            if position > 1:
                row[f"iki_{position - 1}_{position}_sec"] = ""

    row["rt_go_to_first_key_sec"] = f"{keypresses[0][1]:.6f}" if keypresses else ""
    row["typing_complete_time_sec"] = f"{typing_complete_time:.6f}"
    row["rest_onset"] = f"{rest_onset:.6f}"
    row["n_correct"] = n_correct
    row["sequence_correct"] = int(n_correct == n_expected and len(keypresses) == n_expected)
    row["timed_out"] = int(timed_out)
    row["quit"] = int(quit_requested)
    return row


def csv_fieldnames(sequence_length: int) -> list[str]:
    fields = [
        "trial_index",
        "block",
        "condition",
        "sequence",
        "sequence_length",
        "fixation_onset",
        "sequence_cue_onset",
        "preparation_onset",
        "go_onset",
    ]
    for position in range(1, sequence_length + 1):
        fields += [
            f"key{position}_expected",
            f"key{position}_actual",
            f"key{position}_time_sec",
            f"key{position}_correct",
        ]
        if position > 1:
            fields.append(f"iki_{position - 1}_{position}_sec")
    fields += [
        "rt_go_to_first_key_sec",
        "typing_complete_time_sec",
        "rest_onset",
        "n_correct",
        "sequence_correct",
        "timed_out",
        "quit",
    ]
    return fields


def show_message(win: Any, message_stim: Any, text: str) -> None:
    from psychopy import event

    message_stim.text = text
    message_stim.draw()
    win.flip()
    event.waitKeys()


def show_finish(win: Any, message_stim: Any, rows: list[dict[str, Any]]) -> None:
    from psychopy import core

    completed = [row for row in rows if not row["quit"]]
    if completed:
        n_correct = sum(row["sequence_correct"] for row in completed)
        percent = round((n_correct / len(completed)) * 100)
        text = f"Done\n\nSequence accuracy: {percent}%\n{n_correct} / {len(completed)} trials fully correct"
    else:
        text = "Done"
    message_stim.text = text
    message_stim.draw()
    win.flip()
    core.wait(1.5)


def make_output_path(cfg: TypingTaskConfig, suffix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    participant = safe_name(cfg.experiment.participant)
    session = safe_name(cfg.experiment.session)
    return cfg.output.data_dir / f"{cfg.experiment.name}_{participant}_ses-{session}_{timestamp}_{suffix}.csv"


def write_rows(path: Path, rows: list[dict[str, Any]], sequence_length: int) -> Path:
    fieldnames = csv_fieldnames(sequence_length)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def dry_run(cfg: TypingTaskConfig, max_preview: int = 12) -> tuple[list[Trial], list[Trial]]:
    import random
    from collections import Counter

    rng = random.Random(cfg.experiment.random_seed)
    practice_trials, main_trials = build_trial_plan(cfg.keys.characters, cfg.trials, rng)

    print(f"Loaded {cfg.config_path}")
    print(f"Participant/session: {cfg.experiment.participant} / {cfg.experiment.session}")
    print(f"Characters: {cfg.keys.characters}  (finger map: {cfg.keys.finger_map})")
    print(
        f"Timing (s): fixation={cfg.timing.fixation_duration_sec} "
        f"cue={cfg.timing.sequence_cue_duration_sec} "
        f"prep={cfg.timing.preparation_duration_sec} "
        f"go={cfg.timing.go_duration_sec} "
        f"rest={cfg.timing.rest_duration_sec} "
        f"typing_timeout={cfg.timing.typing_timeout_sec}"
    )
    print(f"Practice trials: {len(practice_trials)}   Main trials: {len(main_trials)}")
    counts = Counter(t.condition for t in main_trials)
    print(f"Main condition counts: {dict(counts)}")
    if cfg.trials.break_every_n_trials:
        n = cfg.trials.break_every_n_trials
        break_points = list(range(n, len(main_trials), n))
        print(f"Rest breaks after trials: {break_points}")
    char_counts = Counter(ch for t in main_trials for ch in t.sequence)
    print(f"Main character counts: {dict(char_counts)}")
    duplicates = _find_unexpected_duplicate_sequences(main_trials, cfg.trials)
    if duplicates:
        print(f"WARNING: unexpected duplicate sequences: {duplicates}")
    else:
        print("No unexpected duplicate sequences (same_finger repeats are expected).")
    if cfg.eeg.enabled:
        print(f"EEG markers: LSL stream '{cfg.eeg.lsl_stream_name}'")
    print("")
    for trial in main_trials[:max_preview]:
        print(f"{trial.trial_index:03d} {trial.block:9s} {trial.condition:15s} {trial.sequence_text}")
    if len(main_trials) > max_preview:
        print(f"... {len(main_trials) - max_preview} more")
    return practice_trials, main_trials


def _find_unexpected_duplicate_sequences(trials: list[Trial], trial_cfg) -> dict[str, list[str]]:
    """Flag duplicate sequences only where the paradigm asks us to avoid them.

    same_finger trials are *supposed* to repeat one character across all four
    positions, and can legitimately repeat the same sequence text across
    trials (e.g. two "F F F F" trials) once every character has been used at
    least once -- that's not a bug. one_per_finger / randomized duplicates are
    only unexpected while the trial count for that condition is within the
    size of its sequence pool (5 curated orders, or all 24 permutations)."""
    import math
    from collections import defaultdict

    by_condition: dict[str, list[str]] = defaultdict(list)
    for trial in trials:
        by_condition[trial.condition].append(trial.sequence_text)

    pool_sizes = {
        ONE_PER_FINGER: len(trial_cfg.one_per_finger_pool),
        RANDOMIZED: math.factorial(trial_cfg.sequence_length),
    }

    duplicates = {}
    for condition in (ONE_PER_FINGER, RANDOMIZED):
        seqs = by_condition.get(condition, [])
        if len(seqs) > pool_sizes[condition]:
            continue  # duplicates are unavoidable once the pool is exhausted
        seen: set[str] = set()
        dupes: set[str] = set()
        for s in seqs:
            if s in seen:
                dupes.add(s)
            seen.add(s)
        if dupes:
            duplicates[condition] = sorted(dupes)
    return duplicates


def safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value.strip())
    return safe or "unknown"

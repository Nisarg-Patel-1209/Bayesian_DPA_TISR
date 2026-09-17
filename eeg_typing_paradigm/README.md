# EEG Typing-Sequence Pilot

A PsychoPy demo for an EEG recording pilot investigating what information
about a planned typing sequence is represented in the EEG **before** and
**during** natural typing. Four keys, one per finger:

| Key | Finger |
|-----|--------|
| F | left index |
| D | left middle |
| J | right index |
| K | right middle |

This is a **pilot/demo**, not the final experiment: the goal is to check that
the trial structure, instructions, and timings are practical and comfortable
to run while recording EEG. The default profile is a full 140-trial session
(split ~48/46/46 across the three conditions, see below) plus 3 practice
trials, with a self-paced rest break after every 35 main trials (after
trials 35/70/105) so the participant can relax; all of these numbers are set
in `config/demo.toml` and easy to shrink back down for a quicker test run.

Built with the same architecture as the `Flanker_EEG_RL` project supplied
alongside this one (TOML-config dataclasses, a lazily-imported PsychoPy
runner, an LSL `MarkerSender` that degrades to a no-op when no EEG hardware
is present, and a `--dry-run` trial-plan preview), adapted for this
different paradigm.

## Project layout

```
eeg_typing_paradigm/
  config/demo.toml          all tunable parameters (see below)
  src/typing_task/
    config.py                TOML -> dataclasses
    trials.py                sequence/condition generation
    triggers.py               LSL event markers (+ no-op fallback)
    runner.py                 PsychoPy trial loop, CSV writer
    run.py                    CLI entry point
  scripts/run_typing_task.sh
  tests/test_trial_generation.py   unit tests (no PsychoPy needed)
  data/                      CSV output (git-ignored)
  logs/                      reserved for future use
```

## Running the demo

```bash
cd eeg_typing_paradigm
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
./scripts/run_typing_task.sh --dry-run                       # preview the trial plan, no GUI
./scripts/run_typing_task.sh --practice-only --participant p01
./scripts/run_typing_task.sh --participant p01 --session 001
```

Useful flags: `--show-config` (print the resolved TOML as JSON),
`--marker-test` / `--marker-listen` / `--list-streams` (verify the LSL
marker pipeline without PsychoPy, once `[eeg].enabled = true` in the config
and `pylsl` is installed).

Run the unit tests (trial generation, balance, config loading — no PsychoPy
or display required):

```bash
python3 -m unittest discover -s tests -v
```

## Experiment flow

Every trial, regardless of condition, follows the same six phases:

| Phase | Duration | What's shown | Marker |
|---|---|---|---|
| Fixation | 0.5 s | `+` | `FIXATION_ONSET` |
| Sequence cue | 1.5 s | the full sequence, e.g. `F D J K` — read, do not type | `SEQUENCE_CUE_ONSET` |
| Preparation | 2.0 s | `PREPARE` (sequence hidden, no countdown) | `PREPARATION_ONSET` |
| GO | 0.2 s | `GO` (only cue for the whole sequence) | `GO_ONSET` |
| Natural typing | until 4 keys accepted (or `typing_timeout_sec`) | blank — no on-screen feedback, no keyboard shown | `KEY_F`/`KEY_D`/`KEY_J`/`KEY_K` per accepted keypress, then `TYPING_COMPLETE` |
| Rest | 2.0 s | `+` | `REST_ONSET` |

All six durations, plus trial counts, the participant ID, and the random
seed, are set in `config/demo.toml` — nothing is hard-coded in the script.

### Rest breaks

Every `break_every_n_trials` completed main trials (default 35, giving 3
breaks across a 140-trial session — after trials 35, 70, 105, but not after
the last trial, since the session simply ends there), a self-paced break
screen is shown:

```
Break

35 / 140 trials complete.

Relax for a bit.
Press any key when you are ready to continue.
```

There's no imposed minimum or maximum break length — the participant
continues whenever they press a key. `BREAK_ONSET` / `BREAK_END` markers
bracket the break so it's easy to exclude from EEG analysis (or to look at
separately, e.g. for a post-break warm-up effect on the first few trials).
Practice trials don't include breaks (there are only a few of them).

### The three conditions

1. **`same_finger`** — `F F F F`, `D D D D`, `J J J J`, `K K K K`. Same
   finger + same character four times. Control condition for basic
   motor/finger-related activity.
2. **`one_per_finger`** — each trial uses all four characters, one per
   finger, drawn from a small curated pool (`F D J K`, `K J D F`, `F J D K`,
   `D F K J`, `J K F D` by default, configurable).
3. **`randomized`** — also all four characters, no repeats, but drawn from
   the **full 24-permutation space** of F/D/J/K rather than the curated pool.

Conditions 2 and 3 are deliberately generated from two different pools (a
small hand-picked list vs. the complete permutation space) even though both
satisfy "each finger once, order varies" — see the docstring in
`src/typing_task/trials.py`. If you don't need that distinction, set
`one_per_finger_pool` to more entries, or treat both as one condition in
analysis. Note that at the full 140-trial session size, each condition gets
~46-48 trials — more than either pool (5 curated orders, or 24 permutations),
so sequences necessarily repeat in both conditions; the sampler still
guarantees no sequence is *immediately* repeated back-to-back within a
condition (see `generate_pool_sequences`), it just can no longer guarantee
zero repeats across the whole session the way the original ≤10-trial demo
profile could.

`total_trials` (140 by default) is split across the three conditions as
evenly as possible, with one adjustment: since `same_finger` trials
contribute `sequence_length` (4) copies of one character each, its count is
snapped to the nearest multiple of 4 (48, not the "even" 46-47) so that
character frequency comes out perfectly balanced overall (140 F / D / J / K
each, out of 560 total keypresses) rather than merely close. The other two
conditions split the remainder (46/46) since they contribute one of every
character per trial regardless of their own trial count. Trial order is then
randomized and interleaved across all three conditions (not blocked), with a
fixed `random_seed` so a run is exactly reproducible. The realized order is
itself saved in the CSV (`trial_index` + `sequence` column), so no separate
trial-plan file is needed to reconstruct it.

### Handling incorrect keypresses

No feedback is shown during typing (per the spec), so the participant has no
way to know they hit the wrong key. Requiring an *exact* match before
advancing would risk a trial silently stalling until the timeout — worse for
the participant than a wrong keypress. Instead: **any** of the four assigned
keys (right or wrong for that position) is accepted and consumes one
sequence position; the expected and actual character are both logged per
position, and `sequence_correct` / `key{n}_correct` flag the outcome after
the fact. Keys outside {F, D, J, K, escape} are ignored entirely (not logged,
not consumed) since they can't have been an attempt at the sequence.

## Participant instructions (shown verbatim before the practice block)

```
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
```

No mention is made of expected neural activity, to avoid biasing the EEG.

## Output files

Each run writes up to two CSVs to `data/`:
`eeg_typing_demo_<participant>_ses-<session>_<timestamp>_practice.csv` and
`..._main.csv`. Columns (for `sequence_length = 4`; the schema scales
automatically if you change that):

| Column | Meaning |
|---|---|
| `trial_index`, `block`, `condition`, `sequence`, `sequence_length` | trial identity |
| `fixation_onset`, `sequence_cue_onset`, `preparation_onset`, `go_onset` | phase onset timestamps (PsychoPy clock, seconds) |
| `key{1..4}_expected` / `_actual` / `_time_sec` / `_correct` | per-position expected vs. typed character, time since GO, correctness |
| `iki_{1_2, 2_3, 3_4}_sec` | inter-key intervals between successive accepted keypresses |
| `rt_go_to_first_key_sec` | reaction time from GO onset to the first accepted keypress |
| `typing_complete_time_sec` | time (since GO) the typing phase ended |
| `rest_onset` | rest-phase onset timestamp |
| `n_correct`, `sequence_correct` | trial scoring |
| `timed_out`, `quit` | whether the safety timeout or escape ended the trial |

All timestamps use PsychoPy's high-resolution monotonic clock (`win.flip()` /
`core.Clock()`), never `time.time()` or ordinary Python timers, so
within-session latencies (RT, IKI, phase durations) are sub-millisecond
accurate regardless of what the OS clock is doing.

## Aligning timestamps to EEG

The CSV timestamps are all in PsychoPy's own clock domain — great for
behavioral QA (verifying phase durations, RT, IKI) but **not** directly
comparable to your amplifier's clock. Alignment instead goes through LSL:

1. Set `[eeg].enabled = true` in the config (requires `pylsl`). The task then
   opens an LSL outlet named `TypingTaskMarkers` and pushes one of
   `FIXATION_ONSET` / `SEQUENCE_CUE_ONSET` / `PREPARATION_ONSET` / `GO_ONSET`
   / `KEY_F..K` / `TYPING_COMPLETE` / `REST_ONSET` (plus a
   `TRIAL_START/{index}/{condition}/{sequence}` marker at the top of each
   trial, and `BREAK_ONSET` / `BREAK_END` around each rest break) at the
   exact moment each phase's stimulus is flipped to screen.
2. Record that marker stream **and** your EEG amplifier's LSL stream into the
   same session with LabRecorder (or your amp's LSL recorder) → one XDF file.
3. Because both streams are timestamped from LSL's synchronized clock (not
   the stimulus PC's wall clock or PsychoPy's clock), the marker timestamps
   in the XDF file line up with the EEG samples with no manual offset
   correction — epoch directly on them in your analysis pipeline (e.g. MNE).

If no EEG hardware/LSL is available, the task still runs identically; markers
are simply dropped (`NullMarkerSender`), which is exactly the mode this repo
was tested in.

## Methodological notes and open issues

- **"Correct" keypress ambiguity in `same_finger` trials.** Because all four
  expected characters are identical, a keypress on the wrong finger at any
  position is scored as `key{n}_correct = 0` even though it's the same key
  physically pressed four times — worth deciding up front whether you care
  about position-level accuracy here at all, or only sequence-level (whether
  they typed 4 of the assigned letter, in any position ordering — which is
  moot when all 4 letters are identical).
- **1.5 s to read + memorize a 4-character sequence** is workable for
  the fixed pools (2 and 3), but for `randomized` trials specifically it's
  worth watching whether participants report needing more time — 4 items is
  within typical working-memory span, but combined with immediate motor
  translation it's more demanding than pure recall.
- **2 s preparation with no countdown** means the participant alone decides
  when they're "ready," but GO always fires at a fixed 2 s regardless —
  watch for anticipatory EMG/motor-prep artifacts building right at the
  boundary if participants start to predict the GO timing (it's fixed, not
  jittered). If early ERPs during preparation look contaminated by
  anticipation, consider jittering `preparation_duration_sec` slightly
  trial-to-trial in the next iteration.
- **8 s safety timeout** is generous for a 4-keypress sequence typed
  naturally; if a participant is still typing at 8 s something is already
  wrong (confusion, missed GO, motor difficulty) — treat `timed_out=1` rows
  as a signal to check in with the participant, not just data to exclude.
- **No feedback during or after each trial** (as specified) means typing
  errors go uncorrected within a trial; over many trials this could let
  errors compound if a participant develops a wrong habit for one sequence
  in the curated `one_per_finger_pool`. The end-of-run accuracy summary is
  the only feedback given, deliberately, to avoid biasing the EEG — reconsider
  if pilot accuracy comes back low.

## What to watch for during the demo, to tune the timings

- **Sequence cue (1.5 s):** do participants report needing to re-read the
  sequence, or do they clearly finish reading well before it disappears? Too
  short risks a memory-encoding failure that looks like a `sequence_correct
  = 0` error; too long adds dead time to every trial.
- **Preparation (2 s):** ask afterward whether participants felt "ready and
  waiting" or "still preparing" when GO appeared. If consistently the
  latter, lengthen it; if consistently the former for a while before GO,
  either shorten it or jitter it to avoid anticipation.
- **Natural typing:** look at the `iki_*_sec` distribution — is there a
  consistent floor (e.g. everyone's fastest IKI is ~150 ms, consistent with
  normal typing) or occasional very long gaps (confusion about the sequence,
  motor hesitation)? Also check `rt_go_to_first_key_sec` for outliers
  suggesting the GO cue was missed or ambiguous at only 0.2 s.
- **Rest (2 s):** confirm participants are actually relaxed and not still
  finishing a keypress or thinking about the next trial — if `EMG`/movement
  artifacts show up here, either the typing phase is bleeding into rest
  (increase the gap) or 2 s is enough and can even be shortened.

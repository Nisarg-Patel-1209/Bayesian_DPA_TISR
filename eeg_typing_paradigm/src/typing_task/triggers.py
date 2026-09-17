"""Event markers for EEG, sent over Lab Streaming Layer (LSL).

A ``MarkerSender`` pushes string markers onto an LSL ``StreamOutlet`` that a
recorder (e.g. LabRecorder) captures alongside the EEG into an XDF file, so
the continuous EEG can later be epoched on these markers. If ``pylsl`` is not
installed, or EEG is disabled in the config, a no-op sender is used instead so
the task still runs (and can be timed/piloted) without any EEG hardware.

Marker names are exactly the ones requested for this paradigm, each a plain
string (no parsing needed downstream):

    FIXATION_ONSET
    SEQUENCE_CUE_ONSET
    PREPARATION_ONSET
    GO_ONSET
    KEY_F / KEY_D / KEY_J / KEY_K   (one per accepted keypress)
    TYPING_COMPLETE
    REST_ONSET

Trial boundaries additionally carry the trial index/condition/sequence so the
marker stream alone (even without the CSV) is enough to identify each trial:

    TRIAL_START/{index}/{condition}/{sequence}
    TRIAL_END/{index}
"""

from __future__ import annotations

from typing import Optional, Protocol

FIXATION_ONSET = "FIXATION_ONSET"
SEQUENCE_CUE_ONSET = "SEQUENCE_CUE_ONSET"
PREPARATION_ONSET = "PREPARATION_ONSET"
GO_ONSET = "GO_ONSET"
TYPING_COMPLETE = "TYPING_COMPLETE"
REST_ONSET = "REST_ONSET"


class MarkerSender(Protocol):
    def send(self, marker: str, timestamp: Optional[float] = None) -> None: ...
    def close(self) -> None: ...


class NullMarkerSender:
    """Discards markers. Used when EEG is disabled or LSL is not installed."""

    enabled = False

    def send(self, marker: str, timestamp: Optional[float] = None) -> None:
        return None

    def close(self) -> None:
        return None


class LslMarkerSender:
    """Pushes string markers onto an LSL outlet (stream type ``Markers``)."""

    enabled = True

    def __init__(self, stream_name: str = "TypingTaskMarkers", source_id: str = "typing_task") -> None:
        from pylsl import StreamInfo, StreamOutlet  # imported lazily

        info = StreamInfo(
            name=stream_name,
            type="Markers",
            channel_count=1,
            nominal_srate=0,  # irregular event stream
            channel_format="string",
            source_id=source_id,
        )
        self._outlet = StreamOutlet(info)

    def send(self, marker: str, timestamp: Optional[float] = None) -> None:
        if timestamp is None:
            self._outlet.push_sample([marker])
        else:
            self._outlet.push_sample([marker], timestamp)

    def close(self) -> None:
        self._outlet = None


def make_marker_sender(
    enabled: bool, stream_name: str = "TypingTaskMarkers", source_id: str = "typing_task"
) -> MarkerSender:
    """Return an LSL sender if ``enabled`` and ``pylsl`` is importable, else a
    no-op sender. Never raises for a missing dependency so the task still runs."""
    if not enabled:
        return NullMarkerSender()
    try:
        return LslMarkerSender(stream_name=stream_name, source_id=source_id)
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        print(f"[triggers] LSL unavailable ({exc}); EEG markers disabled.", flush=True)
        return NullMarkerSender()


def key_marker(character: str) -> str:
    return f"KEY_{character.upper()}"


def trial_start_marker(index: int, condition: str, sequence_compact: str) -> str:
    return f"TRIAL_START/{index}/{condition}/{sequence_compact}"


def trial_end_marker(index: int) -> str:
    return f"TRIAL_END/{index}"


def run_marker_test(
    stream_name: str = "TypingTaskMarkers",
    source_id: str = "marker_test",
    repeats: int = 5,
    interval_sec: float = 0.4,
) -> None:
    """Emit an example trial's worth of markers over LSL to verify the pipeline
    with the EEG amplifier connected -- no PsychoPy, no experiment. Start your
    LSL recorder first; you should see a 'TypingTaskMarkers' stream and these
    markers landing alongside the EEG. Ctrl-C to stop early."""
    import time

    try:
        from pylsl import StreamInfo, StreamOutlet  # noqa: F401  (import check)
    except ImportError:
        print("pylsl is not installed. Install it with:\n    pip install pylsl")
        return

    sender = LslMarkerSender(stream_name=stream_name, source_id=source_id)
    print(f"Opened LSL marker stream '{stream_name}' (type=Markers).")
    print("Start your LSL recorder now (record this stream + the EEG).")
    print("Sending an example trial's markers... Ctrl-C to stop.\n")
    sequence = [
        trial_start_marker(1, "one_per_finger", "FDJK"),
        FIXATION_ONSET,
        SEQUENCE_CUE_ONSET,
        PREPARATION_ONSET,
        GO_ONSET,
        key_marker("F"),
        key_marker("D"),
        key_marker("J"),
        key_marker("K"),
        TYPING_COMPLETE,
        REST_ONSET,
        trial_end_marker(1),
    ]
    time.sleep(1.0)  # let consumers connect before the first sample
    try:
        for _ in range(repeats):
            for marker in sequence:
                sender.send(marker)
                print(f"  sent: {marker}")
                time.sleep(interval_sec)
    except KeyboardInterrupt:
        print("\nStopped.")
    print("\nDone. Stop the recording; the markers should be time-aligned with the EEG.")


def list_lsl_streams(wait_time: float = 2.0) -> None:
    """Print every LSL stream currently on the network."""
    try:
        from pylsl import resolve_streams
    except ImportError:
        print("pylsl is not installed. Install it with:\n    pip install pylsl")
        return
    streams = resolve_streams(wait_time=wait_time)
    if not streams:
        print("No LSL streams found.")
        return
    print(f"Found {len(streams)} LSL stream(s):")
    for s in streams:
        print(f"  name={s.name()!r}  type={s.type()!r}  channels={s.channel_count()}  rate={s.nominal_srate()} Hz")


def run_marker_listener(stream_name: str = "TypingTaskMarkers", resolve_timeout_sec: float = 10.0) -> None:
    """Connect to the task's LSL marker stream and print markers as they arrive.
    Run in a second terminal while the task (or ``--marker-test``) is running."""
    try:
        from pylsl import StreamInlet, local_clock, resolve_byprop
    except ImportError:
        print("pylsl is not installed. Install it with:\n    pip install pylsl")
        return

    print(f"Resolving LSL stream name='{stream_name}' (up to {resolve_timeout_sec:.0f}s)...")
    streams = resolve_byprop("name", stream_name, 1, timeout=resolve_timeout_sec)
    if not streams:
        print(f"No '{stream_name}' stream found. Is the task or --marker-test running on the network?")
        return
    inlet = StreamInlet(streams[0])
    print(f"Connected to '{stream_name}'. Printing markers (Ctrl-C to stop)...\n")
    t0 = local_clock()
    n = 0
    try:
        while True:
            sample, timestamp = inlet.pull_sample(timeout=1.0)
            if sample:
                n += 1
                print(f"  +{timestamp - t0:7.3f}s  {sample[0]}")
    except KeyboardInterrupt:
        print(f"\nStopped after {n} markers.")

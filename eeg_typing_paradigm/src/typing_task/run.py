from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import as_nested_dict, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the EEG typing-sequence pilot (PsychoPy).")
    parser.add_argument("--config", default="config/demo.toml", help="Path to a TOML config file.")
    parser.add_argument("--participant", help="Override experiment.participant.")
    parser.add_argument("--session", help="Override experiment.session.")
    parser.add_argument("--practice-only", action="store_true", help="Run only the practice block, then exit.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate config and print the trial plan without opening PsychoPy."
    )
    parser.add_argument("--marker-test", action="store_true", help="Emit example LSL markers (no PsychoPy).")
    parser.add_argument("--marker-listen", action="store_true", help="Print markers from the LSL stream in real time.")
    parser.add_argument("--list-streams", action="store_true", help="List all LSL streams on the network.")
    parser.add_argument("--show-config", action="store_true", help="Print the resolved config as JSON and exit.")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    if args.participant or args.session:
        cfg = _with_cli_overrides(cfg, participant=args.participant, session=args.session)

    if args.show_config:
        print(json.dumps(as_nested_dict(cfg), indent=2, default=str))
        return
    if args.dry_run:
        from .runner import dry_run

        dry_run(cfg)
        return
    if args.marker_test:
        from .triggers import run_marker_test

        run_marker_test(stream_name=cfg.eeg.lsl_stream_name, source_id=cfg.eeg.source_id)
        return
    if args.marker_listen:
        from .triggers import run_marker_listener

        run_marker_listener(stream_name=cfg.eeg.lsl_stream_name)
        return
    if args.list_streams:
        from .triggers import list_lsl_streams

        list_lsl_streams()
        return

    from .runner import run_task

    output_paths = run_task(cfg, practice_only=args.practice_only)
    for label, path in output_paths.items():
        print(f"Wrote {label} data: {path}")


def _with_cli_overrides(cfg, participant: str | None, session: str | None):
    from dataclasses import replace

    experiment = replace(
        cfg.experiment,
        participant=participant or cfg.experiment.participant,
        session=session or cfg.experiment.session,
    )
    return replace(cfg, experiment=experiment)


if __name__ == "__main__":
    main()

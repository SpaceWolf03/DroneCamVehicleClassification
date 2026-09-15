#!/usr/bin/env python3
"""Extract the embedded DJI subtitle telemetry for later spatial grounding.

This does not claim to georeference vehicles yet. It preserves the original
per-frame subtitle stream and makes it easy to inspect alongside trajectories.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/telemetry.srt"))
    args = parser.parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(f"Video not found: {args.source}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(args.source), "-map", "0:s:0", str(args.output)],
        check=True,
    )
    print(f"Telemetry subtitle stream written to {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate tracking exports from a completed run without processing video."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def validate_run(run_dir: Path) -> bool:
    detections_path = run_dir / "detections.csv"
    tracks_path = run_dir / "tracks.csv"
    if not detections_path.is_file() or not tracks_path.is_file():
        raise FileNotFoundError(f"Expected detections.csv and tracks.csv in {run_dir}")

    classes_by_id: dict[str, set[str]] = defaultdict(set)
    frames_by_id: dict[str, list[int]] = defaultdict(list)
    detection_count = 0
    with detections_path.open(newline="", encoding="utf-8") as detections_file:
        for row in csv.DictReader(detections_file):
            detection_count += 1
            track_id = row["track_id"]
            classes_by_id[track_id].add(row["class_name"])
            frames_by_id[track_id].append(int(row["frame"]))

    track_count = 0
    with tracks_path.open(newline="", encoding="utf-8") as tracks_file:
        for _ in csv.DictReader(tracks_file):
            track_count += 1

    unstable_classes = {
        track_id: sorted(classes)
        for track_id, classes in classes_by_id.items()
        if len(classes) > 1
    }
    invalid_frame_order = [
        track_id for track_id, frames in frames_by_id.items()
        if frames != sorted(frames) or len(frames) == 0
    ]
    missing_track_rows = sorted(set(classes_by_id) - {
        row["track_id"]
        for row in csv.DictReader(tracks_path.open(newline="", encoding="utf-8"))
    })

    passed = bool(detection_count and track_count) and not unstable_classes and not invalid_frame_order and not missing_track_rows
    print(f"run: {run_dir}")
    print(f"detections: {detection_count}")
    print(f"tracks: {track_count}")
    print(f"stable_classes: {'PASS' if not unstable_classes else 'FAIL'}")
    print(f"ordered_trajectories: {'PASS' if not invalid_frame_order else 'FAIL'}")
    print(f"track_summary_rows: {'PASS' if not missing_track_rows else 'FAIL'}")
    if unstable_classes:
        print(f"class_changes: {unstable_classes}")
    if invalid_frame_order:
        print(f"invalid_frame_order: {invalid_frame_order}")
    if missing_track_rows:
        print(f"missing_track_rows: {missing_track_rows}")
    print(f"tracking_validation: {'PASS' if passed else 'FAIL'}")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Completed run directory containing detections.csv and tracks.csv")
    args = parser.parse_args()
    return 0 if validate_run(args.run_dir) else 1


if __name__ == "__main__":
    raise SystemExit(main())

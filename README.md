# FlytBase Traffic Analysis

This repository begins with the Level 1 foundation: detect road users, keep a stable ID with ByteTrack, and export trajectories in a format that later levels can use for movement counts, speeds, lane assignment, and map grounding.

## Current capability

`track_traffic.py` uses an Ultralytics YOLO detector and ByteTrack to detect
these COCO classes:

- `pedestrian`, `car`, `motorcycle`, `bus`, `truck`

It produces an annotated video, `detections.csv` (one observation for each
object/frame), `tracks.csv` (one row per trajectory), and `run.json`.

The label shown for a track is locked when that ID first appears. This prevents
per-frame detector jitter (such as `truck` briefly changing to `bus`) from
making the video unreadable. `detections.csv` preserves both that stable class
(`class_name`) and the detector's per-frame opinion (`raw_class_name`).

The annotated video displays an object only after three observations. This
suppresses one-frame false positives while retaining every raw detection in
`detections.csv`. The default `traffic_bytetrack.yaml` profile also keeps a
lost ID available for roughly three seconds to improve recovery after short
occlusions.

For vehicle tracks, a second-stage CLIP classifier samples several crops of
the same ID and assigns one consensus mode: `car`, `LGV`, `HGV`, `bus`, or
`truck`. The detector's original category remains in `raw_class_name`; the
consensus result and confidence appear in `class_name` and `class_consensus`.

## Run a short validation clip first

Open `run_level1.py` in VS Code and edit only its SETTINGS block. Then use
**Run Python File**. It defaults to ten seconds from the busy intersection
(300 frames at ~29.97 fps) and writes results to `outputs/latest/`.

Review `outputs/latest/Intersection_Merged_from_003000/annotated.mp4`. In particular,
check whether a vehicle keeps its ID while it enters, crosses, and exits the
junction. Do not process a full 4K video until that sample looks credible.

If your local GPU is available, the default `--device auto` uses GPU 0. If it
is not, the script makes the CPU fallback explicit. A system with the NVIDIA
driver working should report a GPU for:

```bash
python gpu_test.py
```

## Useful iterations

To preserve a comparison, change `OUTPUT_ROOT` in `run_level1.py`, for example
to `Path("outputs/experiment_02")`. Change `START_FRAME` to inspect another
part of the video. Set `RUN_FULL_VIDEO = True` only after review has passed.

## Telemetry preservation

The videos contain an embedded DJI subtitle stream. Preserve it now for Level
4 work; this is not yet a camera-to-ground calibration:

```bash
python extract_telemetry.py --source Intersection_Merged.MP4 --output outputs/intersection_telemetry.srt
```

## Next milestones

1. Validate detector recall and ID stability on several short clips.
2. Add road/lane polygons and derive directional counts, turns, and queues.
3. Parse telemetry and calibrate the pixel-to-ground transform for real speeds
   and map-native trajectories.

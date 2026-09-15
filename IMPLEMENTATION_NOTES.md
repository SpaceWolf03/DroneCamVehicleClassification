# Traffic Tracking Implementation Notes

## Purpose

The project processes fixed aerial traffic videos, detects road users, maintains track IDs across frames, and exports an annotated video plus CSV trajectory data.

## Video separation

The two sources are configured independently in `run_level1.py`:

- **Video 1:** `Intersection_Merged.MP4`
- **Video 2:** `Multi_Road_Merged.MP4`

Each video has its own frame range, inference size, confidence threshold, display threshold, road mask, classifier settings, and output folder. Every run receives a serial number and timestamp, for example:

```text
outputs/video_1_intersection_merged/run_001_YYYYMMDD_HHMMSS/
outputs/video_2_multi_road_merged/run_001_YYYYMMDD_HHMMSS/
```

`VIDEO` in `run_level1.py` selects which source is processed. The current default is Video 2 for the ongoing multi-road work.

## Detection model and classes

The primary detector is the Ultralytics YOLO model `yolo11m.pt`. Detection is restricted to the relevant COCO road-user classes:

- pedestrian
- car
- motorcycle
- bus
- truck

The detector runs at a configurable image size and confidence threshold. Video 2 uses a larger image size and lower confidence threshold for small aerial vehicles.

## Road masking

Road masks are defined independently for both fixed camera views as normalized coordinates. They are converted into binary pixel masks before processing.

Video 1 uses a densely hand-traced intersection boundary passed through the spline mask path. The spline sampler clamps coordinates to the frame bounds. Video 2 retains its separate multi-road mask.

The binary mask is applied to the inference frame before YOLO and ByteTrack. Objects are accepted only when their bottom-center footpoint is inside the road region:

```text
foot_x = (x1 + x2) / 2
foot_y = y2
```

This is more suitable than bounding-box overlap or box-center testing for vehicles near road edges. The output video remains visually normal; when enabled, it displays only `ROAD MASK ACTIVE`.

## Tracking and class stability

Ultralytics ByteTrack is configured through `traffic_bytetrack.yaml`. The track buffer is intentionally very large so normal track expiry does not occur during a video. The first accepted detector class is locked to the track ID, preventing per-frame class flicker.

The CSV preserves both the stable class and the detector's raw per-frame class. A severe detection failure can still cause a genuinely new ID; unlimited buffering cannot recover an object that is no longer detected.

## Scooter correction

Video 1 can use the optional CLIP vehicle-mode classifier from `vehicle_classifier.py`. It samples repeated crops for a track and aggregates evidence across frames. Pedestrian detections are eligible for guarded reclassification only in Video 1; a pedestrian becomes `motorcycle` only when repeated classifier evidence is sufficiently strong. The resulting class is then stable for the track.

The classifier uses `openai/clip-vit-base-patch32` through Transformers and supports car, LGV, HGV, bus, truck, and motorcycle modes. Its weights are downloaded on first use.

## Small-object detection settings

Video 2 is configured with SOD-oriented settings: higher inference resolution and a lower confidence threshold. SAHI is declared in `requirements.txt` and requested for Video 2. Install it in the active virtual environment with:

```bash
python -m pip install "sahi>=0.11.15"
```

The current tracking loop still calls Ultralytics `model.track()` directly. SAHI availability is checked and missing SAHI produces an explicit installation error; full sliced-prediction integration into the ByteTrack input remains a future enhancement.

## Outputs

Each completed run writes:

- `annotated.mp4`: annotated video
- `detections.csv`: frame-level detections and stable/raw class information
- `tracks.csv`: one summary row per track
- `run.json`: source, dimensions, frame counts, model, thresholds, device, and run metadata

## Main execution path

- `run_level1.py` selects the video and supplies its independent parameters.
- `track_traffic.py` opens the video, builds the road mask, runs YOLO and ByteTrack, applies footpoint filtering, manages class stability, and writes outputs.
- `traffic_bytetrack.yaml` defines ByteTrack behavior.
- `vehicle_classifier.py` provides optional track-level vehicle-mode classification.

Run the configured source with:

```bash
source .venv/bin/activate
python run_level1.py
```

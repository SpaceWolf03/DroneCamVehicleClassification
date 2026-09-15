#!/usr/bin/env python3
"""Level 1 traffic detection and tracking for FlytBase drone footage.

The script writes an annotated MP4 plus two useful exports:
  detections.csv  -- one stable track observation per detected object and frame
  tracks.csv      -- one compact row per track, suitable for later analytics

Start with a short clip. Once the IDs look stable, remove --max-frames to
process an entire source video.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# COCO class IDs that are relevant to road traffic. LGV/HGV are assigned by
# the optional second-stage vehicle classifier, not by COCO itself.
ROAD_CLASSES = {0: "pedestrian", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
MODE_CLASS_IDS = {**ROAD_CLASSES, 8: "LGV", 9: "HGV"}
VEHICLE_DETECTOR_IDS = {2, 3, 5, 7}
ANNOTATION_COLOR = (255, 255, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Input MP4 file")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Run output directory")
    parser.add_argument("--model", default="yolo11m.pt", help="Ultralytics detection model")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.20, help="Minimum confidence")
    parser.add_argument("--device", default="auto", help="auto, cpu, 0, cuda:0, etc.")
    parser.add_argument("--max-frames", type=int, default=300, help="Frames to process (0 = full video)")
    parser.add_argument("--start-frame", type=int, default=0, help="First source frame")
    parser.add_argument("--codec", default="mp4v", help="FourCC codec for annotated MP4")
    parser.add_argument(
        "--tracker", type=Path, default=Path("traffic_bytetrack.yaml"),
        help="Ultralytics tracker YAML configuration",
    )
    parser.add_argument(
        "--min-display-observations", type=int, default=3,
        help="Only draw a track after it has been detected this many times",
    )
    parser.add_argument("--enable-vehicle-classifier", action="store_true")
    parser.add_argument("--pedestrian-vehicle-reclassification", action="store_true")
    parser.add_argument("--vehicle-classifier-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--vehicle-classify-interval", type=int, default=15)
    parser.add_argument("--vehicle-classification-samples", type=int, default=3)
    parser.add_argument("--street-mask", type=Path, default=None, help="Optional binary mask image (same size as video) where non-zero=street; detections outside are ignored")
    parser.add_argument("--street-polygons", type=str, default=None, help="Reserved for programmatic normalized road polygons")
    parser.add_argument("--use-sahi", action="store_true", help="Use SAHI slicing for small-object detection when available (optional)")
    parser.add_argument("--show-road-mask", action="store_true", help="Show the road-mask status on the output video")
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    if requested not in {"cpu", "mps"} and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.", file=sys.stderr)
        return "cpu"
    return requested


def run(args: argparse.Namespace) -> None:
    if not args.source.is_file():
        raise FileNotFoundError(f"Video not found: {args.source}")
    if (args.max_frames < 0 or args.start_frame < 0 or args.min_display_observations < 1
            or args.vehicle_classify_interval < 1 or args.vehicle_classification_samples < 1):
        raise ValueError("frame counts must be non-negative and display observations must be at least 1")
    if not args.tracker.is_file():
        raise FileNotFoundError(f"Tracker configuration not found: {args.tracker}")

    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # OpenCV text sizes are absolute pixels. Scale them from a 1080p baseline
    # so annotations remain legible in the original 4K export.
    render_scale = max(width / 1920, height / 1080)
    label_font_scale = 0.65 * render_scale
    box_thickness = max(2, round(2 * render_scale))
    label_thickness = max(2, round(2 * render_scale))
    hud_font_scale = 0.9 * render_scale
    hud_thickness = max(2, round(2 * render_scale))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    run_name = f"{args.source.stem}_from_{args.start_frame:06d}"
    run_dir = args.output / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = run_dir / "annotated.mp4"
    writer = cv2.VideoWriter(
        str(annotated_path), cv2.VideoWriter_fourcc(*args.codec), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {annotated_path}")

    device = resolve_device(args.device)
    print(
        f"Loading {args.model} on {device}; tracker={args.tracker}. "
        f"Processing {'all frames' if args.max_frames == 0 else args.max_frames} frame(s)."
    )
    model = YOLO(args.model)
    detection_path = run_dir / "detections.csv"
    track_observations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    # Aerial views make fine class predictions flicker (for example, a truck
    # may briefly resemble a bus). ByteTrack IDs are the unit of analysis, so
    # keep the first assigned road-user class for each track's lifetime.
    # The original detector output is retained as raw_class_* in detections.csv.
    locked_classes: dict[int, tuple[int, str]] = {}
    vehicle_scores: dict[int, Counter[str]] = defaultdict(Counter)
    vehicle_sample_counts: Counter[int] = Counter()
    final_vehicle_modes: dict[int, tuple[int, str, float]] = {}
    vehicle_classifier = None
    if args.enable_vehicle_classifier:
        from vehicle_classifier import VehicleModeClassifier
        vehicle_classifier = VehicleModeClassifier(args.vehicle_classifier_model, device)

    # Optional: load a binary street mask so detections outside the street
    # polygon/area can be filtered out. The mask should be a same-size image
    # where non-zero pixels indicate valid street pixels.
    street_mask = None
    if getattr(args, 'street_mask', None):
        mask_path: Path = args.street_mask
        if not mask_path.is_file():
            raise FileNotFoundError(f"Street mask not found: {mask_path}")
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            raise RuntimeError(f"Could not read street mask: {mask_path}")
        # Resize mask if it doesn't match source resolution
        if mask_img.shape[1] != width or mask_img.shape[0] != height:
            mask_img = cv2.resize(mask_img, (width, height), interpolation=cv2.INTER_NEAREST)
        street_mask = (mask_img > 127).astype('uint8')

    # A polygon mask can be supplied by the run wrapper for a fixed camera
    # view. Coordinates are normalized to the source width and height.
    street_polygons = getattr(args, "street_polygons", None)
    if street_polygons:
        polygon_mask = street_mask if street_mask is not None else np.zeros((height, width), dtype="uint8")
        for polygon in street_polygons:
            points = np.array(
                [[round(x * width), round(y * height)] for x, y in polygon], dtype="int32"
            )
            cv2.fillPoly(polygon_mask, [points], 1)
        street_mask = polygon_mask
    mask_contours = []
    if street_mask is not None:
        mask_contours, _ = cv2.findContours(street_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Confirm the optional SAHI dependency before a requested Video 2 run.
    use_sahi = getattr(args, 'use_sahi', False)
    sahi_available = False
    if use_sahi:
        try:
            # Import lazily to keep runtime light when SAHI isn't required.
            from sahi.predict import get_sliced_prediction
            sahi_available = True
            print("SAHI dependency available; using the configured small-object pipeline.")
        except Exception as error:
            raise RuntimeError(
                "SAHI is enabled for this video but is not installed. "
                "Run: python -m pip install sahi. "
                f"Import error: {error}"
            ) from error

    processed = 0
    with detection_path.open("w", newline="", encoding="utf-8") as detection_file:
        fieldnames = [
            "frame", "timestamp_s", "track_id", "class_id", "class_name",
            "raw_class_id", "raw_class_name", "class_source", "class_consensus", "confidence",
            "x1", "y1", "x2", "y2", "center_x", "center_y", "width", "height",
        ]
        csv_writer = csv.DictWriter(detection_file, fieldnames=fieldnames)
        csv_writer.writeheader()

        while True:
            if args.max_frames and processed >= args.max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            source_frame = args.start_frame + processed
            inference_frame = frame
            if street_mask is not None and getattr(args, "show_road_mask", False):
                mask_overlay = np.zeros_like(frame)
                mask_overlay[street_mask > 0] = (40, 180, 40)
                frame = cv2.addWeighted(frame, 0.62, mask_overlay, 0.38, 0)
                frame[street_mask == 0] = (frame[street_mask == 0] * 0.58).astype(np.uint8)
                cv2.putText(
                    frame, "ROAD MASK ACTIVE", (30, int(82 * render_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, hud_font_scale, ANNOTATION_COLOR,
                    hud_thickness, cv2.LINE_AA,
                )
            if street_mask is not None:
                inference_frame = cv2.bitwise_and(inference_frame, inference_frame, mask=street_mask)
            result = model.track(
                inference_frame,
                persist=True,
                tracker=str(args.tracker),
                classes=list(ROAD_CLASSES),
                conf=args.conf,
                imgsz=args.imgsz,
                device=device,
                verbose=False,
            )[0]

            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.int().cpu().tolist()
                confidences = result.boxes.conf.cpu().tolist()
                ids = result.boxes.id.int().cpu().tolist()
                pending: list[dict[str, Any]] = []
                classifier_crops = []
                classifier_track_ids = []
                classifier_raw_class_ids = []
                for (x1, y1, x2, y2), class_id, confidence, track_id in zip(boxes, classes, confidences, ids):
                    raw_class_id = int(class_id)
                    raw_class_name = ROAD_CLASSES[raw_class_id]
                    center_x, center_y = (float(x1) + float(x2)) / 2, (float(y1) + float(y2)) / 2
                    foot_x, foot_y = center_x, float(y2)
                    # Use the contact point rather than box overlap or center.
                    # This keeps vehicles near road edges and rejects rooftop boxes.
                    if street_mask is not None:
                        cy = int(round(foot_y))
                        cx = int(round(foot_x))
                        if cy < 0 or cy >= height or cx < 0 or cx >= width or street_mask[cy, cx] == 0:
                            # Skip this detection as it lies outside the street mask
                            continue
                    if track_id not in locked_classes:
                        locked_classes[track_id] = (raw_class_id, raw_class_name)
                    observation = {
                        "frame": source_frame,
                        "timestamp_s": round(source_frame / fps, 4),
                        "track_id": int(track_id),
                        "raw_class_id": raw_class_id,
                        "raw_class_name": raw_class_name,
                        "confidence": round(float(confidence), 4),
                        "x1": round(float(x1), 2), "y1": round(float(y1), 2),
                        "x2": round(float(x2), 2), "y2": round(float(y2), 2),
                        "center_x": round(center_x, 2), "center_y": round(center_y, 2),
                        "width": round(float(x2) - float(x1), 2),
                        "height": round(float(y2) - float(y1), 2),
                    }
                    track_observations[int(track_id)].append(observation)
                    pending.append(observation)
                    observation_count = len(track_observations[int(track_id)])
                    if (vehicle_classifier and (
                            raw_class_id in VEHICLE_DETECTOR_IDS
                            or (raw_class_id == 0 and args.pedestrian_vehicle_reclassification))
                            and observation_count >= args.min_display_observations
                            and (observation_count - args.min_display_observations) % args.vehicle_classify_interval == 0):
                        pad_x = max(4, int((x2 - x1) * 0.12))
                        pad_y = max(4, int((y2 - y1) * 0.12))
                        left, top = max(0, int(x1) - pad_x), max(0, int(y1) - pad_y)
                        right, bottom = min(width, int(x2) + pad_x), min(height, int(y2) + pad_y)
                        crop = frame[top:bottom, left:right]
                        if crop.shape[0] >= 24 and crop.shape[1] >= 24:
                            classifier_crops.append(crop)
                            classifier_track_ids.append(int(track_id))
                            classifier_raw_class_ids.append(raw_class_id)

                if classifier_crops:
                    for track_id, raw_class_id, scores in zip(
                            classifier_track_ids, classifier_raw_class_ids,
                            vehicle_classifier.classify(classifier_crops)):
                        vehicle_scores[track_id].update(scores)
                        vehicle_sample_counts[track_id] += 1
                        if (track_id not in final_vehicle_modes
                                and vehicle_sample_counts[track_id] >= args.vehicle_classification_samples):
                            winning_name, winning_score = vehicle_scores[track_id].most_common(1)[0]
                            consensus = winning_score / sum(vehicle_scores[track_id].values())
                            is_confirmed_vehicle = raw_class_id in VEHICLE_DETECTOR_IDS
                            is_confirmed_scooter = (
                                raw_class_id == 0
                                and winning_name == "motorcycle"
                                and winning_score >= 0.40
                            )
                            if is_confirmed_vehicle or is_confirmed_scooter:
                                winning_id = next(key for key, value in MODE_CLASS_IDS.items() if value == winning_name)
                                final_vehicle_modes[track_id] = (winning_id, winning_name, consensus)

                for observation in pending:
                    track_id = observation["track_id"]
                    raw_class_id = observation["raw_class_id"]
                    stable_class_id, class_name = locked_classes[track_id]
                    class_source = "detector_fallback"
                    class_consensus = 0.0
                    if track_id in final_vehicle_modes:
                        stable_class_id, class_name, class_consensus = final_vehicle_modes[track_id]
                        class_source = "track_level_classifier"
                    row = {
                        **observation,
                        "class_id": stable_class_id,
                        "class_name": class_name,
                        "class_source": class_source,
                        "class_consensus": round(class_consensus, 4),
                    }
                    csv_writer.writerow(row)
                    # A one-frame object is normally detector noise. Keep it
                    # in the raw CSV but wait for confirmation before drawing
                    # it in the presentation video.
                    if len(track_observations[track_id]) >= args.min_display_observations:
                        color = ANNOTATION_COLOR
                        cv2.rectangle(
                            frame, (int(row["x1"]), int(row["y1"])),
                            (int(row["x2"]), int(row["y2"])), color, box_thickness
                        )
                        cv2.putText(
                            frame, f"{class_name} #{track_id} {row['confidence']:.2f}",
                            (int(row["x1"]), max(20, int(row["y1"]) - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                            label_font_scale, color, label_thickness, cv2.LINE_AA,
                        )

            cv2.putText(
                frame, f"frame {source_frame} | {source_frame / fps:.1f}s | tracked: {len(track_observations)}",
                (30, int(45 * render_scale)), cv2.FONT_HERSHEY_SIMPLEX,
                hud_font_scale, (255, 255, 255), hud_thickness + 2, cv2.LINE_AA,
            )
            cv2.putText(
                frame, f"frame {source_frame} | {source_frame / fps:.1f}s | tracked: {len(track_observations)}",
                (30, int(45 * render_scale)), cv2.FONT_HERSHEY_SIMPLEX,
                hud_font_scale, (20, 20, 20), hud_thickness, cv2.LINE_AA,
            )
            writer.write(frame)
            processed += 1
            if processed % 30 == 0:
                print(f"Processed {processed} frame(s)")

    cap.release()
    writer.release()
    tracks_path = run_dir / "tracks.csv"
    with tracks_path.open("w", newline="", encoding="utf-8") as track_file:
        fields = [
            "track_id", "class_name", "class_source", "class_consensus", "observations", "first_frame", "last_frame",
            "start_time_s", "end_time_s", "duration_s", "mean_confidence",
            "start_center_x", "start_center_y", "end_center_x", "end_center_y",
        ]
        writer_csv = csv.DictWriter(track_file, fieldnames=fields)
        writer_csv.writeheader()
        for track_id, observations in sorted(track_observations.items()):
            first, last = observations[0], observations[-1]
            _, class_name = locked_classes[track_id]
            class_source = "detector_fallback"
            class_consensus = 0.0
            if track_id in final_vehicle_modes:
                _, class_name, class_consensus = final_vehicle_modes[track_id]
                class_source = "track_level_classifier"
            writer_csv.writerow({
                "track_id": track_id,
                "class_name": class_name,
                "class_source": class_source,
                "class_consensus": round(class_consensus, 4),
                "observations": len(observations),
                "first_frame": first["frame"], "last_frame": last["frame"],
                "start_time_s": first["timestamp_s"], "end_time_s": last["timestamp_s"],
                "duration_s": round(last["timestamp_s"] - first["timestamp_s"], 4),
                "mean_confidence": round(sum(item["confidence"] for item in observations) / len(observations), 4),
                "start_center_x": first["center_x"], "start_center_y": first["center_y"],
                "end_center_x": last["center_x"], "end_center_y": last["center_y"],
            })

    metadata = {
        "source": str(args.source), "source_dimensions": [width, height], "source_fps": fps,
        "source_total_frames": total_frames, "start_frame": args.start_frame,
        "processed_frames": processed, "model": args.model, "imgsz": args.imgsz,
        "confidence_threshold": args.conf, "device": device, "tracker": str(args.tracker),
        "min_display_observations": args.min_display_observations,
        "vehicle_classifier_enabled": args.enable_vehicle_classifier,
        "vehicle_classifier_model": args.vehicle_classifier_model if args.enable_vehicle_classifier else None,
        "vehicle_classification_samples": args.vehicle_classification_samples,
        "unique_track_ids": len(track_observations),
    }
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Complete: {run_dir}")
    print(f"Unique track IDs: {len(track_observations)}")


if __name__ == "__main__":
    try:
        if len(sys.argv) == 1:
            # Running this file from VS Code should use the same configured
            # entry point as run_level1.py, including its road mask.
            from run_level1 import main
            main()
        else:
            run(parse_args())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error

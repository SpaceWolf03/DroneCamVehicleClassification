#!/usr/bin/env python3

"""
Run this file directly from VS Code.

Edit only the SETTINGS block below before a new run.
No terminal parameters are required.

Leave run_full_video=False until the short sample looks correct.
"""

from __future__ import annotations

from argparse import Namespace
from datetime import datetime
from pathlib import Path

from track_traffic import run


def sample_closed_spline(
    control_points: list[tuple[float, float]],
    samples_per_edge: int = 2,
) -> list[tuple[float, float]]:
    """Sample a closed Catmull-Rom spline from normalized controls."""

    sampled = []
    point_count = len(control_points)

    for index in range(point_count):

        p0 = control_points[(index - 1) % point_count]
        p1 = control_points[index]
        p2 = control_points[(index + 1) % point_count]
        p3 = control_points[(index + 2) % point_count]

        for step in range(samples_per_edge):

            t = step / samples_per_edge
            t2 = t * t
            t3 = t2 * t

            x = 0.5 * (
                (2 * p1[0])
                + (-p0[0] + p2[0]) * t
                + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
            )

            y = 0.5 * (
                (2 * p1[1])
                + (-p0[1] + p2[1]) * t
                + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
            )

            sampled.append(
                (
                    max(0.0, min(1.0, x)),
                    max(0.0, min(1.0, y)),
                )
            )

    return sampled


# ============================================================
# SETTINGS
# ============================================================

VIDEO = "multi-road"  # "intersection" or "multi-road"


# ============================================================
# MULTI-ROAD MASK
# ============================================================

MULTI_ROAD_MASK = [

    [
        (0.00, 0.54),
        (0.20, 0.47),
        (0.46, 0.72),
        (0.75, 1.00),
        (0.00, 1.00),
    ],

    [
        (0.18, 0.00),
        (0.42, 0.00),
        (0.58, 0.38),
        (0.72, 0.52),
        (0.62, 0.66),
        (0.38, 0.46),
    ],

    [
        (0.58, 0.00),
        (1.00, 0.00),
        (1.00, 0.34),
        (0.83, 0.39),
        (0.68, 0.25),
    ],
]


# ============================================================
# INTERSECTION MASK
#
# Directly fitted to the GREEN ROAD REGION in the reference
# image you supplied.
#
# Coordinates are normalized:
# x: 0 = left, 1 = right
# y: 0 = top,  1 = bottom
# ============================================================

INTERSECTION_MASK = [

    [
        (0.0000, 0.6366),
        (0.0000, 0.8102),
        (0.0456, 0.8079),
        (0.0000, 0.9236),
        (0.0299, 0.9120),
        (0.0378, 0.9468),
        (0.0143, 0.9745),
        (0.0430, 0.9653),
        (0.0443, 0.9977),
        (0.0990, 0.9977),
        (0.1302, 0.9398),
        (0.1536, 0.9676),
        (0.1536, 0.9861),
        (0.1393, 0.9977),
        (0.1615, 0.9977),
        (0.1706, 0.9606),
        (0.2174, 0.9444),
        (0.3034, 0.8426),
        (0.3698, 0.7894),
        (0.3737, 0.7616),
        (0.3958, 0.7338),
        (0.4154, 0.7431),
        (0.4596, 0.6759),
        (0.5052, 0.7153),
        (0.4987, 0.7546),
        (0.4427, 0.7917),
        (0.4505, 0.8333),
        (0.4674, 0.8194),
        (0.4805, 0.8426),
        (0.4701, 0.8819),
        (0.4987, 0.9421),
        (0.5013, 0.9769),
        (0.5391, 0.9699),
        (0.5391, 0.9977),
        (0.6107, 0.9977),
        (0.5898, 0.9792),
        (0.5951, 0.9537),
        (0.5820, 0.9491),
        (0.5807, 0.9074),
        (0.6211, 0.9190),
        (0.6328, 0.9537),
        (0.6589, 0.9306),
        (0.6536, 0.9630),
        (0.6732, 0.9977),
        (0.9805, 0.9977),
        (0.9440, 0.9306),
        (0.8854, 0.9444),
        (0.8776, 0.9907),
        (0.8542, 0.9792),
        (0.6940, 0.7130),
        (0.6198, 0.6250),
        (0.6003, 0.5231),
        (0.6602, 0.4306),
        (0.6901, 0.4282),
        (0.7044, 0.3935),
        (0.7240, 0.3981),
        (0.7344, 0.4213),
        (0.7617, 0.4190),
        (0.7695, 0.3148),
        (0.8190, 0.2569),
        (0.8438, 0.2778),
        (0.8529, 0.2639),
        (0.8490, 0.2338),
        (0.8607, 0.2222),
        (0.8789, 0.2292),
        (0.9336, 0.1968),
        (0.9740, 0.1389),
        (0.9909, 0.1574),
        (0.9987, 0.1412),
        (0.9909, 0.1296),
        (0.9987, 0.1042),
        (0.9896, 0.1019),
        (0.9987, 0.0787),
        (0.9987, 0.0000),
        (0.9336, 0.0000),
        (0.8372, 0.1134),
        (0.7721, 0.1505),
        (0.7552, 0.1944),
        (0.6823, 0.2616),
        (0.6693, 0.2431),
        (0.6133, 0.2824),
        (0.6068, 0.1852),
        (0.5859, 0.1898),
        (0.5703, 0.1667),
        (0.5729, 0.2245),
        (0.5625, 0.2616),
        (0.5404, 0.2639),
        (0.5052, 0.3194),
        (0.4740, 0.3148),
        (0.4401, 0.3356),
        (0.4206, 0.3773),
        (0.4258, 0.4051),
        (0.4453, 0.4329),
        (0.4896, 0.4213),
        (0.5078, 0.3611),
        (0.5638, 0.3218),
        (0.5703, 0.2847),
        (0.6094, 0.2870),
        (0.6198, 0.3264),
        (0.5560, 0.3866),
        (0.5443, 0.3727),
        (0.5286, 0.3843),
        (0.5365, 0.3866),
        (0.5286, 0.4259),
        (0.4284, 0.5139),
        (0.4102, 0.5139),
        (0.4076, 0.4954),
        (0.4036, 0.5162),
        (0.3867, 0.5139),
        (0.3906, 0.4213),
        (0.3724, 0.4213),
        (0.3542, 0.3889),
        (0.3424, 0.3333),
        (0.2786, 0.2199),
        (0.2161, 0.0556),
        (0.2174, 0.0231),
        (0.2643, 0.0255),
        (0.2760, 0.0417),
        (0.2760, 0.0787),
        (0.2917, 0.0972),
        (0.2786, 0.0000),
        (0.2109, 0.0000),
        (0.2122, 0.0440),
        (0.1797, 0.0833),
        (0.1810, 0.1181),
        (0.3268, 0.4352),
        (0.3281, 0.4583),
        (0.3073, 0.4861),
        (0.3346, 0.4769),
        (0.3581, 0.5139),
        (0.3516, 0.5741),
        (0.3333, 0.5579),
        (0.2917, 0.5972),
        (0.2982, 0.6088),
        (0.3307, 0.5880),
        (0.3307, 0.6134),
        (0.3125, 0.6366),
        (0.2904, 0.6366),
        (0.2826, 0.6134),
        (0.2604, 0.6366),
        (0.2656, 0.6644),
        (0.2161, 0.6713),
        (0.1719, 0.7153),
        (0.1576, 0.7153),
        (0.1562, 0.6944),
        (0.1836, 0.6620),
        (0.1641, 0.6690),
        (0.1185, 0.6088),
        (0.0833, 0.6181),
        (0.0755, 0.5972),
        (0.0729, 0.6111),
        (0.0312, 0.6134),
        (0.0260, 0.6435),
    ],
]


# ============================================================
# SPLINE
# ============================================================

INTERSECTION_SPLINE_MASK = [
    sample_closed_spline(
        control_points,
        samples_per_edge=2,
    )
    for control_points in INTERSECTION_MASK
]


# ============================================================
# VIDEO CONFIG
# ============================================================

VIDEO_CONFIG = {

    "intersection": {
        "run_full_video": False,
        "start_frame": 3000,
        "frames_to_process": 150,

        "image_size": 1536,
        "detection_confidence": 0.15,

        "min_display_observations": 3,

        "street_polygons": INTERSECTION_SPLINE_MASK,

        "enable_vehicle_classifier": True,
        "pedestrian_vehicle_reclassification": True,
        "vehicle_classify_interval": 15,
        "vehicle_classification_samples": 3,
    },

    "multi-road": {
        "run_full_video": True,
        "start_frame": 3000,
        "frames_to_process": 150,

        # Restore the higher-resolution setting for small aerial vehicles.
        "image_size": 2048,
        "detection_confidence": 0.10,

        "min_display_observations": 3,

        "street_polygons": None,

        "enable_vehicle_classifier": False,
        "pedestrian_vehicle_reclassification": False,
        "vehicle_classify_interval": 15,
        "vehicle_classification_samples": 3,
        "use_sahi": True,
        "show_road_mask": False,
    },
}


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_ROOT = Path("outputs")


# ============================================================
# VIDEOS
# ============================================================

VIDEOS = {
    "intersection": Path("Intersection_Merged.MP4"),
    "multi-road": Path("Multi_Road_Merged.MP4"),
}


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    if VIDEO not in VIDEOS:
        raise ValueError(
            f"VIDEO must be one of: {', '.join(VIDEOS)}"
        )

    source = VIDEOS[VIDEO]
    settings = VIDEO_CONFIG[VIDEO]

    max_frames = (
        0
        if settings["run_full_video"]
        else settings["frames_to_process"]
    )

    start_frame = (
        0
        if settings["run_full_video"]
        else settings["start_frame"]
    )

    if VIDEO == "intersection":

        video_output_root = (
            OUTPUT_ROOT /
            "video_1_intersection_merged"
        )

    elif VIDEO == "multi-road":

        video_output_root = (
            OUTPUT_ROOT /
            "video_2_multi_road_merged"
        )

    else:

        raise ValueError(
            f"No output folder configured for VIDEO={VIDEO!r}"
        )

    run_serial = 1

    while (
        video_output_root.exists()
        and any(
            path.name.startswith(
                f"run_{run_serial:03d}_"
            )
            for path in video_output_root.iterdir()
        )
    ):
        run_serial += 1

    created_at = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    run_output = (
        video_output_root /
        f"run_{run_serial:03d}_{created_at}"
    )

    print(
        f"Running {source} from frame {start_frame}; "
        f"{'full video' if settings['run_full_video'] else f'{max_frames} frames'}; "
        f"output: {run_output}"
    )

    run(
        Namespace(

            source=source,
            output=run_output,

            model="yolo11m.pt",

            imgsz=settings["image_size"],
            conf=settings["detection_confidence"],

            # Your local RTX 5060
            device=0,

            max_frames=max_frames,
            start_frame=start_frame,

            codec="mp4v",

            tracker=Path("traffic_bytetrack.yaml"),

            min_display_observations=(
                settings["min_display_observations"]
            ),

            enable_vehicle_classifier=(
                settings["enable_vehicle_classifier"]
            ),

            pedestrian_vehicle_reclassification=(
                settings["pedestrian_vehicle_reclassification"]
            ),

            vehicle_classifier_model=(
                "openai/clip-vit-base-patch32"
            ),

            vehicle_classify_interval=(
                settings["vehicle_classify_interval"]
            ),

            vehicle_classification_samples=(
                settings["vehicle_classification_samples"]
            ),

            # Spatial filtering
            street_mask=None,
            street_polygons=settings["street_polygons"],
            use_sahi=settings.get("use_sahi", False),
            show_road_mask=settings.get("show_road_mask", False),
        )
    )


if __name__ == "__main__":
    main()
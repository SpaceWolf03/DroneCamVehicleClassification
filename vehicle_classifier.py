"""Track-level zero-shot vehicle mode classification.

The detector is responsible for finding objects and maintaining IDs. CLIP is
only asked to distinguish the vehicle modes that COCO does not provide, and
its evidence is accumulated across multiple crops from the same track.
"""

from __future__ import annotations

from typing import Iterable

import cv2
import torch
from PIL import Image


VEHICLE_MODES = ("car", "LGV", "HGV", "bus", "truck", "motorcycle")
PROMPTS = (
    "an aerial drone image of a passenger car",
    "an aerial drone image of a light goods vehicle or delivery van",
    "an aerial drone image of a heavy goods vehicle or articulated lorry",
    "an aerial drone image of a bus",
    "an aerial drone image of a medium cargo truck",
    "an aerial drone image of a motorcycle or motor scooter",
)


class VehicleModeClassifier:
    """Classifies batches of BGR OpenCV crops into the required vehicle modes."""

    def __init__(self, model_name: str, device: str) -> None:
        try:
            from transformers import AutoModelForZeroShotImageClassification, AutoProcessor
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RuntimeError(
                "Vehicle classification needs transformers. Install project requirements first."
            ) from error

        self.device = "cuda" if device != "cpu" and torch.cuda.is_available() else "cpu"
        print(f"Loading vehicle classifier {model_name} on {self.device} (first run downloads its weights).")
        # Resolve the snapshot first. This makes subsequent runs deterministic
        # and allows Transformers to load the already-downloaded local files.
        model_path = snapshot_download(repo_id=model_name)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForZeroShotImageClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def classify(self, bgr_crops: Iterable[object]) -> list[dict[str, float]]:
        images = [Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) for crop in bgr_crops]
        if not images:
            return []
        inputs = self.processor(text=list(PROMPTS), images=images, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        probabilities = self.model(**inputs).logits_per_image.softmax(dim=1).cpu().tolist()
        return [dict(zip(VEHICLE_MODES, scores)) for scores in probabilities]

"""YOLOv3 vehicle detector using OpenCV DNN."""
from pathlib import Path
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np

from config import (
    COCO_NAMES,
    CONFIDENCE_THRESHOLD,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    NMS_THRESHOLD,
    VEHICLE_CLASSES,
    YOLO_CFG,
    YOLO_WEIGHTS,
)
from src.utils import apply_nms, get_output_layers, load_classes


class VehicleDetector:
    """Detect cars, trucks and buses with YOLOv3."""

    def __init__(
        self,
        weights_path: Path = YOLO_WEIGHTS,
        cfg_path: Path = YOLO_CFG,
        names_path: Path = COCO_NAMES,
        input_width: int = INPUT_WIDTH,
        input_height: int = INPUT_HEIGHT,
        conf_threshold: float = CONFIDENCE_THRESHOLD,
        nms_threshold: float = NMS_THRESHOLD,
        vehicle_classes: Optional[Set[str]] = None,
    ):
        weights_path = Path(weights_path)
        cfg_path = Path(cfg_path)
        names_path = Path(names_path)

        if not weights_path.exists():
            raise FileNotFoundError(
                f"YOLO weights not found: {weights_path}\n"
                "Download from: https://pjreddie.com/media/files/yolov3.weights"
            )
        if not cfg_path.exists():
            raise FileNotFoundError(f"YOLO cfg not found: {cfg_path}")
        if not names_path.exists():
            raise FileNotFoundError(f"Class names not found: {names_path}")

        self.input_width = input_width
        self.input_height = input_height
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.vehicle_classes = vehicle_classes or set(VEHICLE_CLASSES)

        self.classes = load_classes(str(names_path))
        self.net = cv2.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.output_layers = get_output_layers(self.net)

    def detect(
        self, frame: np.ndarray
    ) -> Tuple[List[Tuple[int, int, int, int]], List[float], List[str]]:
        """Run detection on one frame.

        Returns:
            boxes: (x, y, w, h) in original frame coordinates
            confidences: confidence scores
            class_names: detected class labels
        """
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1 / 255.0,
            size=(self.input_width, self.input_height),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes: List[Tuple[int, int, int, int]] = []
        confidences: List[float] = []
        class_ids: List[int] = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])

                if confidence < self.conf_threshold:
                    continue

                class_name = self.classes[class_id]
                if class_name not in self.vehicle_classes:
                    continue

                cx = int(detection[0] * w)
                cy = int(detection[1] * h)
                bw = int(detection[2] * w)
                bh = int(detection[3] * h)

                x = max(0, cx - bw // 2)
                y = max(0, cy - bh // 2)
                bw = min(bw, w - x)
                bh = min(bh, h - y)

                if bw <= 0 or bh <= 0:
                    continue

                boxes.append((x, y, bw, bh))
                confidences.append(confidence)
                class_ids.append(class_id)

        keep = apply_nms(boxes, confidences, self.conf_threshold, self.nms_threshold)

        final_boxes = [boxes[i] for i in keep]
        final_confidences = [confidences[i] for i in keep]
        final_class_names = [self.classes[class_ids[i]] for i in keep]

        return final_boxes, final_confidences, final_class_names

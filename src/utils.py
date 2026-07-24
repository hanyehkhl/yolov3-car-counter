"""Utility helpers: class loading, NMS, drawing, IoU."""
from typing import List, Sequence, Tuple, Union

import cv2
import numpy as np


def load_classes(names_path: str) -> List[str]:
    with open(names_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def get_output_layers(net) -> List[str]:
    layer_names = net.getLayerNames()
    out_layers = net.getUnconnectedOutLayers()
    if isinstance(out_layers, np.ndarray):
        flat = out_layers.flatten()
    else:
        flat = np.array(out_layers).flatten()
    return [layer_names[int(i) - 1] for i in flat]


def compute_iou(box_a: Tuple[int, ...], box_b: Tuple[int, ...]) -> float:
    """IoU for boxes in (x, y, w, h) format."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax1, ay1, ax2, ay2 = ax, ay, ax + aw, ay + ah
    bx1, by1, bx2, by2 = bx, by, bx + bw, by + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = aw * ah + bw * bh - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def apply_nms(
    boxes: List[Tuple[int, int, int, int]],
    confidences: List[float],
    conf_threshold: float,
    nms_threshold: float,
) -> List[int]:
    """Run OpenCV NMS and return kept indices."""
    if not boxes:
        return []
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    if indices is None or len(indices) == 0:
        return []
    return [int(i) for i in np.array(indices).flatten()]


def draw_predictions(
    frame: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    confidences: List[float],
    class_names: List[str],
    color: Tuple[int, int, int] = (0, 255, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    output = frame.copy()
    for (x, y, w, h), conf, name in zip(boxes, confidences, class_names):
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        label = f"{name}: {conf:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        label_y = max(y, text_h + 8)
        cv2.rectangle(
            output,
            (x, label_y - text_h - 8),
            (x + text_w + 4, label_y),
            color,
            -1,
        )
        cv2.putText(
            output,
            label,
            (x + 2, label_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            1,
            cv2.LINE_AA,
        )
    return output


def draw_count_overlay(
    frame: np.ndarray,
    frame_number: int,
    car_count: int,
) -> np.ndarray:
    text = f"Frame: {frame_number}  Vehicles: {car_count}"
    cv2.rectangle(frame, (5, 5), (420, 45), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame

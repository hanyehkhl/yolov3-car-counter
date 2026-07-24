"""Main pipeline: detect and count vehicles in a traffic video."""
import argparse
import csv
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from config import (
    DEFAULT_CSV,
    DEFAULT_INPUT_VIDEO,
    DEFAULT_OUTPUT_VIDEO,
    INPUT_HEIGHT,
    INPUT_WIDTH,
)
from src.detector import VehicleDetector
from src.tracker import VehicleTracker
from src.utils import draw_count_overlay, draw_predictions


def process_video(
    input_path: Path,
    output_path: Path,
    csv_path: Path,
    input_width: int = INPUT_WIDTH,
    input_height: int = INPUT_HEIGHT,
    display: bool = False,
):
    """Process video frame-by-frame; write annotated video + CSV counts."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    csv_path = Path(csv_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video writer: {output_path}")

    detector = VehicleDetector(input_width=input_width, input_height=input_height)
    tracker = VehicleTracker()

    csv_rows = []
    frame_number = 0

    print(f"Processing: {input_path}")
    print(f"Resolution: {width}x{height} @ {fps:.1f} FPS | frames≈{total_frames}")
    print(f"Output video: {output_path}")
    print(f"Output CSV:   {csv_path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1

            boxes, confidences, class_names = detector.detect(frame)
            tracker.update(boxes)
            car_count = len(boxes)

            annotated = draw_predictions(frame, boxes, confidences, class_names)
            annotated = draw_count_overlay(annotated, frame_number, car_count)

            writer.write(annotated)
            csv_rows.append([frame_number, car_count])

            if frame_number % 30 == 0:
                print(f"  frame {frame_number}/{total_frames or '?'}  vehicles={car_count}")

            if display:
                cv2.imshow("YOLOv3 Vehicle Counter", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped by user.")
                    break
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["frame_number", "car_count"])
        csv_writer.writerows(csv_rows)

    print(f"Done. Processed {frame_number} frames.")
    print(f"Unique vehicles tracked: {tracker.total_unique_count}")
    print(f"CSV saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Detect and count vehicles in traffic videos with YOLOv3."
    )
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT_VIDEO))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_VIDEO))
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV))
    parser.add_argument("--width", type=int, default=INPUT_WIDTH)
    parser.add_argument("--height", type=int, default=INPUT_HEIGHT)
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()

    process_video(
        input_path=Path(args.input),
        output_path=Path(args.output),
        csv_path=Path(args.csv),
        input_width=args.width,
        input_height=args.height,
        display=args.display,
    )


if __name__ == "__main__":
    main()

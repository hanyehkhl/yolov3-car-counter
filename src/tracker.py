"""Simple IoU-based vehicle tracker across frames."""
from typing import Dict, List, Tuple

from src.utils import compute_iou


class VehicleTracker:
    """Assign stable IDs to vehicles between consecutive frames."""

    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 5):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks: Dict[int, Dict] = {}
        self.total_unique_count = 0

    def update(self, boxes: List[Tuple[int, int, int, int]]) -> int:
        """Match detections to existing tracks; return active track count."""
        new_tracks: Dict[int, Dict] = {}
        matched_detections = set()
        matched_tracks = set()

        if boxes and self.tracks:
            track_ids = list(self.tracks.keys())
            pairs = []
            for tid in track_ids:
                t_box = self.tracks[tid]["box"]
                for j, d_box in enumerate(boxes):
                    pairs.append((compute_iou(t_box, d_box), tid, j))
            pairs.sort(reverse=True, key=lambda p: p[0])

            for iou, tid, didx in pairs:
                if iou < self.iou_threshold:
                    break
                if tid in matched_tracks or didx in matched_detections:
                    continue
                new_tracks[tid] = {"box": boxes[didx], "missed": 0}
                matched_tracks.add(tid)
                matched_detections.add(didx)

        for didx, box in enumerate(boxes):
            if didx not in matched_detections:
                new_tracks[self.next_id] = {"box": box, "missed": 0}
                self.total_unique_count += 1
                self.next_id += 1

        for tid, track in self.tracks.items():
            if tid not in matched_tracks and track["missed"] < self.max_missed:
                track = dict(track)
                track["missed"] += 1
                new_tracks[tid] = track

        self.tracks = new_tracks
        return len([t for t in self.tracks.values() if t["missed"] == 0])

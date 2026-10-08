"""
modules/image_emotion/tracker.py — Lightweight IOU-based multi-object
tracker.

Purpose: give each detected person a stable `track_id` across video
frames so that (a) emotion probabilities can be temporally smoothed per
person instead of flickering frame-to-frame, and (b) kinematic features
like movement speed / pacing can be computed from an actual position
history rather than a single frame.

This is intentionally simple (no re-identification embeddings, no Kalman
filter) — it is a real, deterministic geometric algorithm (IOU + nearest
match), not a placeholder. For clinical-room use with a handful of
people and a stationary camera it performs well; swapping in DeepSORT/
ByteTrack later is a drop-in extension point if you need robustness to
long occlusions.
"""
import itertools
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from .detector import FaceBox


def _iou(a: FaceBox, b: FaceBox) -> float:
    ax1, ay1, ax2, ay2 = a.x, a.y, a.x + a.w, a.y + a.h
    bx1, by1, bx2, by2 = b.x, b.y, b.x + b.w, b.y + b.h
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = a.w * a.h
    area_b = b.w * b.h
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class TrackHistoryPoint:
    t: float
    cx: float
    cy: float
    box_w: int
    box_h: int


@dataclass
class Track:
    track_id: int
    box: FaceBox
    last_seen_t: float
    missed_frames: int = 0
    history: Deque[TrackHistoryPoint] = field(default_factory=lambda: deque(maxlen=256))
    emotion_ema: Optional[Dict[str, float]] = None

    def centroid(self) -> Tuple[float, float]:
        return self.box.x + self.box.w / 2.0, self.box.y + self.box.h / 2.0


class PersonTracker:
    def __init__(self, iou_threshold: float, max_missed_frames: int, history_window: int):
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.history_window = history_window
        self.tracks: Dict[int, Track] = {}
        self._id_counter = itertools.count(1)

    def update(self, detections: List[FaceBox], timestamp: Optional[float] = None) -> Dict[int, FaceBox]:
        """Match new detections to existing tracks; returns {track_id: FaceBox}."""
        t = timestamp if timestamp is not None else time.time()
        unmatched_dets = list(range(len(detections)))
        matched_track_ids = set()

        # Greedy best-IOU matching (fine for small N; swap for Hungarian if N grows large)
        pairs = []
        for tid, track in self.tracks.items():
            for di in unmatched_dets:
                score = _iou(track.box, detections[di])
                if score >= self.iou_threshold:
                    pairs.append((score, tid, di))
        pairs.sort(key=lambda p: p[0], reverse=True)

        assigned_dets = set()
        for score, tid, di in pairs:
            if tid in matched_track_ids or di in assigned_dets:
                continue
            track = self.tracks[tid]
            track.box = detections[di]
            track.last_seen_t = t
            track.missed_frames = 0
            cx, cy = track.centroid()
            track.history.append(TrackHistoryPoint(t, cx, cy, detections[di].w, detections[di].h))
            if len(track.history) > self.history_window:
                track.history.popleft()
            matched_track_ids.add(tid)
            assigned_dets.add(di)

        # New tracks for unmatched detections
        for di in unmatched_dets:
            if di in assigned_dets:
                continue
            new_id = next(self._id_counter)
            track = Track(track_id=new_id, box=detections[di], last_seen_t=t)
            cx, cy = track.centroid()
            track.history.append(TrackHistoryPoint(t, cx, cy, detections[di].w, detections[di].h))
            self.tracks[new_id] = track
            matched_track_ids.add(new_id)

        # Age out unmatched tracks
        stale_ids = []
        for tid, track in self.tracks.items():
            if tid not in matched_track_ids:
                track.missed_frames += 1
                if track.missed_frames > self.max_missed_frames:
                    stale_ids.append(tid)
        for tid in stale_ids:
            del self.tracks[tid]

        return {tid: track.box for tid, track in self.tracks.items() if tid in matched_track_ids}

    def get_track(self, track_id: int) -> Optional[Track]:
        return self.tracks.get(track_id)
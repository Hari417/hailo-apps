# region imports
from __future__ import annotations

import gi

gi.require_version("Gst", "1.0")
import cv2
import numpy as np
import hailo
from gi.repository import Gst

from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.common.hailo_logger import get_logger
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class

from hailo_apps.hailo_app_python.apps.object_counting.object_counting_pipeline import (
    GStreamerObjectCountingApp,
    RegionConfig,
)

hailo_logger = get_logger(__name__)
# endregion imports


def _segment_intersects(p1: tuple[float, float], p2: tuple[float, float], q1: tuple[float, float], q2: tuple[float, float]) -> bool:
    """Return True if segment p1-p2 intersects segment q1-q2."""

    def orient(a, b, c) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a, b, c) -> bool:
        return (
            min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
        )

    o1 = orient(p1, p2, q1)
    o2 = orient(p1, p2, q2)
    o3 = orient(q1, q2, p1)
    o4 = orient(q1, q2, p2)

    # General case
    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
        return True

    # Collinear cases
    eps = 1e-9
    if abs(o1) < eps and on_segment(p1, p2, q1):
        return True
    if abs(o2) < eps and on_segment(p1, p2, q2):
        return True
    if abs(o3) < eps and on_segment(q1, q2, p1):
        return True
    if abs(o4) < eps and on_segment(q1, q2, p2):
        return True

    return False


def _point_in_polygon(point: tuple[float, float], polygon: tuple[tuple[int, int], ...]) -> bool:
    """Ray casting point-in-polygon. Polygon must have >=3 points."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        # Check if edge crosses the ray to +inf in x direction
        intersects = ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1)
        if intersects:
            inside = not inside
    return inside


def _region_orientation(region: RegionConfig) -> str:
    # Used only for deciding IN/OUT direction.
    xs = [p[0] for p in region.points]
    ys = [p[1] for p in region.points]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    return "vertical" if width < height else "horizontal"


class ObjectCountingUserData(app_callback_class):
    def __init__(self):
        super().__init__()
        self.region: RegionConfig | None = None
        self.count_labels: set[str] | None = None

        self.in_count: int = 0
        self.out_count: int = 0

        # Per-track state
        self.prev_centroid: dict[int, tuple[float, float]] = {}
        self.prev_inside: dict[int, bool] = {}
        self.counted_line_ids: set[int] = set()


def app_callback(pad, info, user_data: ObjectCountingUserData):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    frame_idx = user_data.get_count()

    fmt, width, height = get_caps_from_pad(pad)
    if width is None or height is None:
        # Without frame dimensions we cannot map normalized bboxes to pixel coords.
        if frame_idx % 30 == 0:
            hailo_logger.warning("No caps on pad; cannot compute pixel centroids")
        return Gst.PadProbeReturn.OK

    frame = None
    if user_data.use_frame and fmt is not None:
        frame = get_numpy_from_buffer(buffer, fmt, width, height)

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    region = user_data.region
    if region is None:
        return Gst.PadProbeReturn.OK

    orientation = _region_orientation(region)

    # Optional region drawing
    if frame is not None:
        pts = list(region.points)
        if region.is_line:
            cv2.line(frame, pts[0], pts[1], (255, 0, 255), 2)
        else:
            cv2.polylines(
                frame,
                [np.array(pts, dtype=np.int32)],
                isClosed=True,
                color=(255, 0, 255),
                thickness=2,
            )

    # Count updates
    for det in detections:
        label = det.get_label()
        if user_data.count_labels is not None and label not in user_data.count_labels:
            continue

        bbox = det.get_bbox()
        cx = (bbox.xmin() + (bbox.width() / 2.0)) * float(width)
        cy = (bbox.ymin() + (bbox.height() / 2.0)) * float(height)
        centroid = (cx, cy)

        # Track id comes from HailoTracker metadata
        track_id = 0
        track_meta = det.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track_meta) == 1:
            track_id = int(track_meta[0].get_id())
        else:
            # Without a stable id, counting will double-count; skip.
            continue

        prev = user_data.prev_centroid.get(track_id)
        user_data.prev_centroid[track_id] = centroid

        if prev is None:
            # init inside state for polygon mode
            if region.is_polygon:
                user_data.prev_inside[track_id] = _point_in_polygon(centroid, region.points)
            continue

        if region.is_line:
            if track_id in user_data.counted_line_ids:
                continue
            p1, p2 = prev, centroid
            q1, q2 = (float(region.points[0][0]), float(region.points[0][1])), (
                float(region.points[1][0]),
                float(region.points[1][1]),
            )
            if _segment_intersects(p1, p2, q1, q2):
                if orientation == "vertical":
                    moved_in = centroid[0] > prev[0]
                else:
                    moved_in = centroid[1] > prev[1]

                if moved_in:
                    user_data.in_count += 1
                else:
                    user_data.out_count += 1

                user_data.counted_line_ids.add(track_id)

        else:
            # Polygon: count boundary crossings (outside->inside = IN, inside->outside = OUT)
            inside_now = _point_in_polygon(centroid, region.points)
            inside_prev = user_data.prev_inside.get(track_id, inside_now)
            user_data.prev_inside[track_id] = inside_now

            if inside_now == inside_prev:
                continue

            if inside_now and not inside_prev:
                user_data.in_count += 1
            elif inside_prev and not inside_now:
                user_data.out_count += 1

    # Draw overlay text
    if frame is not None:
        cv2.putText(
            frame,
            f"IN: {user_data.in_count}  OUT: {user_data.out_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    if frame_idx % 30 == 0:
        hailo_logger.info("Frame=%s | IN=%s OUT=%s", frame_idx, user_data.in_count, user_data.out_count)

    return Gst.PadProbeReturn.OK


def main():
    hailo_logger.info("Starting Hailo Object Counting App...")
    user_data = ObjectCountingUserData()
    app = GStreamerObjectCountingApp(app_callback, user_data)
    app.run()


if __name__ == "__main__":
    main()

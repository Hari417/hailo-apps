# region imports
# Standard library imports
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import setproctitle

from hailo_apps.hailo_app_python.core.common.core import get_default_parser, get_resource_path
from hailo_apps.hailo_app_python.core.common.defines import (
    DETECTION_POSTPROCESS_FUNCTION,
    DETECTION_POSTPROCESS_SO_FILENAME,
    DETECTION_PIPELINE,
    RESOURCES_MODELS_DIR_NAME,
    RESOURCES_SO_DIR_NAME,
)
from hailo_apps.hailo_app_python.core.common.hailo_logger import get_logger
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import GStreamerApp
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_helper_pipelines import (
    DISPLAY_PIPELINE,
    INFERENCE_PIPELINE,
    INFERENCE_PIPELINE_WRAPPER,
    SOURCE_PIPELINE,
    TRACKER_PIPELINE,
    USER_CALLBACK_PIPELINE,
)

hailo_logger = get_logger(__name__)
# endregion imports


@dataclass(frozen=True)
class RegionConfig:
    points: tuple[tuple[int, int], ...]

    @property
    def is_line(self) -> bool:
        return len(self.points) == 2

    @property
    def is_polygon(self) -> bool:
        return len(self.points) >= 3


def _parse_region_points(region: str) -> tuple[tuple[int, int], ...]:
    """Parse a region string into integer points.

    Supported formats (choose one):
      - "x1,y1;x2,y2" (semicolon separated)
      - "x1,y1 x2,y2" (space separated)

    Returns:
      Tuple of (x, y) integer points.

    Raises:
      ValueError if parsing fails.
    """
    if region is None:
        raise ValueError("region is None")

    raw = region.strip()
    if not raw:
        raise ValueError("region is empty")

    # Normalize delimiters to semicolon
    raw = raw.replace(" ", ";")
    parts = [p for p in raw.split(";") if p]
    points: list[tuple[int, int]] = []
    for part in parts:
        if "," not in part:
            raise ValueError(f"Invalid region point '{part}'. Expected 'x,y'.")
        xs, ys = part.split(",", 1)
        try:
            x = int(xs.strip())
            y = int(ys.strip())
        except ValueError as e:
            raise ValueError(f"Invalid region point '{part}'. x/y must be ints.") from e
        points.append((x, y))

    if len(points) < 2:
        raise ValueError("Region must have at least 2 points (a line).")

    return tuple(points)


def _parse_count_labels(labels: str | None) -> set[str] | None:
    if labels is None:
        return None
    stripped = labels.strip()
    if not stripped:
        return None
    return {part.strip() for part in stripped.split(",") if part.strip()}


class GStreamerObjectCountingApp(GStreamerApp):
    """Detection+tracking pipeline with a user callback intended for object counting.

    This uses the same inference resources as the detection pipeline and adds HailoTracker so
    detections include a stable `HAILO_UNIQUE_ID` for per-object counting.
    """

    def __init__(self, app_callback, user_data, parser=None):
        if parser is None:
            parser = get_default_parser()

        parser.add_argument(
            "--region",
            default="20,400;1080,400",
            help=(
                "Counting region points. Examples: '20,400;1080,400' (line) or "
                "'20,400;1080,400;1080,360;20,360' (polygon)."
            ),
        )
        parser.add_argument(
            "--count-labels",
            default=None,
            help="Comma-separated labels to count (e.g. 'person,car'). Default: count all tracked labels.",
        )
        parser.add_argument(
            "--tracker-class-id",
            type=int,
            default=1,
            help=(
                "Class ID for HailoTracker. Use -1 to track across all classes. "
                "For COCO, 'person' is usually 1."
            ),
        )

        hailo_logger.info("Initializing GStreamer Object Counting App...")
        super().__init__(parser, user_data)

        # Model resources (reuse detection pipeline resources)
        self.batch_size = 2
        nms_score_threshold = 0.3
        nms_iou_threshold = 0.45

        if self.options_menu.hef_path is not None:
            self.hef_path = self.options_menu.hef_path
        else:
            self.hef_path = get_resource_path(DETECTION_PIPELINE, RESOURCES_MODELS_DIR_NAME, self.arch)

        self.post_process_so = get_resource_path(
            DETECTION_PIPELINE,
            RESOURCES_SO_DIR_NAME,
            self.arch,
            DETECTION_POSTPROCESS_SO_FILENAME,
        )
        self.post_function_name = DETECTION_POSTPROCESS_FUNCTION

        if self.hef_path is None or not Path(self.hef_path).exists():
            hailo_logger.error("HEF path is invalid or missing: %s", self.hef_path)
        if self.post_process_so is None or not Path(self.post_process_so).exists():
            hailo_logger.error("Post-process .so path is invalid or missing: %s", self.post_process_so)

        self.tracker_class_id = int(self.options_menu.tracker_class_id)
        self.thresholds_str = (
            f"nms-score-threshold={nms_score_threshold} "
            f"nms-iou-threshold={nms_iou_threshold} "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )

        # Configure user_data from CLI
        try:
            points = _parse_region_points(self.options_menu.region)
            user_data.region = RegionConfig(points=points)
        except Exception as e:
            hailo_logger.error("Failed parsing --region '%s': %s", self.options_menu.region, e)
            raise

        user_data.count_labels = _parse_count_labels(self.options_menu.count_labels)

        self.app_callback = app_callback
        setproctitle.setproctitle("Hailo Object Counting")

        self.create_pipeline()

    def get_pipeline_string(self):
        source_pipeline = SOURCE_PIPELINE(
            video_source=self.video_source,
            video_width=self.video_width,
            video_height=self.video_height,
            frame_rate=self.frame_rate,
            sync=self.sync,
        )
        detection_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.post_function_name,
            batch_size=self.batch_size,
            config_json=None,
            additional_params=self.thresholds_str,
        )
        detection_pipeline_wrapper = INFERENCE_PIPELINE_WRAPPER(detection_pipeline)
        tracker_pipeline = TRACKER_PIPELINE(class_id=self.tracker_class_id)
        user_callback_pipeline = USER_CALLBACK_PIPELINE()
        display_pipeline = DISPLAY_PIPELINE(
            video_sink=self.video_sink,
            sync=self.sync,
            show_fps=self.show_fps,
        )

        pipeline_string = (
            f"{source_pipeline} ! "
            f"{detection_pipeline_wrapper} ! "
            f"{tracker_pipeline} ! "
            f"{user_callback_pipeline} ! "
            f"{display_pipeline}"
        )
        hailo_logger.debug("Pipeline string: %s", pipeline_string)
        return pipeline_string

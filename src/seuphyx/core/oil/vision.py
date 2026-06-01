"""
Backend scaffolding for future oil-drop video measurement.

The real tracker will be added after collecting representative videos. This
module keeps the data contract stable for the Streamlit page and later OpenCV
or model-based implementations.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class VisionMeasurementConfig:
    """Calibration and experiment metadata for one video measurement."""

    grid_size_mm: float = 0.25
    frame_rate: float = 30.0
    voltage_v: float = 0.0
    measurement_distance_mm: float = 0.25


@dataclass
class TrackedOilDropMeasurement:
    """Output expected from the future tracker."""

    start_time_s: float
    end_time_s: float
    displacement_mm: float
    velocity_mm_s: float
    equivalent_falling_time_s: float
    voltage_v: float


class OilDropVisionPipeline:
    """Placeholder pipeline for click-to-track oil-drop video measurement."""

    def __init__(self, config: VisionMeasurementConfig):
        self.config = config

    def load_video(self, video_path: Path):
        """Future hook for loading a camera stream or uploaded video."""
        raise NotImplementedError("油滴视频读取将在数据集采集后实现。")

    def calibrate_grid(self):
        """Future hook for detecting white grid lines and mm-per-pixel scale."""
        raise NotImplementedError("网格自动标定将在数据集采集后实现。")

    def track_from_click(self, start_point_px: tuple[int, int]):
        """Future hook for tracking the clicked oil drop."""
        raise NotImplementedError("点击选中油滴后的自动跟踪将在数据集采集后实现。")


def build_empty_measurement(config: VisionMeasurementConfig
                            ) -> TrackedOilDropMeasurement:
    """Create a record template matching the future tracker output."""
    return TrackedOilDropMeasurement(
        start_time_s=0.0,
        end_time_s=0.0,
        displacement_mm=0.0,
        velocity_mm_s=0.0,
        equivalent_falling_time_s=0.0,
        voltage_v=config.voltage_v,
    )

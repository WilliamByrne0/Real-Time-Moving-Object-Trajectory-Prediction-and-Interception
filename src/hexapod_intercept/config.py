"""Configuration management for the Hexapod Object Interception System.

All tunable parameters live here. Values can be overridden by YAML files in
``configs/`` or by environment variables prefixed with ``HEXAPOD_``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass(frozen=True)
class CameraConfig:
    width: int = 1280
    height: int = 720
    device_index: int = 1
    auto_exposure: int = 1            # 1=manual, 3=auto
    exposure: int = -6
    auto_wb: int = 0
    wb_temperature: int = 4600
    autofocus: int = 0
    focus: int = 0
    gain: int = 0
    brightness: int = 128
    contrast: int = 128


@dataclass(frozen=True)
class ArenaConfig:
    """Pixel-to-metre mapping derived from camera calibration."""
    x_start_m: float = -0.010812152843194799
    x_end_m: float = 0.46
    y_start_m: float = -0.19926
    y_end_m: float = -0.32166


@dataclass(frozen=True)
class RobotConfig:
    """UR3e kinematic and communication parameters."""
    xmlrpc_port: int = 50000
    rest_pose: tuple[float, float, float, float, float, float] = (
        0.100, 2.28, -2.161, 0.0, 0.0, 0.0
    )
    # Joint dynamics (UR3e defaults)
    q_max_rad_s: float = (32 / 180) * 3.141592653589793
    accel_rad_s2: float = (100 / 180) * 3.141592653589793
    # Empirical latency (IK compute + XML-RPC round-trip) in seconds
    measured_latency_s: float = 0.064


@dataclass(frozen=True)
class PredictionConfig:
    model: Literal["kalman", "lstm", "transformer", "linear"] = "kalman"
    prediction_length: int = 150
    history_length: int = 60
    fps: float = 25.223
    max_uncertainty: float = 9_999_999_990.0
    linear_smooth_frames: int = 60
    occlusions_enabled: bool = False
    kf_centre_mode: bool = False
    # Kalman noise
    kf_process_sigma: float = 12.0
    kf_sensor_sigma: float = 3.0


@dataclass(frozen=True)
class InterceptionConfig:
    min_time_s: float = 2.0
    max_time_s: float = 99.0
    grad_threshold: float = 0.50
    short_error_window: int = 100
    error_threshold: float = 1.0
    ball_radius_px: int = 15


@dataclass(frozen=True)
class RuntimeConfig:
    simulation: bool = False
    send_xmlrpc: bool = True
    debug: bool = False
    save_video: bool = False
    real_speed: bool = True
    video_path: str = r"Datasets/Bug_walk_8_03_2026/Cropped/bug (55)_cropped.mp4"
    output_path: str = r"Datasets/Bug_walk_8_03_2026/Output/bug_output.mp4"
    color_filter: Literal["blue"] = "blue"


@dataclass(frozen=True)
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    interception: InterceptionConfig = field(default_factory=InterceptionConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(
            camera=CameraConfig(**data.get("camera", {})),
            arena=ArenaConfig(**data.get("arena", {})),
            robot=RobotConfig(**data.get("robot", {})),
            prediction=PredictionConfig(**data.get("prediction", {})),
            interception=InterceptionConfig(**data.get("interception", {})),
            runtime=RuntimeConfig(**data.get("runtime", {})),
        )

    @classmethod
    def load(cls, profile: str | None = None) -> "Config":
        """Load config by profile name; fall back to defaults.

        Order of precedence:
          1. ``HEXAPOD_CONFIG`` env var pointing to a YAML file
          2. ``configs/<profile>.yaml`` if profile given
          3. ``configs/default.yaml`` if it exists
          4. Built-in defaults
        """
        env_path = os.environ.get("HEXAPOD_CONFIG")
        if env_path and Path(env_path).exists():
            return cls.from_yaml(env_path)
        if profile:
            p = Path("configs") / f"{profile}.yaml"
            if p.exists():
                return cls.from_yaml(p)
        default = Path("configs/default.yaml")
        if default.exists():
            return cls.from_yaml(default)
        return cls()

"""
Detector ABC and detection schema.

Every detector — YOLOv8, RT-DETR, classical OpenCV, mock — implements the
same single method: ``detect(frame) -> list[Detection]``. This keeps
detector implementations swappable behind one stable contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, model_validator


class DetectorError(RuntimeError):
    """Raised when a detector cannot process a frame."""


class ObjectClass(str, Enum):
    """High-level object classes the pipeline cares about."""

    BALL = "ball"
    PLAYER = "player"
    GOALKEEPER = "goalkeeper"
    REFEREE = "referee"
    UNKNOWN = "unknown"


class Detection(BaseModel):
    """
    A single detection in pixel-space.

    The pipeline applies homography downstream to convert ``(x_center,
    y_center)`` from pixels to FIFA pitch metres.
    """

    object_class: ObjectClass
    # Bounding box in pixel coordinates (x_min, y_min, x_max, y_max).
    x_min: float = Field(ge=0)
    y_min: float = Field(ge=0)
    x_max: float = Field(ge=0)
    y_max: float = Field(ge=0)
    score: float = Field(ge=0.0, le=1.0)
    # Free-form attributes (jersey number, team id, …) populated by downstream
    # classifiers; absent on raw detections.
    attributes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_box(self) -> Detection:
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("bounding box must have positive width and height")
        return self

    @property
    def center(self) -> tuple[float, float]:
        """Centre of the bounding box in pixel coordinates."""
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)

    @property
    def foot_point(self) -> tuple[float, float]:
        """
        Image-space "feet" of the detection — bottom-centre of the bbox.

        For player detections this is the canonical contact point to feed
        into the pitch-homography (you want the player's feet on the
        pitch, not their head).
        """
        return ((self.x_min + self.x_max) / 2.0, self.y_max)


class Detector(ABC):
    """Common detector contract."""

    name: str = "Detector"

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a single frame; return zero or more Detections."""

    # Optional method: subclasses with model state can override.
    def warmup(self) -> None:
        """Optionally pre-load model weights so the first inference is hot."""
        return None

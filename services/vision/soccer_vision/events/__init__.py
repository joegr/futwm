"""Bridge from vision detections to canonical ``soccer_model`` events."""

from __future__ import annotations

from .emitter import EmitterConfig, EmitterError, EventEmitter

__all__ = ["EmitterConfig", "EmitterError", "EventEmitter"]

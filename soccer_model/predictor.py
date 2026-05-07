"""
EventPredictor — high-level interface for next-event prediction.

Wraps TransitionModel to provide:
  * single-step prediction with probability rankings
  * beam search over the k most likely event-type sequences
  * xG accumulation helpers
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .events import AnyEvent, EventType
from .stochastic import EventDistribution, TransitionModel
from .world_model import WorldState

Beam = list[tuple[float, list[AnyEvent]]]   # (log_prob, event_sequence)


@dataclass
class Prediction:
    """
    Output of a single prediction step.

    Attributes
    ----------
    distribution    : EventDistribution    Full probability distribution.
    top_events      : List[AnyEvent]       Sampled representatives of the top-k types.
    top_probs       : List[float]          Corresponding probabilities.
    expected_xg     : float                Expected goals contribution this step.
    """

    distribution:  EventDistribution
    top_events:    list[AnyEvent]
    top_probs:     list[float]
    expected_xg:   float

    def most_likely(self) -> AnyEvent:
        """Return the highest-probability sampled event."""
        return self.top_events[0]

    def summary(self) -> str:
        lines = ["Next-event prediction:"]
        for ev, p in zip(self.top_events, self.top_probs):
            lines.append(f"  [{p:.1%}] {ev.event_type.value:22s} @ ({ev.pitch_x:.1f}, {ev.pitch_y:.1f})")
        lines.append(f"  Expected xG this step: {self.expected_xg:.4f}")
        return "\n".join(lines)


class EventPredictor:
    """
    High-level prediction interface.

    Parameters
    ----------
    pitch : Pitch
    seed  : Optional[int]   RNG seed for reproducibility.
    """

    def __init__(self, pitch, seed: int | None = None) -> None:
        self.pitch = pitch
        self.model = TransitionModel(pitch, seed=seed)

    # ── single-step prediction ────────────────────────────────────────────────

    def predict_next(self, state: WorldState, top_k: int = 3) -> Prediction:
        """
        Predict the most likely next events for the given world state.

        Samples one concrete event per top-k event type and returns them
        ranked by probability.
        """
        dist        = self.model.predict(state)
        top_types   = dist.top_k(top_k)

        top_events: list[AnyEvent] = []
        top_probs:  list[float]    = []

        for et, p in top_types:
            dest  = dist.sample_destination(self.model.rng)
            dest  = self.pitch.clamp(*dest)
            event = self.model._build_event(state, et, dest, dist)
            top_events.append(event)
            top_probs.append(p)

        xg = self._expected_xg(dist, state)
        return Prediction(
            distribution=dist,
            top_events=top_events,
            top_probs=top_probs,
            expected_xg=xg,
        )

    # ── sequence prediction ───────────────────────────────────────────────────

    def predict_sequence(
        self,
        state: WorldState,
        horizon: int = 5,
        samples: int = 20,
    ) -> list[list[AnyEvent]]:
        """
        Sample `samples` independent event sequences of length `horizon`.

        Each sequence is a possible future from the current world state.
        The world state is cloned before each rollout so the original is
        not modified.

        Returns
        -------
        List of sampled sequences, each a list of `horizon` events.
        """
        sequences = []
        for _ in range(samples):
            clone = state.snapshot()
            seq   = []
            for _ in range(horizon):
                event = self.model.sample(clone)
                clone.apply_event(event)
                seq.append(event)
                if not clone.ball.in_play:
                    break
            sequences.append(seq)
        return sequences

    def beam_search(
        self,
        state: WorldState,
        horizon: int = 4,
        beam_width: int = 3,
    ) -> list[tuple[float, list[EventType]]]:
        """
        Beam search over event-type sequences (not full events).

        Returns top `beam_width` sequences as (log_probability, [EventType, ...]).
        """
        dist  = self.model.predict(state)
        beams: list[tuple[float, list[EventType], WorldState]] = [
            (math.log(p + 1e-12), [et], state.snapshot())
            for et, p in dist.top_k(beam_width)
        ]

        for step in range(1, horizon):
            candidates: list[tuple[float, list[EventType], WorldState]] = []
            for log_p, seq, s in beams:
                dist_s = self.model.predict(s)
                for et, p in dist_s.top_k(beam_width):
                    dest  = dist_s.sample_destination(self.model.rng)
                    dest  = self.pitch.clamp(*dest)
                    event = self.model._build_event(s, et, dest, dist_s)
                    s2    = s.snapshot()
                    s2.apply_event(event)
                    candidates.append((log_p + math.log(p + 1e-12), seq + [et], s2))
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_width]

        return [(lp, seq) for lp, seq, _ in beams]

    # ── xG helpers ────────────────────────────────────────────────────────────

    def _expected_xg(self, dist: EventDistribution, state: WorldState) -> float:
        shot_p = 0.0
        for et, p in zip(dist.event_types, dist.probs):
            if et == EventType.SHOT:
                shot_p = p
                break
        feats      = state.feature_vector()
        xg_if_shot = float(np.clip(
            1.0 / (1.0 + math.exp(
                0.08 * feats["dist_to_goal"] - 1.2 * feats["goal_angle"] + 0.5
            )),
            0.02, 0.75,
        ))
        return shot_p * xg_if_shot

    def cumulative_xg(
        self,
        sequences: list[list[AnyEvent]],
    ) -> dict[str, float]:
        """
        Aggregate xG from a list of sampled sequences.

        Returns
        -------
        Dict with keys ``"mean_xg"``, ``"max_xg"``, ``"min_xg"``, ``"std_xg"``.
        """
        xg_values: list[float] = []
        for seq in sequences:
            total = sum(
                getattr(e, "xg", 0.0)
                for e in seq
                if e.event_type == EventType.SHOT
            )
            xg_values.append(total)

        arr = np.array(xg_values)
        return {
            "mean_xg": float(arr.mean()),
            "max_xg":  float(arr.max()),
            "min_xg":  float(arr.min()),
            "std_xg":  float(arr.std()),
        }

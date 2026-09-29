"""Short-horizon confirmation without requiring external speed telemetry."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from lidar_geometry.detect_obstacles import Obstacle
from lidar_geometry.fast_detector import SafetyDetection


@dataclass(frozen=True)
class TemporalDecision:
    state: str
    obstacle: bool
    nearest_distance_m: float | None
    confidence: float
    confirmations: int
    reason: str


class TemporalConfirmer:
    """Require a spatially compatible alarm in two of the last three frames.

    At the stated worst-case 85 km/h and 10 Hz this adds at most ~0.2 s / 4.7 m
    for an ordinary candidate. Very strong close candidates bypass the delay.
    Matching allows up to 3.5 m longitudinal change per frame, so speed is
    inferred only as a permissive compatibility range, not a required input.
    """

    def __init__(
        self,
        window: int = 3,
        required: int = 2,
        max_distance_step_m: float = 3.5,
        immediate_confidence: float = 0.88,
        immediate_distance_m: float = 15.0,
    ):
        self.history: deque[tuple[float | None, float]] = deque(maxlen=window)
        self.required = required
        self.max_distance_step_m = max_distance_step_m
        self.immediate_confidence = immediate_confidence
        self.immediate_distance_m = immediate_distance_m

    def update(self, detection: SafetyDetection) -> TemporalDecision:
        current = (
            detection.nearest_distance_m if detection.state == "OBSTACLE" else None
        )
        self.history.append((current, detection.confidence))
        if current is None:
            return TemporalDecision(
                detection.state,
                False,
                detection.nearest_distance_m,
                detection.confidence,
                0,
                detection.reason,
            )
        compatible = 0
        for age, (distance, _confidence) in enumerate(reversed(self.history)):
            if distance is None:
                continue
            expected_extra = age * self.max_distance_step_m
            if current - 1.0 <= distance <= current + expected_extra + 1.0:
                compatible += 1
        immediate = detection.confidence >= self.immediate_confidence or (
            current <= self.immediate_distance_m and detection.confidence >= 0.70
        )
        confirmed = compatible >= self.required or immediate
        if confirmed:
            return TemporalDecision(
                "OBSTACLE",
                True,
                current,
                detection.confidence,
                compatible,
                "immediate_high_risk" if immediate else "confirmed_2_of_3_frames",
            )
        return TemporalDecision(
            "UNKNOWN",
            False,
            current,
            detection.confidence,
            compatible,
            "awaiting_temporal_confirmation",
        )


@dataclass
class _ComponentTrack:
    distance_m: float
    lateral_m: float
    height_m: float
    velocity_m_per_frame: float
    hits: int
    first_frame: int
    last_frame: int
    confidence: float
    hit_frames: deque[int] = field(default_factory=deque)


def _centers(item: Obstacle) -> tuple[float, float, float]:
    return (
        (item.distance_min_m + item.distance_max_m) / 2,
        (item.lateral_min_m + item.lateral_max_m) / 2,
        (item.height_min_m + item.height_max_m) / 2,
    )


class ComponentTracker:
    def __init__(
        self,
        required_hits: int = 2,
        max_missed_frames: int = 2,
        max_distance_step_m: float = 3.5,
        max_lateral_step_m: float = 0.45,
        max_height_step_m: float = 0.75,
        max_confirmed_velocity_m_per_frame: float | None = None,
        confirmation_window_frames: int | None = None,
    ):
        self.required_hits = required_hits
        self.max_missed_frames = max_missed_frames
        self.max_distance_step_m = max_distance_step_m
        self.max_lateral_step_m = max_lateral_step_m
        self.max_height_step_m = max_height_step_m
        self.max_confirmed_velocity_m_per_frame = max_confirmed_velocity_m_per_frame
        self.confirmation_window_frames = confirmation_window_frames
        self.frame = -1
        self.tracks: list[_ComponentTrack] = []

    def update(self, detection: SafetyDetection) -> TemporalDecision:
        self.frame += 1
        candidates = list(detection.obstacles) if detection.state == "OBSTACLE" else []
        pairs = []
        for track_index, track in enumerate(self.tracks):
            gap = self.frame - track.last_frame
            predicted_distance = track.distance_m + track.velocity_m_per_frame * gap
            for candidate_index, candidate in enumerate(candidates):
                distance, lateral, height = _centers(candidate)
                distance_error = abs(distance - predicted_distance)
                lateral_error = abs(lateral - track.lateral_m)
                height_error = abs(height - track.height_m)
                if (
                    distance_error <= self.max_distance_step_m * gap + 0.5
                    and lateral_error <= self.max_lateral_step_m
                    and height_error <= self.max_height_step_m
                ):
                    cost = (
                        distance_error / self.max_distance_step_m
                        + lateral_error / self.max_lateral_step_m
                        + height_error / self.max_height_step_m
                    )
                    pairs.append((cost, track_index, candidate_index))

        used_tracks = set()
        used_candidates = set()
        for _, track_index, candidate_index in sorted(pairs):
            if track_index in used_tracks or candidate_index in used_candidates:
                continue
            track = self.tracks[track_index]
            distance, lateral, height = _centers(candidates[candidate_index])
            gap = self.frame - track.last_frame
            velocity = (distance - track.distance_m) / gap
            track.velocity_m_per_frame = (
                0.5 * track.velocity_m_per_frame + 0.5 * velocity
            )
            track.distance_m = distance
            track.lateral_m = lateral
            track.height_m = height
            track.last_frame = self.frame
            track.hits += 1
            track.hit_frames.append(self.frame)
            track.confidence = max(track.confidence, detection.confidence)
            used_tracks.add(track_index)
            used_candidates.add(candidate_index)

        for index, candidate in enumerate(candidates):
            if index in used_candidates:
                continue
            distance, lateral, height = _centers(candidate)
            self.tracks.append(
                _ComponentTrack(
                    distance,
                    lateral,
                    height,
                    0.0,
                    1,
                    self.frame,
                    self.frame,
                    detection.confidence,
                    deque([self.frame]),
                )
            )

        self.tracks = [
            track
            for track in self.tracks
            if self.frame - track.last_frame <= self.max_missed_frames
        ]
        confirmed = [
            track
            for track in self.tracks
            if (
                track.last_frame == self.frame
                and self._hits_in_window(track) >= self.required_hits
                and (
                    self.max_confirmed_velocity_m_per_frame is None
                    or track.velocity_m_per_frame
                    <= self.max_confirmed_velocity_m_per_frame
                )
            )
        ]
        if confirmed:
            nearest = min(confirmed, key=lambda track: track.distance_m)
            return TemporalDecision(
                "OBSTACLE",
                True,
                nearest.distance_m,
                nearest.confidence,
                self._hits_in_window(nearest),
                "range_lateral_component_track",
            )
        if candidates:
            return TemporalDecision(
                "UNKNOWN",
                False,
                detection.nearest_distance_m,
                detection.confidence,
                1,
                "awaiting_component_track",
            )
        return TemporalDecision(
            detection.state,
            False,
            detection.nearest_distance_m,
            detection.confidence,
            0,
            detection.reason,
        )

    def _hits_in_window(self, track: _ComponentTrack) -> int:
        if self.confirmation_window_frames is None:
            return track.hits
        first = self.frame - self.confirmation_window_frames + 1
        return sum(frame >= first for frame in track.hit_frames)


@dataclass
class _WorldTrack:
    world_m: float
    lateral_m: float
    height_m: float
    distance_m: float
    hits: int
    last_frame: int
    confidence: float
    hit_frames: deque[int] = field(default_factory=deque)


class WorldComponentTracker:
    def __init__(
        self,
        required_hits: int = 2,
        max_missed_frames: int = 2,
        max_world_step_m: float = 0.8,
        max_lateral_step_m: float = 0.45,
        max_height_step_m: float = 0.75,
        confirmation_window_frames: int | None = None,
    ):
        self.required_hits = required_hits
        self.max_missed_frames = max_missed_frames
        self.max_world_step_m = max_world_step_m
        self.max_lateral_step_m = max_lateral_step_m
        self.max_height_step_m = max_height_step_m
        self.confirmation_window_frames = confirmation_window_frames
        self.frame = -1
        self.tracks: list[_WorldTrack] = []

    def update(
        self, detection: SafetyDetection, cumulative_m: float
    ) -> TemporalDecision:
        self.frame += 1
        candidates = list(detection.obstacles) if detection.state == "OBSTACLE" else []
        centers = []
        for candidate in candidates:
            distance, lateral, height = _centers(candidate)
            centers.append((cumulative_m + distance, lateral, height, distance))

        pairs = []
        for track_index, track in enumerate(self.tracks):
            gap = self.frame - track.last_frame
            for candidate_index, (world, lateral, height, _distance) in enumerate(
                centers
            ):
                world_error = abs(world - track.world_m)
                lateral_error = abs(lateral - track.lateral_m)
                height_error = abs(height - track.height_m)
                if (
                    world_error <= self.max_world_step_m * gap
                    and lateral_error <= self.max_lateral_step_m
                    and height_error <= self.max_height_step_m
                ):
                    cost = (
                        world_error / self.max_world_step_m
                        + lateral_error / self.max_lateral_step_m
                        + height_error / self.max_height_step_m
                    )
                    pairs.append((cost, track_index, candidate_index))

        used_tracks = set()
        used_candidates = set()
        for _, track_index, candidate_index in sorted(pairs):
            if track_index in used_tracks or candidate_index in used_candidates:
                continue
            track = self.tracks[track_index]
            world, lateral, height, distance = centers[candidate_index]
            track.world_m = world
            track.lateral_m = lateral
            track.height_m = height
            track.distance_m = distance
            track.hits += 1
            track.hit_frames.append(self.frame)
            track.last_frame = self.frame
            track.confidence = max(track.confidence, detection.confidence)
            used_tracks.add(track_index)
            used_candidates.add(candidate_index)

        for index, (world, lateral, height, distance) in enumerate(centers):
            if index in used_candidates:
                continue
            self.tracks.append(
                _WorldTrack(
                    world,
                    lateral,
                    height,
                    distance,
                    1,
                    self.frame,
                    detection.confidence,
                    deque([self.frame]),
                )
            )
        self.tracks = [
            track
            for track in self.tracks
            if self.frame - track.last_frame <= self.max_missed_frames
        ]
        confirmed = [
            track
            for track in self.tracks
            if (
                track.last_frame == self.frame
                and self._hits_in_window(track) >= self.required_hits
            )
        ]
        if confirmed:
            nearest = min(confirmed, key=lambda track: track.distance_m)
            return TemporalDecision(
                "OBSTACLE",
                True,
                nearest.distance_m,
                nearest.confidence,
                self._hits_in_window(nearest),
                "ego_motion_consistent_world_track",
            )
        if candidates:
            return TemporalDecision(
                "UNKNOWN",
                False,
                detection.nearest_distance_m,
                detection.confidence,
                1,
                "awaiting_world_track",
            )
        return TemporalDecision(
            detection.state,
            False,
            detection.nearest_distance_m,
            detection.confidence,
            0,
            detection.reason,
        )

    def _hits_in_window(self, track: _WorldTrack) -> int:
        if self.confirmation_window_frames is None:
            return track.hits
        first = self.frame - self.confirmation_window_frames + 1
        return sum(frame >= first for frame in track.hit_frames)

from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np

from lidar_geometry.detect_obstacles import Obstacle
from lidar_geometry.fast_detector import SafetyDetection


def center(item: Obstacle) -> tuple[float, float, float]:
    return (
        (item.distance_min_m + item.distance_max_m) / 2,
        (item.lateral_min_m + item.lateral_max_m) / 2,
        (item.height_min_m + item.height_max_m) / 2,
    )


@dataclass
class EvidenceTrack:
    longitudinal_m: float
    lateral_m: float
    height_m: float
    velocity_m_per_frame: float
    last_frame: int
    observations: deque[tuple[int, float]] = field(default_factory=deque)


class EvidenceMemory:
    def __init__(
        self,
        threshold: float,
        world_coordinates: bool,
        score_floor_ratio: float = 0.75,
        required_observations: int = 2,
        window_frames: int = 3,
        max_missed_frames: int = 2,
        max_world_step_m: float = 0.8,
        max_distance_step_m: float = 3.5,
        max_lateral_step_m: float = 0.45,
        max_height_step_m: float = 0.75,
    ):
        self.threshold = threshold
        self.world_coordinates = world_coordinates
        self.score_floor = threshold * score_floor_ratio
        self.required_observations = required_observations
        self.window_frames = window_frames
        self.max_missed_frames = max_missed_frames
        self.max_world_step_m = max_world_step_m
        self.max_distance_step_m = max_distance_step_m
        self.max_lateral_step_m = max_lateral_step_m
        self.max_height_step_m = max_height_step_m
        self.frame = -1
        self.tracks: list[EvidenceTrack] = []

    def update(
        self,
        geometric: SafetyDetection,
        scores: np.ndarray,
        cumulative_m: float = 0.0,
    ) -> SafetyDetection:
        self.frame += 1
        first_frame = self.frame - self.window_frames + 1
        self.tracks = [
            track
            for track in self.tracks
            if self.frame - track.last_frame <= self.max_missed_frames
        ]
        for track in self.tracks:
            while track.observations and track.observations[0][0] < first_frame:
                track.observations.popleft()

        items = list(geometric.obstacles)
        if len(items) != len(scores):
            raise ValueError("candidate and score counts differ")
        candidates = []
        for index, (item, score) in enumerate(zip(items, scores)):
            if score < self.score_floor:
                continue
            distance, lateral, height = center(item)
            longitudinal = (
                cumulative_m + distance if self.world_coordinates else distance
            )
            candidates.append((index, longitudinal, lateral, height, float(score)))

        pairs = []
        for track_index, track in enumerate(self.tracks):
            gap = self.frame - track.last_frame
            predicted = track.longitudinal_m + (
                0.0 if self.world_coordinates else track.velocity_m_per_frame * gap
            )
            longitudinal_limit = (
                self.max_world_step_m * gap
                if self.world_coordinates
                else self.max_distance_step_m * gap + 0.5
            )
            for candidate_index, (_, longitudinal, lateral, height, _) in enumerate(
                candidates
            ):
                longitudinal_error = abs(longitudinal - predicted)
                lateral_error = abs(lateral - track.lateral_m)
                height_error = abs(height - track.height_m)
                if (
                    longitudinal_error <= longitudinal_limit
                    and lateral_error <= self.max_lateral_step_m
                    and height_error <= self.max_height_step_m
                ):
                    cost = (
                        longitudinal_error / longitudinal_limit
                        + lateral_error / self.max_lateral_step_m
                        + height_error / self.max_height_step_m
                    )
                    pairs.append((cost, track_index, candidate_index))

        track_for_candidate = {}
        used_tracks = set()
        used_candidates = set()
        for _, track_index, candidate_index in sorted(pairs):
            if track_index in used_tracks or candidate_index in used_candidates:
                continue
            _, longitudinal, lateral, height, score = candidates[candidate_index]
            track = self.tracks[track_index]
            gap = self.frame - track.last_frame
            if not self.world_coordinates:
                velocity = (longitudinal - track.longitudinal_m) / gap
                track.velocity_m_per_frame = (
                    0.5 * track.velocity_m_per_frame + 0.5 * velocity
                )
            track.longitudinal_m = longitudinal
            track.lateral_m = lateral
            track.height_m = height
            track.last_frame = self.frame
            track.observations.append((self.frame, score))
            track_for_candidate[candidate_index] = track_index
            used_tracks.add(track_index)
            used_candidates.add(candidate_index)

        for candidate_index, (_, longitudinal, lateral, height, score) in enumerate(
            candidates
        ):
            if candidate_index in used_candidates:
                continue
            self.tracks.append(
                EvidenceTrack(
                    longitudinal,
                    lateral,
                    height,
                    0.0,
                    self.frame,
                    deque([(self.frame, score)]),
                )
            )
            track_for_candidate[candidate_index] = len(self.tracks) - 1

        accepted = []
        confidence = []
        for candidate_index, (item_index, _, _, _, _) in enumerate(candidates):
            track = self.tracks[track_for_candidate[candidate_index]]
            values = [score for _, score in track.observations]
            if (
                len(values) >= self.required_observations
                and max(values) >= self.threshold
            ):
                accepted.append(items[item_index])
                confidence.append(max(values))

        if accepted:
            nearest = min(accepted, key=lambda item: item.distance_min_m)
            return replace(
                geometric,
                state="OBSTACLE",
                obstacle=True,
                nearest_distance_m=nearest.distance_min_m,
                confidence=max(confidence),
                obstacles=tuple(accepted),
                reason="online_evidence_memory",
            )
        if items:
            return replace(
                geometric,
                state="UNKNOWN",
                obstacle=False,
                nearest_distance_m=None,
                confidence=float(np.max(scores)),
                obstacles=(),
                reason="awaiting_online_memory_confirmation",
            )
        return geometric


class WorldEvidenceMemory(EvidenceMemory):
    def __init__(self, threshold: float, **kwargs):
        super().__init__(threshold, world_coordinates=True, **kwargs)


class RangeEvidenceMemory(EvidenceMemory):
    def __init__(self, threshold: float, **kwargs):
        super().__init__(threshold, world_coordinates=False, **kwargs)

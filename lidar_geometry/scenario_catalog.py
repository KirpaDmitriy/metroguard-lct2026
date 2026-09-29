from __future__ import annotations

from dataclasses import dataclass

from lidar_geometry.detect_obstacles import Obstacle
from lidar_geometry.synthetic import SyntheticObject


@dataclass(frozen=True)
class Scenario:
    name: str
    expected_alarm: bool
    obstacle: SyntheticObject


SCENARIOS = (
    Scenario("large_center", True, SyntheticObject("box", 20, 0, 2, 2, 2)),
    Scenario("minimum_center", True, SyntheticObject("box", 20, 0, 0.3, 0.3, 0.1)),
    Scenario("minimum_on_rail", True, SyntheticObject("box", 20, 0.76, 0.3, 0.3, 0.1)),
    Scenario("minimum_edge_inside", True, SyntheticObject("box", 20, 0.89, 0.3, 0.3, 0.1)),
    Scenario("minimum_just_outside", False, SyntheticObject("box", 20, 1.21, 0.3, 0.3, 0.1)),
    Scenario("large_edge_inside", True, SyntheticObject("box", 20, 0.04, 2, 2, 2)),
    Scenario("large_outside", False, SyntheticObject("box", 20, 2.06, 2, 2, 2)),
    Scenario("large_above", False, SyntheticObject("box", 20, 0, 2, 2, 2, bottom_m=3.01)),
    Scenario("long_low_on_rails", True, SyntheticObject("box", 20, 0, 2, 0.2, 0.2)),
    Scenario("hanging_cable", True, SyntheticObject("box", 20, 0, 0.05, 0.05, 2, bottom_m=1)),
    Scenario("trough_below_rail", False, SyntheticObject("box", 20, 0, 0.3, 0.3, 0.1, bottom_m=-0.25)),
)


def matches_target(component: Obstacle, obstacle: SyntheticObject, tolerance_m: float = 0.005) -> bool:
    target_d_min = obstacle.distance_m - obstacle.length_m / 2
    target_d_max = obstacle.distance_m + obstacle.length_m / 2
    target_x_min = obstacle.lateral_m - obstacle.width_m / 2
    target_x_max = obstacle.lateral_m + obstacle.width_m / 2
    return (
        component.distance_max_m >= target_d_min - tolerance_m
        and component.distance_min_m <= target_d_max + tolerance_m
        and component.lateral_max_m >= target_x_min - tolerance_m
        and component.lateral_min_m <= target_x_max + tolerance_m
    )

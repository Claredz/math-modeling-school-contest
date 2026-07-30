"""Exact certificates for ship-relative radius and pairwise UAV separation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smoke_defense.paths import ShipborneUavPath


@dataclass(frozen=True)
class PathConstraintCertificate:
    status: str
    minimum_value: float | None = None
    maximum_value: float | None = None
    critical_time_s: float | None = None
    reason: str = ""


def certify_operation_radius(
    path: ShipborneUavPath,
    *,
    operation_radius_m: float,
    tolerance_m: float = 1e-9,
) -> PathConstraintCertificate:
    """Certify a linear path against the uniformly moving ship.

    Relative position is affine on each segment. Its norm is convex, so the
    maximum over a closed segment occurs at an endpoint.
    """

    if operation_radius_m <= 0:
        raise ValueError("operation radius must be positive")
    endpoint_times = {
        path.takeoff_time_s,
        *(segment.end_time_s for segment in path.segments),
    }
    distances = [
        (
            time_s,
            float(np.linalg.norm(path.position(time_s) - path.ship.position(time_s))),
        )
        for time_s in endpoint_times
    ]
    critical_time_s, maximum_distance = max(distances, key=lambda item: item[1])
    feasible = maximum_distance <= operation_radius_m + tolerance_m
    return PathConstraintCertificate(
        status="certified_feasible" if feasible else "certified_infeasible",
        maximum_value=maximum_distance,
        critical_time_s=critical_time_s,
        reason=(
            "all segment endpoint maxima satisfy the moving-ship radius"
            if feasible
            else "moving-ship operation radius exceeded"
        ),
    )


def _closest_time_on_affine_interval(
    relative_position_m: np.ndarray,
    relative_velocity_mps: np.ndarray,
    duration_s: float,
) -> float:
    speed_squared = float(relative_velocity_mps @ relative_velocity_mps)
    if speed_squared == 0:
        return 0.0
    unconstrained = -float(relative_position_m @ relative_velocity_mps) / (
        speed_squared
    )
    return float(np.clip(unconstrained, 0.0, duration_s))


def certify_pairwise_separation(
    path_a: ShipborneUavPath,
    path_b: ShipborneUavPath,
    *,
    safe_distance_m: float,
    tolerance_m: float = 1e-9,
) -> PathConstraintCertificate:
    """Check exact closest points while both UAVs are airborne."""

    if safe_distance_m < 0:
        raise ValueError("safe distance cannot be negative")
    overlap_start = max(path_a.takeoff_time_s, path_b.takeoff_time_s)
    overlap_end = min(path_a.end_time_s, path_b.end_time_s)
    if overlap_end < overlap_start:
        return PathConstraintCertificate(
            status="certified_feasible",
            reason="UAV airborne intervals do not overlap",
        )

    breakpoints = {overlap_start, overlap_end}
    for path in (path_a, path_b):
        for segment in path.segments:
            if overlap_start < segment.end_time_s < overlap_end:
                breakpoints.add(segment.end_time_s)
    ordered = sorted(breakpoints)

    candidates: list[tuple[float, float]] = []
    if len(ordered) == 1:
        distance = float(
            np.linalg.norm(
                path_a.position(overlap_start) - path_b.position(overlap_start)
            )
        )
        candidates.append((overlap_start, distance))
    for left_s, right_s in zip(ordered[:-1], ordered[1:], strict=True):
        midpoint_s = 0.5 * (left_s + right_s)
        relative_position = path_a.position(left_s) - path_b.position(left_s)
        relative_velocity = (
            path_a.velocity(midpoint_s) - path_b.velocity(midpoint_s)
        )
        offset_s = _closest_time_on_affine_interval(
            relative_position,
            relative_velocity,
            right_s - left_s,
        )
        time_s = left_s + offset_s
        distance = float(
            np.linalg.norm(path_a.position(time_s) - path_b.position(time_s))
        )
        candidates.append((time_s, distance))

    critical_time_s, minimum_distance = min(candidates, key=lambda item: item[1])
    feasible = minimum_distance + tolerance_m >= safe_distance_m
    return PathConstraintCertificate(
        status="certified_feasible" if feasible else "certified_infeasible",
        minimum_value=minimum_distance,
        critical_time_s=critical_time_s,
        reason=(
            "pairwise safe distance holds while both UAVs are airborne"
            if feasible
            else "pairwise safe distance violated"
        ),
    )

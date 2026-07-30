"""Analytic single-smoke and bounded polygon union-coverage certificates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import cos, pi

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union


class CertificationStatus(StrEnum):
    CERTIFIED_FEASIBLE = "certified_feasible"
    CERTIFIED_INFEASIBLE = "certified_infeasible"
    INDETERMINATE = "indeterminate_at_tolerance"


@dataclass(frozen=True)
class Disk:
    center_m: np.ndarray
    radius_m: float

    def __post_init__(self) -> None:
        center = np.asarray(self.center_m, dtype=float).copy()
        if center.shape != (2,):
            raise ValueError("disk centre must be two-dimensional")
        if self.radius_m < 0:
            raise ValueError("disk radius cannot be negative")
        object.__setattr__(self, "center_m", center)


@dataclass(frozen=True)
class CoverageCertificate:
    status: CertificationStatus
    polygon_sides: int | None = None
    witness_m: np.ndarray | None = None
    reason: str = ""


def single_smoke_gap(
    ship_center_m: np.ndarray,
    smoke_center_m: np.ndarray,
    smoke_radius_m: float,
    *,
    ship_radius_m: float = 80.0,
) -> float:
    """Exact gap; nonpositive is equivalent to full disk containment."""

    separation = np.linalg.norm(
        np.asarray(ship_center_m, dtype=float)
        - np.asarray(smoke_center_m, dtype=float)
    )
    return float(separation + ship_radius_m - smoke_radius_m)


def _regular_polygon(
    disk: Disk,
    *,
    sides: int,
    outer: bool,
) -> Polygon:
    radius = disk.radius_m / cos(pi / sides) if outer else disk.radius_m
    angles = np.linspace(0.0, 2.0 * pi, sides, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles))) * radius
    points += disk.center_m
    return Polygon(points)


def _exact_uncovered_witness(
    point_m: np.ndarray,
    target: Disk,
    covers: tuple[Disk, ...],
    tolerance_m: float,
) -> bool:
    inside_target = (
        np.linalg.norm(point_m - target.center_m)
        <= target.radius_m + tolerance_m
    )
    outside_all_covers = all(
        np.linalg.norm(point_m - cover.center_m)
        > cover.radius_m + tolerance_m
        for cover in covers
    )
    return bool(inside_target and outside_all_covers)


def certify_union_coverage(
    target: Disk,
    covers: tuple[Disk, ...],
    *,
    initial_polygon_sides: int = 32,
    maximum_polygon_sides: int = 2048,
    witness_tolerance_m: float = 1e-8,
) -> CoverageCertificate:
    """Certify disk containment in a union using inner/outer regular polygons."""

    if initial_polygon_sides < 4:
        raise ValueError("polygon approximation needs at least four sides")
    if maximum_polygon_sides < initial_polygon_sides:
        raise ValueError("maximum polygon sides is too small")
    active_covers = tuple(cover for cover in covers if cover.radius_m > 0)
    if not active_covers:
        return CoverageCertificate(
            CertificationStatus.CERTIFIED_INFEASIBLE,
            witness_m=target.center_m.copy(),
            reason="no positive-radius smoke disk is active",
        )

    sides = initial_polygon_sides
    while sides <= maximum_polygon_sides:
        target_outer = _regular_polygon(target, sides=sides, outer=True)
        cover_inner = unary_union(
            [
                _regular_polygon(cover, sides=sides, outer=False)
                for cover in active_covers
            ]
        )
        if cover_inner.covers(target_outer):
            return CoverageCertificate(
                CertificationStatus.CERTIFIED_FEASIBLE,
                polygon_sides=sides,
                reason="outer target polygon is covered by inner smoke polygons",
            )

        target_inner = _regular_polygon(target, sides=sides, outer=False)
        cover_outer = unary_union(
            [
                _regular_polygon(cover, sides=sides, outer=True)
                for cover in active_covers
            ]
        )
        uncovered = target_inner.difference(cover_outer)
        if not uncovered.is_empty:
            representative = uncovered.representative_point()
            witness = np.array([representative.x, representative.y])
            if _exact_uncovered_witness(
                witness,
                target,
                active_covers,
                witness_tolerance_m,
            ):
                return CoverageCertificate(
                    CertificationStatus.CERTIFIED_INFEASIBLE,
                    polygon_sides=sides,
                    witness_m=witness,
                    reason="exact point witness lies in target and outside all smokes",
                )
        sides *= 2

    return CoverageCertificate(
        CertificationStatus.INDETERMINATE,
        polygon_sides=maximum_polygon_sides,
        reason="inner/outer polygon bounds overlap at requested tolerance",
    )

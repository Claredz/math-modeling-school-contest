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


@dataclass(frozen=True)
class UnionGapEvaluation:
    status: CertificationStatus
    lower_bound_m: float
    upper_bound_m: float
    estimated_gap_m: float
    spatial_error_m: float
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


def _offset_disks(
    covers: tuple[Disk, ...],
    offset_m: float,
) -> tuple[Disk, ...]:
    return tuple(
        Disk(cover.center_m, cover.radius_m + offset_m)
        for cover in covers
        if cover.radius_m + offset_m > 0.0
    )


def evaluate_union_gap_at_time(
    target: Disk,
    covers: tuple[Disk, ...],
    *,
    spatial_tolerance_m: float = 0.05,
    initial_polygon_sides: int = 32,
    maximum_polygon_sides: int = 2048,
    witness_tolerance_m: float = 1e-8,
) -> UnionGapEvaluation:
    """Bound the exact joint gap using certified radius-offset decisions.

    Increasing every smoke radius by ``delta`` covers the target exactly when
    the unmodified joint gap is no greater than ``delta``. Certified decisions
    therefore provide a monotone bisection oracle for the gap.
    """

    if spatial_tolerance_m <= 0:
        raise ValueError("spatial tolerance must be positive")
    active_covers = tuple(cover for cover in covers if cover.radius_m > 0.0)
    if not active_covers:
        return UnionGapEvaluation(
            status=CertificationStatus.CERTIFIED_INFEASIBLE,
            lower_bound_m=float("inf"),
            upper_bound_m=float("inf"),
            estimated_gap_m=float("inf"),
            spatial_error_m=0.0,
            witness_m=target.center_m.copy(),
            reason="no positive-radius smoke disk is active",
        )
    if len(active_covers) == 1:
        separation_vector = target.center_m - active_covers[0].center_m
        separation = float(np.linalg.norm(separation_vector))
        witness_direction = (
            separation_vector / separation
            if separation > 0.0
            else np.array([1.0, 0.0])
        )
        gap_m = single_smoke_gap(
            target.center_m,
            active_covers[0].center_m,
            active_covers[0].radius_m,
            ship_radius_m=target.radius_m,
        )
        return UnionGapEvaluation(
            status=(
                CertificationStatus.CERTIFIED_FEASIBLE
                if gap_m <= 0.0
                else CertificationStatus.CERTIFIED_INFEASIBLE
            ),
            lower_bound_m=gap_m,
            upper_bound_m=gap_m,
            estimated_gap_m=gap_m,
            spatial_error_m=0.0,
            witness_m=(
                None
                if gap_m <= 0.0
                else target.center_m
                + target.radius_m * witness_direction
            ),
            reason="single-smoke gap is analytic",
        )

    def certify(offset_m: float) -> CoverageCertificate:
        return certify_union_coverage(
            target,
            _offset_disks(active_covers, offset_m),
            initial_polygon_sides=initial_polygon_sides,
            maximum_polygon_sides=maximum_polygon_sides,
            witness_tolerance_m=witness_tolerance_m,
        )

    zero_certificate = certify(0.0)
    point_values = [
        min(
            float(np.linalg.norm(point - cover.center_m) - cover.radius_m)
            for cover in active_covers
        )
        for point in (
            target.center_m,
            *(
                target.center_m
                + target.radius_m
                * np.array([cos(angle), np.sin(angle)])
                for angle in np.linspace(0.0, 2.0 * pi, 16, endpoint=False)
            ),
        )
    ]
    lower_bound_m = max(point_values)
    upper_bound_m = min(
        single_smoke_gap(
            target.center_m,
            cover.center_m,
            cover.radius_m,
            ship_radius_m=target.radius_m,
        )
        for cover in active_covers
    )
    witness_m = zero_certificate.witness_m
    polygon_sides = zero_certificate.polygon_sides

    if zero_certificate.status is CertificationStatus.CERTIFIED_FEASIBLE:
        upper_bound_m = min(upper_bound_m, 0.0)
    elif zero_certificate.status is CertificationStatus.CERTIFIED_INFEASIBLE:
        lower_bound_m = max(lower_bound_m, 0.0)

    while upper_bound_m - lower_bound_m > spatial_tolerance_m:
        midpoint_m = 0.5 * (lower_bound_m + upper_bound_m)
        certificate = certify(midpoint_m)
        polygon_sides = max(
            polygon_sides or 0,
            certificate.polygon_sides or 0,
        )
        if certificate.status is CertificationStatus.CERTIFIED_FEASIBLE:
            upper_bound_m = midpoint_m
        elif certificate.status is CertificationStatus.CERTIFIED_INFEASIBLE:
            lower_bound_m = midpoint_m
            if certificate.witness_m is not None:
                witness_m = certificate.witness_m
        else:
            break

    spatial_error_m = upper_bound_m - lower_bound_m
    return UnionGapEvaluation(
        status=zero_certificate.status,
        lower_bound_m=lower_bound_m,
        upper_bound_m=upper_bound_m,
        estimated_gap_m=0.5 * (lower_bound_m + upper_bound_m),
        spatial_error_m=spatial_error_m,
        polygon_sides=polygon_sides,
        witness_m=(
            witness_m
            if zero_certificate.status
            is CertificationStatus.CERTIFIED_INFEASIBLE
            else None
        ),
        reason=(
            f"{zero_certificate.reason}; radius-offset gap bracket "
            f"width={spatial_error_m:.6g} m"
        ),
    )

from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from smoke_defense.coverage import CertificationStatus
from smoke_defense.detection import DetectionSet
from smoke_defense.dynamics import ShipMotion
from smoke_defense.events import ClosedInterval
from smoke_defense.paths import (
    ReleaseWaypoint,
    build_ordered_release_path,
)
from smoke_defense.q2 import (
    MultiSmokeCandidate,
    MultiSmokePlan,
    OrderedReleasePlan,
    PathNode,
    SmokeReleaseEvent,
    branch_and_bound_q2_combinations,
    enumerate_q2_combinations,
    generate_q2_candidate_library,
    prune_q2_candidates,
    q2_plan_rank_key,
    select_best_q2_plan,
)
from smoke_defense.smoke import detonation_position
from smoke_defense.verification import certify_multi_smoke_coverage


def make_release_event(
    release_time_s: float,
    *,
    release_x_m: float | None = None,
) -> SmokeReleaseEvent:
    release_x = (
        28.0 * release_time_s if release_x_m is None else release_x_m
    )
    return SmokeReleaseEvent(
        candidate_id=f"candidate-{release_time_s}",
        command_time_s=0.0,
        release_time_s=release_time_s,
        release_position_m=(release_x, 0.0),
        release_heading_unit=(1.0, 0.0),
        burst_time_s=release_time_s + 3.5,
        burst_center_m=(release_x + 98.0, 0.0),
    )


def make_ordered_plan(
    release_times_s: tuple[float, ...],
) -> OrderedReleasePlan:
    releases = tuple(make_release_event(time_s) for time_s in release_times_s)
    nodes = (
        PathNode(time_s=0.0, position_m=(0.0, 0.0)),
        *(
            PathNode(
                time_s=event.release_time_s,
                position_m=event.release_position_m,
            )
            for event in releases
        ),
        PathNode(
            time_s=releases[-1].burst_time_s,
            position_m=(
                releases[-1].burst_center_m[0],
                releases[-1].burst_center_m[1],
            ),
        ),
    )
    return OrderedReleasePlan(
        takeoff_time_s=0.0,
        takeoff_position_m=(0.0, 0.0),
        path_nodes=nodes,
        releases=releases,
        adjacent_release_intervals_s=tuple(
            right - left
            for left, right in zip(
                release_times_s[:-1],
                release_times_s[1:],
                strict=True,
            )
        ),
        flight_distance_m=28.0 * releases[-1].burst_time_s,
        continue_until_s=releases[-1].burst_time_s,
    )


def test_q2_models_are_frozen_and_reject_extra_fields():
    node = PathNode(time_s=0.0, position_m=(0.0, 0.0))

    with pytest.raises(ValidationError):
        node.time_s = 1.0
    with pytest.raises(ValidationError):
        PathNode(time_s=0.0, position_m=(0.0, 0.0), unexpected=True)
    assert MultiSmokePlan.model_config["frozen"] is True
    assert MultiSmokePlan.model_config["extra"] == "forbid"


def test_ordered_release_plan_accepts_one_to_three_smokes():
    assert len(make_ordered_plan((10.0,)).releases) == 1
    assert len(make_ordered_plan((10.0, 11.0)).releases) == 2
    assert len(make_ordered_plan((10.0, 11.0, 12.0)).releases) == 3

    four_releases = tuple(make_release_event(10.0 + index) for index in range(4))
    with pytest.raises(ValidationError, match="at most 3"):
        OrderedReleasePlan(
            takeoff_time_s=0.0,
            takeoff_position_m=(0.0, 0.0),
            path_nodes=(
                PathNode(time_s=0.0, position_m=(0.0, 0.0)),
                PathNode(time_s=16.5, position_m=(462.0, 0.0)),
            ),
            releases=four_releases,
            adjacent_release_intervals_s=(1.0, 1.0, 1.0),
            flight_distance_m=462.0,
            continue_until_s=16.5,
        )


def test_release_spacing_exactly_one_second_is_legal():
    plan = make_ordered_plan((10.0, 11.0))

    assert plan.adjacent_release_intervals_s == (1.0,)


def test_release_spacing_below_one_second_is_illegal():
    with pytest.raises(ValidationError, match="at least 1 s"):
        make_ordered_plan((10.0, 10.999))


def test_ordered_path_is_continuous_fixed_speed_and_visits_releases():
    ship = ShipMotion((0.0, 0.0), heading_rad=0.0, speed_mps=0.0)
    releases = (
        ReleaseWaypoint(10.0, np.array([280.0, 0.0]), np.array([1.0, 0.0])),
        ReleaseWaypoint(12.0, np.array([336.0, 0.0]), np.array([1.0, 0.0])),
    )

    path = build_ordered_release_path(
        ship=ship,
        takeoff_time_s=0.0,
        releases=releases,
        continue_until_s=15.5,
    )

    assert path.end_time_s == pytest.approx(15.5)
    assert path.flight_distance_m == pytest.approx(28.0 * 15.5)
    assert all(segment.speed_mps == pytest.approx(28.0) for segment in path.segments)
    for release in releases:
        assert path.position(release.release_time_s) == pytest.approx(
            release.release_position_m
        )
        burst = detonation_position(
            release.release_position_m,
            path.velocity(release.release_time_s),
        )
        assert burst == pytest.approx(
            release.release_position_m + 98.0 * release.heading_unit
        )


def test_takeoff_wait_is_shipborne_and_not_counted_as_flight_distance():
    ship = ShipMotion((0.0, 0.0), heading_rad=0.0, speed_mps=7.71)
    takeoff_time_s = 5.0
    release_time_s = 15.0
    release_position = ship.position(takeoff_time_s) + np.array([280.0, 0.0])

    path = build_ordered_release_path(
        ship=ship,
        takeoff_time_s=takeoff_time_s,
        releases=(
            ReleaseWaypoint(
                release_time_s,
                release_position,
                np.array([1.0, 0.0]),
            ),
        ),
        continue_until_s=18.5,
    )

    assert path.position(3.0) == pytest.approx(ship.position(3.0))
    assert path.position(takeoff_time_s) == pytest.approx(
        ship.position(takeoff_time_s)
    )
    assert path.flight_distance_m == pytest.approx(
        28.0 * (18.5 - takeoff_time_s)
    )


def test_individually_reachable_releases_can_be_jointly_unlinkable():
    ship = ShipMotion((0.0, 0.0), heading_rad=0.0, speed_mps=0.0)
    releases = (
        ReleaseWaypoint(10.0, np.array([280.0, 0.0]), np.array([1.0, 0.0])),
        ReleaseWaypoint(11.0, np.array([-308.0, 0.0]), np.array([-1.0, 0.0])),
    )

    with pytest.raises(ValueError, match="unreachable"):
        build_ordered_release_path(
            ship=ship,
            takeoff_time_s=0.0,
            releases=releases,
            continue_until_s=14.5,
        )


def make_candidate_library():
    ship = ShipMotion((0.0, 0.0), heading_rad=0.0, speed_mps=7.71)
    detection = DetectionSet(
        components=(ClosedInterval(10.0, 20.0),),
        source_events=(),
    )
    library = generate_q2_candidate_library(
        ship=ship,
        detection=detection,
        uav_available_time_s=0.0,
        scenario_id="synthetic-q2",
        scenario_hash="scenario-hash",
        constants_hash="constants-hash",
        assumption_ids=("A-001", "A-022"),
        guidance_model="inertial_pure_pursuit",
        model_layer="formal",
        heading_response_rate_per_s=1.0,
        max_turn_rate_deg_s=10.0,
    )
    return ship, library


def test_q2_candidate_library_preserves_spatial_and_time_diversity():
    ship, library = make_candidate_library()

    assert len(library.candidates) > 3
    assert len(
        {round(candidate.coverage_center_time_s, 6) for candidate in library.candidates}
    ) > 1
    assert len(
        {round(candidate.takeoff_time_s, 6) for candidate in library.candidates}
    ) > 1
    assert any(
        abs(candidate.lateral_offset_m) > 0.0 for candidate in library.candidates
    )
    assert any(
        not np.allclose(
            candidate.release.burst_center_m,
            ship.position(candidate.coverage_center_time_s),
        )
        for candidate in library.candidates
    )
    assert all(
        candidate.scenario_hash == "scenario-hash"
        and candidate.constants_hash == "constants-hash"
        for candidate in library.candidates
    )


def test_individually_incomplete_candidates_survive_q2_pruning():
    _ship, library = make_candidate_library()
    incomplete = tuple(
        candidate
        for candidate in library.candidates
        if candidate.single_coverage_status.value != "certified_feasible"
    )

    assert incomplete
    pruning = prune_q2_candidates(incomplete)

    assert pruning.retained_candidates
    assert any(
        candidate.single_coverage_status.value != "certified_feasible"
        for candidate in pruning.retained_candidates
    )


def test_q2_pruning_removes_only_duplicate_physical_events():
    _ship, library = make_candidate_library()
    original = library.candidates[0]
    duplicate = original.model_copy(update={"candidate_id": "duplicate-id"})

    pruning = prune_q2_candidates((original, duplicate, library.candidates[1]))

    assert pruning.input_count == 3
    assert pruning.duplicate_count == 1
    assert pruning.retained_count == 2
    assert library.candidates[1] in pruning.retained_candidates


def make_joint_only_candidates() -> tuple[MultiSmokeCandidate, ...]:
    common_release = np.array([0.0, np.sqrt(98.0**2 - 50.0**2)])
    candidates = []
    for index, (center_x_m, release_time_s) in enumerate(
        ((-50.0, 10.0), (50.0, 11.0))
    ):
        burst_center = np.array([center_x_m, 0.0])
        heading = (burst_center - common_release) / 98.0
        release = SmokeReleaseEvent(
            candidate_id=f"joint-{index}",
            command_time_s=0.0,
            release_time_s=release_time_s,
            release_position_m=tuple(common_release),
            release_heading_unit=tuple(heading),
            burst_time_s=release_time_s + 3.5,
            burst_center_m=tuple(burst_center),
        )
        candidates.append(
            MultiSmokeCandidate(
                candidate_id=release.candidate_id,
                scenario_id="joint-only",
                scenario_hash="joint-hash",
                constants_hash="constants-hash",
                assumption_ids=("A-001", "A-022"),
                guidance_model="inertial_pure_pursuit",
                model_layer="formal",
                heading_response_rate_per_s=1.0,
                max_turn_rate_deg_s=10.0,
                takeoff_time_s=0.0,
                coverage_center_time_s=15.0,
                longitudinal_offset_m=0.0,
                lateral_offset_m=center_x_m,
                release=release,
                maximum_radius_m=110.0,
                hold_duration_s=18.0,
                decay_duration_s=5.0,
                path_start_position_m=(0.0, 0.0),
                path_end_position_m=tuple(common_release),
                reachability_status="certified_feasible",
                single_coverage_status=CertificationStatus.CERTIFIED_INFEASIBLE,
                covered_duration_s=0.0,
                covered_intervals_s=(),
                exposed_duration_s=3.0,
                maximum_exposed_interval_s=3.0,
                minimum_margin_m=-20.0,
                flight_distance_m=28.0 * release_time_s,
            )
        )
    return tuple(candidates)


def test_enumeration_checks_all_one_and_two_smoke_combinations():
    candidates = make_joint_only_candidates()
    ship = ShipMotion((0.0, 0.0), heading_rad=0.0, speed_mps=0.0)
    detection = DetectionSet(
        components=(ClosedInterval(14.5, 17.5),),
        source_events=(),
    )
    verifier_calls = 0

    def counting_verifier(**kwargs):
        nonlocal verifier_calls
        verifier_calls += 1
        return certify_multi_smoke_coverage(**kwargs)

    result = enumerate_q2_combinations(
        ship=ship,
        detection=detection,
        candidates=candidates,
        operation_radius_m=12000.0,
        verifier=counting_verifier,
    )

    assert result.enumerated_count == 3
    assert verifier_calls == result.retained_count
    assert result.best_plan is not None
    assert result.best_plan.smoke_count == 2


def test_true_spatial_union_beats_independent_interval_baseline():
    candidates = make_joint_only_candidates()
    result = enumerate_q2_combinations(
        ship=ShipMotion((0.0, 0.0), heading_rad=0.0, speed_mps=0.0),
        detection=DetectionSet(
            components=(ClosedInterval(14.5, 17.5),),
            source_events=(),
        ),
        candidates=candidates,
        operation_radius_m=12000.0,
    )

    plan = result.best_plan
    assert plan is not None
    assert plan.coverage_certificate.status is CertificationStatus.CERTIFIED_FEASIBLE
    assert plan.independent_coverage_baseline.covered_duration_s == 0.0
    assert plan.independent_coverage_baseline.union_gain_s == pytest.approx(3.0)


def test_path_incompatible_combination_is_rejected_before_verification():
    candidates = list(make_joint_only_candidates())
    second = candidates[1]
    bad_release = second.release.model_copy(
        update={
            "release_position_m": (-308.0, 0.0),
            "release_heading_unit": (-1.0, 0.0),
            "burst_center_m": (-406.0, 0.0),
        }
    )
    candidates[1] = second.model_copy(update={"release": bad_release})

    result = enumerate_q2_combinations(
        ship=ShipMotion((0.0, 0.0), heading_rad=0.0, speed_mps=0.0),
        detection=DetectionSet(
            components=(ClosedInterval(14.5, 17.5),),
            source_events=(),
        ),
        candidates=tuple(candidates),
        operation_radius_m=12000.0,
    )

    assert any(
        evaluation.status == "rejected"
        and "unreachable" in evaluation.reason
        for evaluation in result.evaluations
        if len(evaluation.candidate_ids) == 2
    )


def test_combination_rechecks_release_to_burst_geometry():
    candidate = make_joint_only_candidates()[0]
    invalid_release = candidate.release.model_copy(
        update={"burst_center_m": (999.0, 999.0)}
    )
    invalid_candidate = candidate.model_copy(update={"release": invalid_release})

    result = enumerate_q2_combinations(
        ship=ShipMotion((0.0, 0.0), heading_rad=0.0, speed_mps=0.0),
        detection=DetectionSet(
            components=(ClosedInterval(14.5, 17.5),),
            source_events=(),
        ),
        candidates=(invalid_candidate,),
        operation_radius_m=12000.0,
    )

    assert result.retained_count == 0
    assert "detonation geometry" in result.evaluations[0].reason


def test_q2_lexicographic_order_cannot_be_overridden_by_shorter_distance():
    lower_exposure = SimpleNamespace(
        coverage_certificate=SimpleNamespace(
            status=CertificationStatus.CERTIFIED_INFEASIBLE
        ),
        maximum_exposed_interval_s=1.0,
        minimum_joint_margin_m=-5.0,
        smoke_count=3,
        uav_total_distance_m=1000.0,
    )
    shorter_but_worse = SimpleNamespace(
        coverage_certificate=SimpleNamespace(
            status=CertificationStatus.CERTIFIED_INFEASIBLE
        ),
        maximum_exposed_interval_s=2.0,
        minimum_joint_margin_m=10.0,
        smoke_count=1,
        uav_total_distance_m=10.0,
    )

    assert select_best_q2_plan((lower_exposure, shorter_but_worse)) is lower_exposure


def test_q2_lexicographic_ties_use_margin_then_bombs_then_distance():
    base = {
        "coverage_certificate": SimpleNamespace(
            status=CertificationStatus.CERTIFIED_INFEASIBLE
        ),
        "maximum_exposed_interval_s": 1.0,
    }
    plans = (
        SimpleNamespace(
            **base,
            minimum_joint_margin_m=0.0,
            smoke_count=2,
            uav_total_distance_m=100.0,
        ),
        SimpleNamespace(
            **base,
            minimum_joint_margin_m=1.0,
            smoke_count=3,
            uav_total_distance_m=200.0,
        ),
        SimpleNamespace(
            **base,
            minimum_joint_margin_m=1.0,
            smoke_count=2,
            uav_total_distance_m=300.0,
        ),
        SimpleNamespace(
            **base,
            minimum_joint_margin_m=1.0,
            smoke_count=2,
            uav_total_distance_m=250.0,
        ),
    )

    assert select_best_q2_plan(plans) is plans[3]


def test_q2_rank_ignores_sub_nanosecond_exposure_noise():
    common = {
        "coverage_certificate": SimpleNamespace(
            status=CertificationStatus.CERTIFIED_INFEASIBLE
        ),
        "minimum_joint_margin_m": 0.0,
        "smoke_count": 2,
    }
    shorter = SimpleNamespace(
        **common,
        maximum_exposed_interval_s=1.0,
        uav_total_distance_m=200.0,
    )
    noisy_longer = SimpleNamespace(
        **common,
        maximum_exposed_interval_s=1.0 + 4e-10,
        uav_total_distance_m=100.0,
    )

    assert q2_plan_rank_key(noisy_longer) > q2_plan_rank_key(shorter)


def test_branch_and_bound_matches_small_exhaustive_enumeration():
    candidates = make_joint_only_candidates()
    kwargs = {
        "ship": ShipMotion((0.0, 0.0), heading_rad=0.0, speed_mps=0.0),
        "detection": DetectionSet(
            components=(ClosedInterval(14.5, 17.5),),
            source_events=(),
        ),
        "candidates": candidates,
        "operation_radius_m": 12000.0,
    }

    exhaustive = enumerate_q2_combinations(**kwargs)
    bounded = branch_and_bound_q2_combinations(**kwargs)

    assert exhaustive.best_plan is not None
    assert bounded.best_plan is not None
    assert (
        exhaustive.best_plan.selected_smokes
        == bounded.best_plan.selected_smokes
    )

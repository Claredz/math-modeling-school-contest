import numpy as np

from smoke_defense.scenario_matrix import (
    generate_instantaneous_ablation_matrix,
    generate_q1_q3_matrix,
    generate_q1_rebuild_matrix,
)


def test_q1_q3_formal_matrix_has_144_scenes():
    scenes = generate_q1_q3_matrix()

    assert len(scenes) == 4 * 4 * 3 * 3
    assert len({scene.scenario_id for scene in scenes}) == len(scenes)
    assert all(scene.model_layer == "formal" for scene in scenes)


def test_ablation_matrix_has_16_scenes():
    scenes = generate_instantaneous_ablation_matrix()

    assert len(scenes) == 16
    assert len({scene.scenario_id for scene in scenes}) == len(scenes)
    assert all(scene.model_layer == "ablation" for scene in scenes)


def test_q1_rebuild_formal_baseline_is_four_by_four_without_inertial_parameters():
    scenes = generate_q1_rebuild_matrix()

    assert len(scenes) == 4 * 4
    assert len({scene.scenario_id for scene in scenes}) == len(scenes)
    assert all(scene.model_layer == "formal_baseline" for scene in scenes)
    assert all(
        scene.missiles[0].guidance_model == "instantaneous_pure_pursuit"
        and scene.missiles[0].heading_response_rate_per_s is None
        and scene.missiles[0].max_turn_rate_deg_s is None
        for scene in scenes
    )


def test_matrix_contains_expected_distances_and_directions():
    scenes = generate_q1_q3_matrix()
    positions = [
        np.asarray(
            scene.missiles[0].initial_position_at_appearance_body_m,
            dtype=float,
        )
        for scene in scenes
    ]
    distances = {round(float(np.linalg.norm(position))) for position in positions}
    unit_directions = {
        tuple(np.round(position / np.linalg.norm(position), 8))
        for position in positions
    }

    assert distances == {8000, 10000, 12000, 15000}
    assert unit_directions == {
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        tuple(np.round([np.cos(np.deg2rad(135)), np.sin(np.deg2rad(135))], 8)),
    }

"""Publication-ready figures for the Q1 lost-coupled model."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from smoke_defense.angles import wrap_to_pi
from smoke_defense.lost_guidance import GuidanceMode
from smoke_defense.q1_lost import LostQ1Candidate

MODE_COLORS = {
    GuidanceMode.PRELOCK: "#8c8c8c",
    GuidanceMode.TRACKED: "#d62728",
    GuidanceMode.LOST: "#1f77b4",
}


def plot_trajectory(
    candidate: LostQ1Candidate,
    *,
    output_path: str | Path,
    title: str = "Q1 lost-coupled defense trajectory",
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    trajectory = candidate.trajectory
    ship = candidate.path.ship
    ship_positions = np.asarray([ship.position(t) for t in trajectory.times_s])
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2), constrained_layout=True)
    engagement_mask = trajectory.times_s <= min(35.0, trajectory.times_s[-1])
    for axis, active_mask, subtitle in (
        (axes[0], engagement_mask, "Engagement detail (first 35 s)"),
        (axes[1], np.ones_like(engagement_mask, dtype=bool), "Full post-loss escape"),
    ):
        axis.plot(
            ship_positions[active_mask, 0], ship_positions[active_mask, 1],
            color="#2ca02c", lw=2.2, label="Ship trajectory",
        )
        for mode in GuidanceMode:
            mode_mask = np.asarray([item is mode for item in trajectory.modes])
            mask = mode_mask & active_mask
            if np.any(mask):
                axis.scatter(
                    trajectory.states[mask, 0], trajectory.states[mask, 1],
                    s=5, color=MODE_COLORS[mode], label=f"Missile: {mode.value}",
                )
        axis.scatter(
            *candidate.release_position_m, marker="v", s=70,
            color="#9467bd", label="Release",
        )
        axis.scatter(
            *candidate.burst_center_m, marker="*", s=130,
            color="#ff7f0e", label="Burst",
        )
        axis.add_patch(
            Circle(
                candidate.burst_center_m, 120.0, fill=False,
                ls="--", lw=1.5, color="#ff7f0e",
            )
        )
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_xlabel("x / m")
        axis.set_ylabel("y / m")
        axis.set_title(subtitle)
        axis.grid(alpha=0.25)
    axes[0].legend(loc="best", fontsize=7)
    fig.suptitle(title)
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def plot_timeline(
    candidate: LostQ1Candidate,
    *,
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    trajectory = candidate.trajectory
    ship = candidate.path.ship
    separations = []
    fov_errors_deg = []
    for time_s, state in zip(trajectory.times_s, trajectory.states, strict=True):
        relative = ship.position(time_s) - state[:2]
        separations.append(np.linalg.norm(relative))
        los = np.arctan2(relative[1], relative[0])
        fov_errors_deg.append(np.degrees(abs(wrap_to_pi(los - state[2]))))
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.6), sharex=True, constrained_layout=True)
    axes[0].plot(trajectory.times_s, separations, color="#222222", lw=1.7)
    axes[0].axhline(8000.0, color="#ff7f0e", ls="--", label="Detection range")
    axes[0].axhline(80.0, color="#d62728", ls=":", label="Hit radius")
    axes[0].set_ylabel("Separation / m")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].plot(trajectory.times_s, fov_errors_deg, color="#1f77b4", lw=1.7)
    axes[1].axhline(15.0, color="#d62728", ls="--", label="FOV boundary")
    axes[1].axvline(candidate.burst_time_s, color="#ff7f0e", ls=":", label="Burst")
    for mode in GuidanceMode:
        mask = np.asarray([item is mode for item in trajectory.modes])
        if np.any(mask):
            indices = np.flatnonzero(mask)
            axes[1].axvspan(
                trajectory.times_s[indices[0]], trajectory.times_s[indices[-1]],
                color=MODE_COLORS[mode], alpha=0.08,
            )
    axes[1].set_xlabel("Time / s")
    axes[1].set_ylabel("LOS error / deg")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def plot_feasibility_heatmap(
    success_matrix: np.ndarray,
    *,
    direction_degrees: tuple[float, ...],
    heading_errors_deg: tuple[float, ...],
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    image = axis.imshow(
        success_matrix,
        origin="lower",
        aspect="auto",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
    )
    axis.set_xticks(range(len(direction_degrees)), [f"{value:g}" for value in direction_degrees])
    axis.set_yticks(range(len(heading_errors_deg)), [f"{value:g}" for value in heading_errors_deg])
    axis.set_xlabel("Missile bearing / deg")
    axis.set_ylabel("Initial heading error / deg")
    axis.set_title("Permanent-loss feasibility map (green = successful)")
    fig.colorbar(image, ax=axis, ticks=[0, 1], label="Defense success")
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output

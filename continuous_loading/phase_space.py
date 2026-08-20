"""Transient particle-ensemble helpers for optional phase-space continuity.

Particle arrays are deliberately not embedded in public JSON result dataclasses.
They are passed only inside a single calculation and reduced to the existing
temperature/atom-number summaries at output boundaries.  The default scalar
pipeline therefore keeps its memory use, serialization and GPU batching intact.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ParticleEnsemble:
    """A weighted classical phase-space sample in an explicitly named frame."""

    positions_m: object
    velocities_m_s: object
    weights: object | None = None
    site_index: object | None = None
    frame: str = "local"

    def host_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = np.asarray(self.positions_m, dtype=float)
        velocities = np.asarray(self.velocities_m_s, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("相空间 positions_m 必须为 (N,3) 数组")
        if velocities.shape != positions.shape:
            raise ValueError("相空间 velocities_m_s 必须与 positions_m 同形")
        if positions.shape[0] == 0:
            raise ValueError("相空间集合不能为空")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
            raise ValueError("相空间位置和速度必须全部为有限数")
        if self.weights is None:
            weights = np.ones(positions.shape[0], dtype=float)
        else:
            weights = np.asarray(self.weights, dtype=float)
            if weights.shape != (positions.shape[0],):
                raise ValueError("相空间 weights 必须为长度 N 的一维数组")
            if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
                raise ValueError("相空间 weights 必须为有限正数")
        return positions, velocities, weights

    @property
    def particle_count(self) -> int:
        return int(np.asarray(self.positions_m).shape[0])

    def resampled(self, count: int, seed: int) -> "ParticleEnsemble":
        """Return a low-variance equal-weight systematic resample.

        Equal-size stage interfaces are copied exactly.  This avoids injecting
        an artificial temperature jump merely because a phase-space ensemble
        crosses a module boundary; unequal sizes use systematic rather than
        multinomial resampling, preserving empirical moments much better.
        """
        if count <= 0:
            raise ValueError("相空间重采样粒子数必须为正")
        positions, velocities, weights = self.host_arrays()
        if count == positions.shape[0] and np.allclose(weights, weights[0]):
            return ParticleEnsemble(
                positions_m=positions.copy(),
                velocities_m_s=velocities.copy(),
                weights=np.ones(count, dtype=float),
                site_index=(
                    None
                    if self.site_index is None
                    else np.asarray(self.site_index).copy()
                ),
                frame=self.frame,
            )
        probability = weights / weights.sum()
        rng = np.random.default_rng(int(seed))
        positions_u = (np.arange(count, dtype=float) + rng.random()) / count
        indices = np.searchsorted(np.cumsum(probability), positions_u, side="right")
        indices = np.minimum(indices, positions.shape[0] - 1)
        site_index = (
            None
            if self.site_index is None
            else np.asarray(self.site_index)[indices].copy()
        )
        return ParticleEnsemble(
            positions_m=positions[indices].copy(),
            velocities_m_s=velocities[indices].copy(),
            weights=np.ones(count, dtype=float),
            site_index=site_index,
            frame=self.frame,
        )

    def translated(self, offset_m) -> "ParticleEnsemble":
        positions, velocities, weights = self.host_arrays()
        offset = np.asarray(offset_m, dtype=float)
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise ValueError("相空间平移量必须为有限三维向量")
        return ParticleEnsemble(
            positions_m=positions + offset,
            velocities_m_s=velocities.copy(),
            weights=weights.copy(),
            site_index=None if self.site_index is None else np.asarray(self.site_index).copy(),
            frame=self.frame,
        )

    def rotated(self, matrix, *, frame: str) -> "ParticleEnsemble":
        positions, velocities, weights = self.host_arrays()
        rotation = np.asarray(matrix, dtype=float)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError("相空间旋转矩阵必须为有限 3x3 数组")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-8):
            raise ValueError("相空间旋转矩阵必须正交")
        if not math.isclose(abs(float(np.linalg.det(rotation))), 1.0, abs_tol=1e-8):
            raise ValueError("相空间旋转矩阵行列式绝对值必须为 1")
        return ParticleEnsemble(
            positions_m=positions @ rotation.T,
            velocities_m_s=velocities @ rotation.T,
            weights=weights.copy(),
            site_index=None if self.site_index is None else np.asarray(self.site_index).copy(),
            frame=frame,
        )


def canonicalize_lattice_phase(
    ensemble: ParticleEnsemble,
    *,
    phase_rad,
    wave_number_m: float,
    axis,
    beam_offset_m=(0.0, 0.0, 0.0),
    lattice_displacement_m: float = 0.0,
    lattice_velocity_m_s: float = 0.0,
    frame: str,
) -> ParticleEnsemble:
    """Map cos²(k(q-s)+phi) to a stationary zero-phase lattice.

    For each trajectory the exact canonical coordinate is
    q' = q - s + phi/k. Transverse coordinates are measured from the
    receiving beam axis, and velocities are transformed to the lattice
    co-moving frame. The operation is vectorised over particles and therefore
    adds only an O(N) stage-boundary copy, not work inside GPU time-step loops.
    """
    if not math.isfinite(wave_number_m) or wave_number_m <= 0.0:
        raise ValueError("晶格波数必须是有限正数")
    if not math.isfinite(lattice_displacement_m):
        raise ValueError("晶格位移必须是有限数")
    if not math.isfinite(lattice_velocity_m_s):
        raise ValueError("晶格速度必须是有限数")
    positions, velocities, weights = ensemble.host_arrays()
    direction = np.asarray(axis, dtype=float)
    offset = np.asarray(beam_offset_m, dtype=float)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError("晶格轴必须是有限三维向量")
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ValueError("晶格轴不能为零向量")
    direction /= norm
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise ValueError("晶格束偏移必须是有限三维向量")
    phase = np.asarray(phase_rad, dtype=float)
    if phase.ndim == 0:
        phase = np.full(positions.shape[0], float(phase))
    if phase.shape != (positions.shape[0],):
        raise ValueError("逐粒子晶格相位必须为长度 N 的一维数组")
    if not np.all(np.isfinite(phase)):
        raise ValueError("晶格相位必须全部为有限数")
    axial_shift = phase / wave_number_m - lattice_displacement_m
    canonical_positions = (
        positions - offset + axial_shift[:, None] * direction
    )
    canonical_velocities = (
        velocities - lattice_velocity_m_s * direction
    )
    return ParticleEnsemble(
        positions_m=canonical_positions,
        velocities_m_s=canonical_velocities,
        weights=weights.copy(),
        site_index=(
            None
            if ensemble.site_index is None
            else np.asarray(ensemble.site_index).copy()
        ),
        frame=frame,
    )


def l1_transport_end_to_handover(
    ensemble: ParticleEnsemble, distance_m: float
) -> ParticleEnsemble:
    """Move an L1 lab-frame end ensemble to the handover-local origin."""
    moved = ensemble.translated((0.0, 0.0, -float(distance_m)))
    return ParticleEnsemble(
        positions_m=moved.positions_m,
        velocities_m_s=moved.velocities_m_s,
        weights=moved.weights,
        site_index=moved.site_index,
        frame="handover_l1_local",
    )


def handover_to_l2_local(
    ensemble: ParticleEnsemble, crossing_angle_deg: float
) -> ParticleEnsemble:
    """Rotate canonical handover coordinates so the L2 axis becomes local +z.

    The handover captured-ensemble output has already removed the L2 beam
    offset, lattice displacement/velocity and per-trajectory final phase.
    This adapter therefore performs only the orthogonal basis rotation and
    preserves kinetic temperature exactly.
    """
    angle = math.radians(float(crossing_angle_deg))
    # Rows are the L2-local basis vectors expressed in the handover frame.
    local_x = np.array((math.cos(angle), 0.0, -math.sin(angle)))
    local_y = np.array((0.0, 1.0, 0.0))
    local_z = np.array((math.sin(angle), 0.0, math.cos(angle)))
    rotation = np.stack((local_x, local_y, local_z), axis=0)
    return ensemble.rotated(rotation, frame="l2_local")

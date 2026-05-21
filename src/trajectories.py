"""Time-parameterised Cartesian end-effector trajectories.

A trajectory is a smooth function of time that returns a full 6-DOF pose
(position + orientation) plus its feed-forward derivatives.  Three families
are provided:

* ``circle``   -- a tilted circle (genuinely 3D, not axis aligned)
* ``figure8``  -- a Gerono lemniscate
* ``random``   -- a smooth sum-of-sinusoids "moving target"

The orientation component slowly oscillates around a downward-pointing pose so
that orientation tracking is non-trivial but always reachable.

Design choice: trajectories are *analytic*.  This gives exact feed-forward
velocity (no numerical-differentiation noise) and lets us sample an arbitrary
look-ahead preview window for free -- the agent literally sees the future.
"""
from __future__ import annotations

import numpy as np

from .math_utils import euler_to_quat, quat_error


def _plane_basis(normal):
    """Return two orthonormal vectors spanning the plane with the given normal."""
    normal = np.asarray(normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    helper = np.array([0.0, 1.0, 0.0]) if abs(normal[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


class Trajectory:
    """Base class -- subclasses implement ``position`` and ``velocity``."""

    def __init__(self, name, center, duration, ori_amp=0.22, yaw_amp=0.40,
                 ori_period=7.0):
        self.name = name
        self.center = np.asarray(center, dtype=np.float64)
        self.duration = duration
        self._ori_amp = ori_amp
        self._yaw_amp = yaw_amp
        self._ori_w = 2.0 * np.pi / ori_period

    # -- position (override) --------------------------------------------------
    def position(self, t):
        raise NotImplementedError

    def velocity(self, t):
        raise NotImplementedError

    # -- orientation (shared) -------------------------------------------------
    def _orientation_rpy(self, t):
        # Nominal pose: hand pointing straight down (roll = pi). A slow
        # oscillation in roll/pitch/yaw makes orientation tracking meaningful.
        roll = np.pi + self._ori_amp * np.sin(self._ori_w * t)
        pitch = self._ori_amp * np.sin(0.7 * self._ori_w * t + 1.0)
        yaw = self._yaw_amp * np.sin(0.5 * self._ori_w * t)
        return np.array([roll, pitch, yaw])

    def orientation(self, t):
        return euler_to_quat(self._orientation_rpy(t))

    def angular_velocity(self, t):
        h = 1e-3
        q0 = self.orientation(max(0.0, t - h))
        q1 = self.orientation(t + h)
        return quat_error(q0, q1) / (2.0 * h)

    # -- convenience ----------------------------------------------------------
    def pose(self, t):
        return self.position(t), self.orientation(t)

    def sample(self, ts):
        """Vectorised position sampling for plotting."""
        return np.array([self.position(t) for t in np.atleast_1d(ts)])


class CircleTrajectory(Trajectory):
    def __init__(self, center, radius=0.16, period=8.0, normal=(0.25, 0.0, 1.0),
                 **kw):
        super().__init__("circle", center, **kw)
        self.radius = radius
        self.w = 2.0 * np.pi / period
        self.u, self.v = _plane_basis(normal)

    def position(self, t):
        return (self.center
                + self.radius * np.cos(self.w * t) * self.u
                + self.radius * np.sin(self.w * t) * self.v)

    def velocity(self, t):
        return self.radius * self.w * (
            -np.sin(self.w * t) * self.u + np.cos(self.w * t) * self.v)


class Figure8Trajectory(Trajectory):
    """Gerono lemniscate: x = A sin(t), y = B sin(t) cos(t)."""

    def __init__(self, center, a=0.20, b=0.15, period=10.0,
                 normal=(0.25, 0.0, 1.0), **kw):
        super().__init__("figure8", center, **kw)
        self.a, self.b = a, b
        self.w = 2.0 * np.pi / period
        self.u, self.v = _plane_basis(normal)

    def position(self, t):
        th = self.w * t
        return (self.center
                + self.a * np.sin(th) * self.u
                + self.b * np.sin(th) * np.cos(th) * self.v)

    def velocity(self, t):
        th = self.w * t
        return self.w * (self.a * np.cos(th) * self.u
                         + self.b * np.cos(2.0 * th) * self.v)


class RandomTrajectory(Trajectory):
    """Smooth "moving target": a low-frequency sum of sinusoids per axis."""

    def __init__(self, center, rng, extent=0.17, n_modes=3, speed=1.0, **kw):
        super().__init__("random", center, **kw)
        self.freq = rng.uniform(0.34, 1.45, size=(3, n_modes)) * speed
        self.phase = rng.uniform(0.0, 2.0 * np.pi, size=(3, n_modes))
        amp = rng.uniform(0.4, 1.0, size=(3, n_modes))
        # normalise so the worst-case excursion equals `extent`
        self.amp = amp / amp.sum(axis=1, keepdims=True) * extent

    def position(self, t):
        s = np.sin(self.freq * t + self.phase)
        return self.center + (self.amp * s).sum(axis=1)

    def velocity(self, t):
        c = np.cos(self.freq * t + self.phase)
        return (self.amp * self.freq * c).sum(axis=1)


_NAMES = ("circle", "figure8", "random")


def make_trajectory(name, rng, cfg, center=None, scale=1.0, speed=1.0):
    """Factory.  ``center`` is the workspace anchor (defaults to the config
    value), ``scale`` enlarges the path (used to push it out of reach) and
    ``speed`` multiplies how fast it is traversed."""
    if name == "any":
        name = _NAMES[rng.integers(len(_NAMES))]
    if center is None:
        center = cfg.workspace_center
    center = np.asarray(center, dtype=np.float64)
    dur = cfg.episode_seconds
    if name == "circle":
        return CircleTrajectory(center, radius=0.16 * scale,
                                period=4.5 / speed, duration=dur)
    if name == "figure8":
        return Figure8Trajectory(center, a=0.20 * scale, b=0.15 * scale,
                                 period=5.6 / speed, duration=dur)
    if name == "random":
        return RandomTrajectory(center, rng, extent=0.17 * scale,
                                speed=speed, duration=dur)
    raise ValueError(f"unknown trajectory '{name}'")

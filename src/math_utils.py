"""Minimal, dependency-free quaternion / rotation helpers.

All quaternions use the ``[x, y, z, w]`` convention (same as PyBullet) so the
values produced here can be fed straight into the simulator and compared with
``getLinkState`` outputs without any conversion.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-9


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < EPS:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def quat_conj(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([-x, -y, -z, w])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a * b`` (apply rotation b, then a)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """Return the rotation vector (axis * angle) of a quaternion.

    The shortest equivalent rotation is always chosen (angle in [0, pi]).
    """
    q = quat_normalize(q)
    if q[3] < 0.0:  # pick the shorter arc
        q = -q
    w = np.clip(q[3], -1.0, 1.0)
    sin_half = np.sqrt(max(0.0, 1.0 - w * w))
    angle = 2.0 * np.arccos(w)
    if sin_half < 1e-6:
        return np.zeros(3)
    axis = q[:3] / sin_half
    return axis * angle


def quat_error(q_cur: np.ndarray, q_des: np.ndarray) -> np.ndarray:
    """Rotation vector (world frame) that rotates ``q_cur`` onto ``q_des``."""
    return quat_to_axis_angle(quat_mul(q_des, quat_conj(q_cur)))


def quat_angle(q_cur: np.ndarray, q_des: np.ndarray) -> float:
    """Geodesic angle between two orientations, in radians."""
    return float(np.linalg.norm(quat_error(q_cur, q_des)))


def axis_angle_to_quat(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = np.linalg.norm(rotvec)
    if angle < EPS:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = rotvec / angle
    s = np.sin(angle / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle / 2.0)])


def euler_to_quat(rpy: np.ndarray) -> np.ndarray:
    """Roll-pitch-yaw (fixed-axis X, Y, Z) to quaternion ``[x, y, z, w]``."""
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """Quaternion to roll-pitch-yaw (radians)."""
    x, y, z, w = quat_normalize(q)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])

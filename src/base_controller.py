"""Model-based base controller: resolved-rate motion control.

This is the *analytic* half of the hybrid system.  Given the desired
end-effector twist (feed-forward velocity + proportional pose-error feedback)
it solves for joint velocities through a damped-least-squares (DLS) inverse of
the geometric Jacobian:

        q_dot = J^T (J J^T + lambda^2 I)^-1  x_dot

The damping term keeps the solution finite near kinematic singularities,
which is exactly where a naive Jacobian-pseudoinverse controller blows up.

On its own this controller already *roughly* tracks a trajectory.  What it
cannot handle well is control delay, sensor noise and the lag introduced by
real second-order arm dynamics -- that is the job left to the RL residual.
"""
from __future__ import annotations

import numpy as np

from .math_utils import quat_error


class ResolvedRateController:
    def __init__(self, cfg):
        self.kp_pos = cfg.kp_pos
        self.kp_ori = cfg.kp_ori
        self.damping = cfg.dls_damping
        self.max_joint_vel = cfg.max_joint_vel
        self.track_orientation = cfg.track_orientation

    def compute(self, ee_pos, ee_quat, jac_lin, jac_ang,
                p_des, q_des, v_des, w_des):
        """Return a 7-vector of joint velocities (rad/s).

        Parameters are all in the world frame.  ``jac_lin`` / ``jac_ang`` are
        the 3x7 translational / rotational Jacobian blocks of the arm.
        """
        e_pos = p_des - ee_pos
        v_cmd = v_des + self.kp_pos * e_pos

        if self.track_orientation:
            e_ori = quat_error(ee_quat, q_des)
            w_cmd = w_des + self.kp_ori * e_ori
            twist = np.concatenate([v_cmd, w_cmd])
            J = np.vstack([jac_lin, jac_ang])          # 6 x 7
        else:
            twist = v_cmd
            J = jac_lin                                 # 3 x 7

        # Damped least squares: stable through singularities.
        m = J.shape[0]
        qdot = J.T @ np.linalg.solve(J @ J.T + (self.damping ** 2) * np.eye(m),
                                     twist)
        return np.clip(qdot, -self.max_joint_vel, self.max_joint_vel)

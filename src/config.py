"""Central configuration for the tracking environment.

Everything that shapes the task (timing, trajectory family, uncertainty,
reward weights) lives here so a reviewer can see the whole problem definition
on one screen.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Tuple


@dataclass
class EnvConfig:
    # ---- simulation timing -------------------------------------------------
    # The physics rate is taken from the MJCF (500 Hz). The agent issues
    # commands at `control_hz`; the env runs the matching number of substeps.
    control_hz: float = 20.0       # rate at which the agent issues commands
    episode_seconds: float = 12.0  # length of one tracking episode

    # ---- task --------------------------------------------------------------
    track_orientation: bool = True
    # Fallback trajectory anchor; the env overrides this with the Sawyer's
    # actual home end-effector position at start-up.
    workspace_center: Tuple[float, float, float] = (0.62, 0.0, 0.5)

    # ---- residual action ---------------------------------------------------
    # The policy outputs a residual joint-velocity correction in [-1, 1]^7,
    # scaled by this value (rad/s). Keeping it small is what guarantees that
    # the policy can never produce violent, jittery motion.
    residual_vel_scale: float = 0.6

    # First-order low-pass filter on the final joint-velocity command. This is
    # the hard guarantee of smooth motion: sensor noise / action noise / any
    # high-frequency residual is attenuated before it reaches the actuators.
    # The small lag it introduces is exactly what the RL preview compensates.
    cmd_filter_beta: float = 0.65

    # ---- trajectory preview ------------------------------------------------
    # The policy sees a short window of *upcoming* targets so it can
    # anticipate instead of chasing -- this is key for low-lag tracking.
    preview_count: int = 5
    preview_dt: float = 0.15       # seconds between preview samples

    # ---- uncertainty/ reboustness ------------------------
    obs_noise_std: float = 0.005   # joint encoder noise (rad)
    ee_obs_noise_std: float = 0.0018  # EE position noise (m)
    act_noise_std: float = 0.025   # applied command noise (rad/s)
    delay_steps: int = 1           # nominal control buffer delay (steps)
    domain_rand: bool = True       # randomise the above per-episode during training
    delay_range: Tuple[int, int] = (0, 4)   # up to 200 ms of control delay
    noise_scale_range: Tuple[float, float] = (0.5, 1.6)

    # ---- reward weights ----------------------------------------------------
    w_pos: float = 1.0             # position tracking term
    w_ori: float = 0.35            # orientation tracking term
    w_action: float = 0.055        # penalise large residuals (-> defer to base
                                   #   controller wherever it already tracks well)
    w_action_rate: float = 0.200   # penalise *changes* in residual -> smoothness
    w_jvel: float = 0.011          # penalise raw joint velocity -> smoothness
    pos_sigma: float = 0.05        # m, width of the position reward kernel
    ori_sigma: float = 0.35        # rad, width of the orientation reward kernel
    precision_bonus: float = 0.5   # extra reward for sub-centimetre tracking
    precision_thresh: float = 0.01  # m

    # ---- base controller gains --------------------------------------------
    kp_pos: float = 5.0
    kp_ori: float = 4.0
    dls_damping: float = 0.06      # DLS; regularies near-singular jacobians
    max_joint_vel: float = 2.0     # total command clip (rad/s)

    seed: int = 0

    # ---- derived -----------------------------------------------------------
    @property
    def dt(self) -> float:
        return 1.0 / self.control_hz

    @property
    def max_steps(self) -> int:
        return round(self.episode_seconds * self.control_hz)

    def to_dict(self) -> dict:
        return asdict(self)

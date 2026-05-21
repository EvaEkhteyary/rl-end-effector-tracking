"""Build `assets/sawyer.xml` -- a self-contained Sawyer + Robotiq-85 model.

`robosuite` is used **only here**, as a build-time source of validated robot /
gripper kinematics and meshes.  The generated `assets/sawyer.xml` plus
`assets/sawyer_meshes/` have no dependency on robosuite -- raw `mujoco` loads
them directly, so the shipped project stays self-contained.

What this script does to the robosuite-composed model:
  * vendors every mesh into `assets/sawyer_meshes/` with relative paths
  * drops collision-only geoms (the task is free-space tracking)
  * removes the gripper's finger joints/tendons -> a rigid Robotiq-85
  * replaces the arm's torque motors with position-servo actuators
  * adds a ground plane, lighting, scene textures and a `home` keyframe
  * disables contacts and sets a fixed 500 Hz timestep

Prerequisite:  pip install robosuite      (needed only to rebuild the model)
Run once:      .venv/bin/python tools/build_sawyer_model.py
"""
from __future__ import annotations

import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from robosuite.models.grippers import gripper_factory
from robosuite.models.robots import Sawyer

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
MESH_DIR = ASSETS / "sawyer_meshes"

ARM_JOINTS = [f"robot0_right_j{i}" for i in range(7)]
# Sawyer joint torque limits (N*m) -> used as PD force limits.
ARM_FORCE = [80, 80, 40, 40, 9, 9, 9]
ARM_KP = [4000, 4000, 2000, 2000, 600, 600, 400]
# Joint damping + rotor armature -> well-damped, stable position control.
ARM_DAMPING = [60, 60, 40, 40, 12, 12, 8]
ARM_ARMATURE = [0.30, 0.30, 0.20, 0.20, 0.10, 0.10, 0.06]
# A roomy, non-singular "ready" pose: gripper at (0.62, 0, 0.5) pointing down
# (found by inverse kinematics -- see tools/ for the probe).
HOME_QPOS = [0.0296, -0.9974, -0.6683, 0.9166, 0.3484, 1.745, -0.3187]


def main():
    robot = Sawyer()
    robot.add_gripper(gripper_factory("Robotiq85Gripper"))
    root = robot.root

    # --- 1. vendor meshes with relative paths -----------------------------
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    n_mesh = 0
    for mesh in root.iter("mesh"):
        f = mesh.get("file")
        if f and os.path.exists(f):
            shutil.copy(f, MESH_DIR / os.path.basename(f))
            mesh.set("file", f"sawyer_meshes/{os.path.basename(f)}")
            n_mesh += 1

    # --- 2. drop collision-only geoms (contacts are disabled anyway) and
    #        robosuite's visualization sites (keep only the EE site) --------
    for body in root.iter("body"):
        for geom in list(body.findall("geom")):
            if geom.get("group") != "1":
                body.remove(geom)
        for site in list(body.findall("site")):
            if site.get("name") != "gripper0_grip_site":
                body.remove(site)

    # --- 3. make the gripper rigid: remove finger joints + tendons --------
    for tendon in list(root.findall("tendon")):
        root.remove(tendon)
    for sensor in list(root.findall("sensor")):
        root.remove(sensor)
    for body in root.iter("body"):
        for joint in list(body.findall("joint")):
            name = joint.get("name")
            if name not in ARM_JOINTS:
                body.remove(joint)
            else:                                  # tune the 7 arm joints
                i = ARM_JOINTS.index(name)
                joint.set("damping", str(ARM_DAMPING[i]))
                joint.set("armature", str(ARM_ARMATURE[i]))

    # --- 4. replace actuators: 7 position servos on the arm --------------
    for act in list(root.findall("actuator")):
        root.remove(act)
    actuator = ET.SubElement(root, "actuator")
    for jn, kp, fr in zip(ARM_JOINTS, ARM_KP, ARM_FORCE):
        ET.SubElement(actuator, "position",
                      {"name": f"act_{jn[7:]}", "joint": jn,
                       "kp": str(kp), "forcerange": f"-{fr} {fr}"})

    # --- 5. compiler / option / visual -----------------------------------
    for tag in ("compiler", "option", "visual", "size", "keyframe"):
        for el in list(root.findall(tag)):
            root.remove(el)
    ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true"})
    opt = ET.SubElement(root, "option",
                        {"timestep": "0.002", "integrator": "implicitfast"})
    ET.SubElement(opt, "flag", {"contact": "disable"})
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "headlight", {"diffuse": "0.6 0.6 0.6",
                  "ambient": "0.35 0.35 0.35", "specular": "0 0 0"})
    ET.SubElement(vis, "global", {"offwidth": "1280", "offheight": "960"})

    # --- 6. scene: textures + ground plane + light -----------------------
    asset = root.find("asset")
    ET.SubElement(asset, "texture", {"name": "skybox", "type": "skybox",
                  "builtin": "gradient", "rgb1": "0.45 0.55 0.7",
                  "rgb2": "0.08 0.1 0.14", "width": "256", "height": "256"})
    ET.SubElement(asset, "texture", {"name": "grid", "type": "2d",
                  "builtin": "checker", "rgb1": "0.22 0.25 0.30",
                  "rgb2": "0.27 0.30 0.36", "width": "300", "height": "300"})
    ET.SubElement(asset, "material", {"name": "grid", "texture": "grid",
                  "texrepeat": "8 8", "reflectance": "0.08"})
    world = root.find("worldbody")
    world.insert(0, ET.Element("geom", {"name": "floor", "type": "plane",
                 "size": "3 3 0.1", "material": "grid", "group": "1"}))
    world.insert(1, ET.Element("light", {"name": "top", "pos": "0.6 0 2.6",
                 "dir": "0 0 -1", "diffuse": "0.55 0.55 0.55"}))

    # --- 7. home keyframe (EE site is gripper0_grip_site) ----------------
    kf = ET.SubElement(root, "keyframe")
    ET.SubElement(kf, "key", {"name": "home",
                  "qpos": " ".join(map(str, HOME_QPOS)),
                  "ctrl": " ".join(map(str, HOME_QPOS))})

    # --- write ------------------------------------------------------------
    ET.indent(root)
    out = ASSETS / "sawyer.xml"
    ET.ElementTree(root).write(out, encoding="unicode")
    print(f"[build] wrote {out}")
    print(f"[build] vendored {n_mesh} meshes -> {MESH_DIR}")


if __name__ == "__main__":
    main()

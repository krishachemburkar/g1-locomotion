"""Build the locomotion version of the G1 model.

Reads assets/unitree_g1/model.xml (29 DoF + Dex3 hands, the same model the
harvest scene uses) and writes assets/unitree_g1/g1_loco.xml where:

- only the 12 leg joints move; waist, arms and hands are rigid (their mass
  and inertia stay, so the robot weighs the same),
- leg joints get rotor armature so explicit PD control at 200 Hz is stable,
- only the foot contact spheres collide (with the floor and each other),
  which keeps the sim fast,
- each foot gets a named site under the sole, used for height and contact.

Run:  .venv/bin/python build_model.py
"""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "assets" / "unitree_g1" / "model.xml"
DST = ROOT / "assets" / "unitree_g1" / "g1_loco.xml"

LEG_JOINTS = [
    f"{side}_{name}_joint"
    for side in ("left", "right")
    for name in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
]

# Rotor inertia reflected through the gearbox (similar to Unitree's values).
ARMATURE = {"hip": 0.01, "knee": 0.025, "ankle": 0.008}


def build() -> None:
    tree = ET.parse(SRC)
    root = tree.getroot()

    ET.SubElement(root, "option", timestep="0.005", integrator="implicitfast")

    for body in root.iter("body"):
        for joint in list(body.findall("joint")):
            name = joint.get("name")
            if joint.get("type") == "free":
                continue
            if name in LEG_JOINTS:
                key = next(k for k in ARMATURE if k in name)
                joint.set("armature", str(ARMATURE[key]))
                joint.set("damping", "0.1")
                joint.set("frictionloss", "0.1")
            else:
                body.remove(joint)  # body becomes rigidly attached to its parent

        for geom in body.findall("geom"):
            is_foot_sphere = geom.get("type") is None and geom.get("size") == "0.005"
            if is_foot_sphere:
                geom.set("contype", "1")
                geom.set("conaffinity", "1")
                geom.set("friction", "1.0 0.02 0.01")
                geom.set("condim", "3")
            else:
                geom.set("contype", "0")
                geom.set("conaffinity", "0")

        name = body.get("name", "")
        if name.endswith("ankle_roll_link"):
            side = name.split("_")[0]
            ET.SubElement(body, "site", name=f"{side}_foot", pos="0.035 0 -0.035", size="0.01")

    # Keep only the leg actuators. Torques are computed by our PD loop in Python.
    actuator = root.find("actuator")
    for motor in list(actuator):
        if motor.get("joint") not in LEG_JOINTS:
            actuator.remove(motor)

    # Floor collides with the feet; add a tracking camera for videos.
    for geom in root.iter("geom"):
        if geom.get("name") == "floor":
            geom.set("contype", "1")
            geom.set("conaffinity", "1")
    pelvis = next(b for b in root.iter("body") if b.get("name") == "pelvis")
    ET.SubElement(pelvis, "camera", name="track", mode="trackcom", pos="0 -3.0 0.6", xyaxes="1 0 0 0 0.2 1")

    ET.indent(tree)
    tree.write(DST)
    print(f"wrote {DST.relative_to(ROOT)}")


if __name__ == "__main__":
    build()

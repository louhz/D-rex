
from __future__ import annotations

import argparse
import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml


ARM_NAME_MAP = {
        "panda_link0": "link0",
        "panda_link1": "link1",
        "panda_link2": "link2",
        "panda_link3": "link3",
        "panda_link4": "link4",
        "panda_link5": "link5",
        "panda_link6": "link6",
        "panda_link7": "link7",
    }

LEAP_LINKS = [
    "palm",
    "if_bs",
    "if_px",
    "if_md",
    "if_ds",
    "mf_bs",
    "mf_px",
    "mf_md",
    "mf_ds",
    "rf_bs",
    "rf_px",
    "rf_md",
    "rf_ds",
    "th_mp",
    "th_bs",
    "th_px",
    "th_ds"
]


def parse_vec(text: str | None, default: list[float] | None = None) -> np.ndarray:
        if text is None:
            return np.array(default if default is not None else [0.0, 0.0, 0.0], dtype=float)
        return np.array([float(x) for x in text.strip().split()], dtype=float)


def parse_quat(text: str | None) -> np.ndarray:
        if text is None:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        return np.array([float(x) for x in text.strip().split()], dtype=float)


def quat_normalize(q: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(q)
        if n == 0:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        return q / n


def quat_to_rot(q: np.ndarray) -> np.ndarray:
        w, x, y, z = quat_normalize(q)
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=float,
        )


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        return quat_to_rot(q) @ v


def box_to_spheres(size: np.ndarray, pos: np.ndarray, quat: np.ndarray, max_spheres_per_box: int = 12):
        s = np.asarray(size, dtype=float)
        major = int(np.argmax(s))
        a = float(s[major])
        b, c = [float(s[i]) for i in range(3) if i != major]
        lateral = max(b, c)
        if lateral < 1e-8:
            n = 1
        else:
            n = max(1, int(math.ceil(a / lateral)))
        n = min(n, max_spheres_per_box)
        seg_half = a / n
        radius = math.sqrt(b * b + c * c + seg_half * seg_half)

        centers_local = [np.zeros(3, dtype=float)]
        if n > 1:
            centers_local = []
            for coord in np.linspace(-a + seg_half, a - seg_half, n):
                p = np.zeros(3, dtype=float)
                p[major] = coord
                centers_local.append(p)

        spheres = []
        for center_local in centers_local:
            center_body = rotate(quat, center_local) + pos
            spheres.append(
                {
                    "center": [round(float(x), 6) for x in center_body.tolist()],
                    "radius": round(float(radius), 6),
                }
            )
        return spheres


def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--mjcf", required=True, type=Path)
        parser.add_argument("--franka-spheres", required=True, type=Path)
        parser.add_argument("--out", required=True, type=Path)
        args = parser.parse_args()

        franka_spheres = yaml.safe_load(args.franka_spheres.read_text())["collision_spheres"]
        merged = {}
        for src, dst in ARM_NAME_MAP.items():
            merged[dst] = copy.deepcopy(franka_spheres[src])

        root = ET.parse(args.mjcf).getroot()
        world = root.find("worldbody")
        if world is None:
            raise ValueError("MJCF is missing <worldbody>.")

        body_lookup = {}

        def recurse_bodies(body):
            body_lookup[body.get("name")] = body
            for child in body.findall("body"):
                recurse_bodies(child)

        for body in world.findall("body"):
            recurse_bodies(body)

        for link_name in LEAP_LINKS:
            body = body_lookup[link_name]
            link_spheres = []
            for geom in body.findall("geom"):
                if geom.get("type", "mesh") != "box":
                    continue
                size = parse_vec(geom.get("size"))
                pos = parse_vec(geom.get("pos"), default=[0.0, 0.0, 0.0])
                quat = parse_quat(geom.get("quat"))
                link_spheres.extend(box_to_spheres(size, pos, quat))
            merged[link_name] = link_spheres

        args.out.write_text(yaml.safe_dump({"collision_spheres": merged}, sort_keys=False))


if __name__ == "__main__":
        main()

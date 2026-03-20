# Franka + LEAP CuRobo bundle

Files in this folder:

- `franka_leap_spheres.yml`
  Merged collision spheres:
  - Franka arm spheres are based on NVIDIA's official `franka_mesh.yml`.
  - LEAP hand spheres are generated from the collision boxes in `panda_leap.xml`.

- `franka_leap_full.yml`
  Full 23-DOF CuRobo robot config:
  7 Franka joints + 16 LEAP joints.

- `franka_leap_arm_only.yml`
  Arm-only CuRobo robot config:
  - CuRobo plans only the 7 Franka joints.
  - The LEAP joints are fixed through `lock_joints`.
  - Edit the lock values if you want a specific grasp posture.

- `franka_leap_world_example.yml`
  Minimal cuboid world for testing collision checking and motion generation.

- `franka_leap_curobo_demo.py`
  Example script that:
  - patches the YAML with your URDF path,
  - checks start-state collision,
  - plans to a target end-effector pose,
  - checks the final trajectory for world/self collision.

- `build_franka_leap_collision_spheres.py`
  Regenerates the hand sphere model from the MJCF if you update the LEAP collision geometry.

## Important note

CuRobo's robot configuration pipeline is URDF/USD-centric, not MJCF-centric.
So these YAMLs are ready for CuRobo, but they still need a URDF with the same link/joint names as the MJCF:
- links: `link0` ... `link7`, `attachment`, `palm`, `if_bs`, ..., `th_ds`
- joints: `joint1` ... `joint7`, `if_mcp`, ..., `th_ipl`

## Recommended starting point

Use `franka_leap_arm_only.yml` first. It is the safest default for reach planning because the hand geometry is present for collision checking, but the fingers do not drift during arm planning.

## Example

```bash
python franka_leap_curobo_demo.py \
  --robot-yaml franka_leap_arm_only.yml \
  --urdf /abs/path/to/franka_leap.urdf \
  --start-joints 0 -1.3 0 -2.5 0 1.0 0 \
  --goal-pose 0.45 0.0 0.25 1 0 0 0
```

For the full 23-DOF config, provide all 23 start joints.

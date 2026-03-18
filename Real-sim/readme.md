# D-Rex  real2sim 


1. take a video,
2. run RoboGS for GS / mesh,
3. segment the object,
4. keep the RoboGS scene and asset setup,
5. drive the object using the D-Rex trajectory,
6. render the result in RoboGS,
7. also export pose files you can reuse in simulation.

---

## Files in this folder

- `real2sim.py`
  - main bridge script
  - parses the D-Rex observation log
  - reproduces the D-Rex trajectory preprocessing used for ketchup
  - applies the resulting relative motion to the object splats inside `final_scene_with_ids.ply`
  - can render selected frames through RoboGS `Runner.render_step`
  - exports trajectory files for MuJoCo / free-joint playback




### On the RoboGS side

You already ran the upstream RoboGS pipeline far enough to have at least:

- a reconstructed scene dataset (`--robogs-data-dir`) that RoboGS can parse,
- a SAM-labeled scene splat file, typically:
  - `final_scene_with_ids.ply`
- the object already assigned a semantic ID in that file
  - default here is `15`, matching the ketchup setup in the current `4drender.py`



### On the D-Rex side

You have the FoundationPose / observation log used by the ketchup system-ID code, e.g.

- `target_ketchup_20250428_1745.txt`

This script reuses the same trajectory preprocessing logic:

- `load_observations(...)`
- `process_and_interpolate(...)`
- sync timestamp filtering
- `offset_vector`
- `sync_with_real_vector`
- the position sign correction used by the D-Rex loss path

---

## Important trajectory note

The D-Rex ketchup script contains **two** possible pose branches:

- the tensors actually used by the trajectory loss
- a separate flip-back debug branch

This wrapper defaults to:

```bash
--trajectory-variant used_gt
```

That matches the trajectory that is actually fed into the D-Rex loss.

If you want the alternate branch, use:

```bash
--trajectory-variant flip_back_debug
```

---

## What the script outputs

Inside `--result-dir`, the script writes:

- `drex_object_trajectory.npz`
  - full absolute and relative pose arrays
- `drex_object_trajectory_rel.csv`
  - relative object motion as `x y z qw qx qy qz` per frame
- `drex_object_trajectory_meta.json`
  - summary metadata
- `frame_manifest.json`
  - mapping from frame ids to render / exported files
- `run_args.json`
  - exact CLI args used

Optional outputs:

- `renders/*.png`
  - if you pass `--render`
- `drex_robogs_render.mp4`
  - if you pass `--render --write-video`
- `scene_ply_sequence/*.ply`
  - only if you set `--export-scene-ply-every N`
- `drex_object_sim_world_qpos.csv`
  - only if you provide `--initial-sim-pose`

---

## Why the script uses relative motion

The script computes:

```text
T_rel(t) = T_abs(t) @ inv(T_abs(0))
```

and applies that relative transform to the object splats already present in the RoboGS scene.

That is the safest way to bridge the two systems, because:

- the RoboGS scene already contains the object in the correct first-frame placement,
- you do not need to hard-code the manual rotation / translation offsets from `4drender.py`,
- constant local-frame differences between the D-Rex object asset and the RoboGS object splats cancel out automatically at frame 0.

---

## Rendering path

When `--render` is enabled, the script:

1. loads the scene splats from `final_scene_with_ids.ply`,
2. rewrites only the object splats for each selected frame,
3. calls RoboGS `Runner.render_step(...)` directly,
4. saves one PNG per selected frame.

So this wrapper does **not** ask RoboGS to recompute the scene deformation logic from `4drender.py`.
It only uses RoboGS as the renderer for the edited scene.

---

## Quick start

### 1. Activate the RoboGS environment

Use the same environment where RoboGS already imports successfully.
That environment should already contain the RoboGS dependencies.

### 2. Run the script

Example:

```bash
python drex_robogs_real2sim.py \
  --robogs-root /path/to/robogs \
  --ply-file /path/to/final_scene_with_ids.ply \
  --gt-path /path/to/target_ketchup_20250428_1745.txt \
  --result-dir /path/to/output/ketchup_bridge \
  --object-semantic-id 15 \
  --sync-str 2025-04-28T17:45:38.996320 \
  --offset-vector 0 0 0.75 \
  --sync-with-real-vector 0.095 -0.08 0.02 \
  --max-frames 16 \
  --n-samples 600 \
  --position-axis-sign -1 -1 1 \
  --render \
  --robogs-data-dir /path/to/robogs/data_dir \
  --camera-index 0 \
  --write-video
```

---

## Ketchup defaults copied from the D-Rex example

These are the current ketchup-oriented defaults used in the script:

- `--sync-str 2025-04-28T17:45:38.996320`
- `--offset-vector 0 0 0.75`
- `--sync-with-real-vector 0.095 -0.08 0.02`
- `--max-frames 16`
- `--n-samples 600`
- `--dt 0.002`
- `--position-axis-sign -1 -1 1`
- `--object-semantic-id 15`
- `--trajectory-variant used_gt`

If your log or scene registration differs, adjust them on the CLI.

---

## Exporting a simulation-ready free-joint trajectory

If your MuJoCo object body already has the correct **initial** world pose at frame 0,
pass that pose as:

```bash
--initial-sim-pose x y z qw qx qy qz
```

Then the script writes:

- `drex_object_sim_world_qpos.csv`

using:

```text
T_world(t) = T_rel(t) @ T_world(0)
```

This is the easiest way to drive the object body in MuJoCo.

### Example

```bash
python drex_robogs_real2sim.py \
  ... \
  --initial-sim-pose 0.12 -0.03 0.18 1 0 0 0
```

---

## Controlling storage

A full-scene PLY sequence can get very large.
So by default the script does **not** export per-frame scene PLYs.

If you want periodic snapshots, enable them explicitly:

```bash
--export-scene-ply-every 10
```

That means: among the frames selected by `--start/--stop/--stride`, save one full scene PLY every 10 selected frames.

---

## Selecting only part of the trajectory

You can render or export only part of the motion:

```bash
--start 120 --stop 240 --stride 4
```

---

## Recommended first checks

After the first run, check these in order:

1. **Frame 0 render**
   - the ketchup should still sit in the correct place in the original RoboGS scene
2. **Trajectory direction**
   - if motion is mirrored, first try changing `--position-axis-sign`
3. **Temporal sync**
   - if the motion starts too early or late, adjust `--sync-str`
4. **Absolute offset**
   - if translation looks shifted, adjust `--offset-vector` or `--sync-with-real-vector`
5. **Semantic ID**
   - if nothing moves, the object semantic ID is probably not 15 in your scene file

---

## How this maps back to the upstream files

### D-Rex side

This script reuses the trajectory logic from:

- `D-rex/system_id/diff_obj/mass_estimator_ketchup.py`

Specifically:

- parsing the FoundationPose log,
- filtering by `sync_str`,
- computing `reference_transform_inv`,
- applying the swap matrix,
- interpolation to `n_samples`,
- position sign correction used by the D-Rex loss path.

### RoboGS side

This script keeps the normal RoboGS outputs and rendering stack:

- `final_scene_with_ids.ply`
- semantic IDs from the segmented scene
- `Runner.render_step(...)` for rasterization

The main difference from upstream `4drender.py` is that the object trajectory is no longer hard-coded.
It is taken from the D-Rex preprocessing path instead.

---

## Minimal integration mental model

If you want to patch your own local `4drender.py` later, the replacement is conceptually:

```python
# old: hand-tuned object pose block
# new:
T_rel = drex_rel_pose[frame_idx]
xyz_obj_new = apply(T_rel, xyz_obj_old)
quat_obj_new = apply_rotation(T_rel[:3, :3], quat_obj_old)
```

The wrapper in this folder already does that without requiring you to rewrite the upstream files first.

---

## Known limitations

- It assumes the object is a **rigid** subset of the scene splats.
- It only replaces the **object** motion, not the arm / hand deformation logic from RoboGS.
- Rendering still depends on the RoboGS Python environment.
- If your FoundationPose-to-scene alignment differs from the ketchup example, you will need to retune the sync / offset flags.


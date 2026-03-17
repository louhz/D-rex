# FoundationPose 
============================


How to use it
-------------
1. Clone or open your normal FoundationPose repository.
2. Replace the repo's multi_object_tracker_w_point.py with the cleaned file in this folder.
3. Replace the repo's estimater.py with your custom estimater.py as well.
4. Run the tracker from inside the FoundationPose environment.

Example run command
-------------------
python multi_object_tracker_w_point.py --camera-serial 147122073100 --object ketchup=assets/ketchup.stl --debug 1

Optional: pass segmentation points directly
-------------------------------------------
python multi_object_tracker_w_point.py \
  --camera-serial 147122073100 \
  --object ketchup=assets/ketchup.stl \
  --seg-point ketchup:320,240 \
  --debug 1



# Hamor


- `demo.py`  
  Drop-in replacement for HaMeR's root `demo.py`.
  - keeps the standard HaMeR image inference flow
  - fixes hardcoded behavior and cleanup issues
  - uses the requested `--batch_size` value correctly
  - adds optional MANO parameter export with `--save_mano_params`
  - writes MANO outputs to `OUT_FOLDER/pose/` by default

- `tools/extract_video_frames.py`  
  Extracts frames from one video or a folder of videos.

- `tools/batch_video_inference.py`  
  Extracts frames and runs `demo.py` for every video in a folder.

- `tools/rename_images.py`  
  Safely renames images into a frame sequence such as `frame_000000.jpg`.

## Before inserting this patch

Finish the normal HaMeR installation first.

You should already have:

1. cloned the repo with submodules
2. created the Python environment
3. installed HaMeR and ViTPose dependencies
4. run `bash fetch_demo_data.sh`
5. placed `MANO_RIGHT.pkl` in `_DATA/data/mano`

## Where to copy the files

Assuming your installed repo looks like:

```text
hamer/
├── demo.py
├── vitpose_model.py
├── hamer/
├── third-party/
└── ...
```

After copying, your repo should look like this:

```text
hamer/
├── demo.py                    # replaced by cleaned version
├── vitpose_model.py           # original HaMeR file
├── tools/
│   ├── batch_video_inference.py
│   ├── extract_video_frames.py
│   └── rename_images.py
├── hamer/
├── third-party/
└── ...
```


### Example

```bash
python demo.py \
  --img_folder example_data \
  --out_folder demo_out \
  --batch_size 48 \
  --side_view \
  --full_frame \
  --save_mesh \
  --save_mano_params
```

## MANO export output

When `--save_mano_params` is enabled, each detected hand gets one file in:

```text
demo_out/pose/
```

Default filename pattern:

```text
<image_stem>_<person_id>_mano_output.yaml
```

Each file contains:

- `image_name`
- `person_id`
- `is_right`
- `global_orient`
- `hand_pose`
- `betas`
- `camera_translation_full`

### Save JSON instead of YAML

```bash
python demo.py \
  --img_folder example_data \
  --out_folder demo_out \
  --save_mano_params \
  --mano_format json
```

### Change the MANO output folder name

```bash
python demo.py \
  --img_folder example_data \
  --out_folder demo_out \
  --save_mano_params \
  --mano_dir mano_params
```

## Extract frames only

### One folder of videos

```bash
python tools/extract_video_frames.py \
  --input /path/to/videos \
  --output_dir /path/to/extracted_frames \
  --skip_existing
```

This creates one subfolder per video:

```text
/path/to/extracted_frames/
├── video_a/
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
└── video_b/
    ├── 000000.jpg
    ├── 000001.jpg
    └── ...
```

## Batch video inference

This is the replacement for your old `batchify_inf.py`, without hardcoded local paths.

### Example

```bash
python tools/batch_video_inference.py \
  --videos_dir /path/to/videos \
  --frames_dir /path/to/extracted_frames \
  --results_dir /path/to/hamer_results \
  --batch_size 48 \
  --side_view \
  --full_frame \
  --save_mano_params \
  --skip_existing_frames
```

This produces:

```text
/path/to/hamer_results/
├── video_a/
│   ├── frame renders
│   ├── optional meshes
│   └── pose/
└── video_b/
    ├── frame renders
    ├── optional meshes
    └── pose/
```






# Mcc-ho



These scripts assume:

- `mcc-ho` is already installed.
- You run the commands from the **MCC-HO repository root** so the project imports resolve.
- HaMeR is already wired into your local MCC-HO setup in the same way as your original script.

## Files

- `run_mccho_with_hamer_clean.py` — run the MCC-HO demo from one RGB image, one object mask, and one camera-intrinsics JSON.
- `distill_mccho_pose_clean.py` — align HaMeR/MANO wrist pose into the object-centered frame using MCC-HO point clouds.
- `convert_mask_to_grayscale.py` — convert a mask image to grayscale.

---

## 1) Convert a mask to grayscale

```bash
python /path/to/convert_mask_to_grayscale.py demo/mask.png
```

This writes:

```text
demo/mask_gray.png
```

You can also choose the output path explicitly:

```bash
python /path/to/convert_mask_to_grayscale.py demo/mask.png -o demo/mask_gray.png
```

---

## 2) Run MCC-HO with HaMeR from one image

Basic example:

```bash
python /path/to/run_mccho_with_hamer_clean.py \
  --image demo/drink_v_1F96GArORtg_frame000084.jpg \
  --obj-seg demo/drink_v_1F96GArORtg_frame000084_mask.png \
  --cam demo/camera_intrinsics_iphone.json \
  --hand-side right \
  --checkpoint mccho_best_checkpoint.pth \
  --out-folder out_demo_cookie
```

If the manipulated hand is the left hand:

```bash
python /path/to/run_mccho_with_hamer_clean.py \
  --image demo/example.jpg \
  --obj-seg demo/example_mask_gray.png \
  --cam demo/camera_intrinsics_iphone.json \
  --hand-side left \
  --checkpoint mccho_best_checkpoint.pth \
  --out-folder out_left_hand_demo
```

Useful optional arguments:

```bash
python /path/to/run_mccho_with_hamer_clean.py \
  --image demo/example.jpg \
  --obj-seg demo/example_mask.png \
  --cam demo/camera_intrinsics_iphone.json \
  --checkpoint mccho_best_checkpoint.pth \
  --out-folder out_example \
  --granularity 0.1 \
  --score-thresholds 0.1 0.2 0.3 0.4 0.5 \
  --crop-padding 60
```

Main outputs in `--out-folder`:

- `hamer/*.obj` — HaMeR hand meshes
- `input_hand.obj` — selected hand mesh copied for debugging
- `hand_mask.png` — rendered hand silhouette
- `obj_mask.png` — cleaned binary object mask
- `combined_mask.png` — union of hand and object masks used for cropping
- `output.html` — HTML visualization
- `output_*.ply` / `output_*.obj` — MCC-HO exported geometry

---

## 3 Distill MANO pose into the object frame

Expected sequence layout:

```text
sequence_0001/
  output_0.1_hand.ply
  output_0.1.ply
  pose/
    000123__1mano_output.yaml
```

Process one sequence:

```bash
python /path/to/distill_mccho_pose_clean.py sequence_0001
```

Process multiple sequences:

```bash
python /path/to/distill_mccho_pose_clean.py sequence_0001 sequence_0002 sequence_0003
```

Also save the extracted object-only point cloud for inspection:

```bash
python /path/to/distill_mccho_pose_clean.py sequence_0001 --save-object-cloud
```

If you need to tune hand-point removal:

```bash
python /path/to/distill_mccho_pose_clean.py sequence_0001 --subtract-threshold 0.001
```

By default, distilled YAMLs are written to:

```text
sequence_0001/pose_distilled/
```

Each output file contains:

- `wrist_quaternion_xyzw`
- `hand_pose_15x3x3`
- `object_center_xyz`
- `object_quaternion_xyzw`

---

## Suggested workflow

1. Prepare or convert the object mask.
2. Run `run_mccho_with_hamer_clean.py` to generate MCC-HO geometry.
3. Run `distill_mccho_pose_clean.py` on the produced sequence folder to export object-frame wrist pose.

---

## Notes

- The camera JSON should contain `fx`, `fy`, `px`, and `py`.
- In the cleaned demo script, mask loading now works for grayscale, RGB, and RGBA images.
- In the cleaned distillation script, hand subtraction is done by nearest-neighbor distance on the point clouds instead of selecting indices from the wrong cloud.




# MANO-to-Leap Retargeting

This script reads MANO pose YAML files, converts rotation matrices into MANO axis-angle pose vectors, runs `manotorch`, and saves Leap-style hand joint values to a NumPy `.npy` file.

## What it does

For every YAML file in an input directory, the script:

1. loads `betas`, `global_orient`, and `hand_pose`,
2. converts rotation matrices to axis-angle,
3. builds a 58D MANO vector:
   - 48 pose values
   - 10 shape values
4. runs the MANO forward pass through `manotorch`,
5. extracts the Leap hand joint targets,
6. writes all outputs to disk as a NumPy array.

## Expected YAML format

Each YAML file should contain these keys:

```yaml
betas: [10 floats]
global_orient:
  - [r11, r12, r13]
  - [r21, r22, r23]
  - [r31, r32, r33]
hand_pose:
  - [[...], [...], [...]]
  - [[...], [...], [...]]
  # 15 or 16 rotation matrices total
```

Accepted shapes are:

- `betas`: `(10,)` or `(1, 10)`
- `global_orient`: `(3, 3)` or `(1, 3, 3)`
- `hand_pose`: `(15, 3, 3)`, `(16, 3, 3)`, or the same with a leading singleton dimension

## Requirements

- Python 3.9+
- NumPy
- PyYAML
- SciPy
- PyTorch
- `manotorch`

Example install command:

```bash
pip install numpy pyyaml scipy torch
```

You will also need a working `manotorch` installation and the MANO asset files available at your chosen `--mano-assets-root`.

## Files

- `retargeting_cleaned.py`: cleaned command-line version of the script
- output `.npy`: saved Leap joint targets
- optional extra `.npy`: saved intermediate 58D MANO vectors

## Usage

Basic example:

```bash
python retargeting_cleaned.py \
  --input-dir ./data/humandemonstration/screwdrivermanipulate/seq/1/pose \
  --output ./data/humandemonstration/screwdrivermanipulate/leap_batched_output_1.npy
```

With more options:

```bash
python retargeting_cleaned.py \
  --input-dir ./data/humandemonstration/screwdrivermanipulate/seq/1/pose \
  --output ./data/humandemonstration/screwdrivermanipulate/leap_batched_output_1.npy \
  --filename-contains _0mano_output.yaml \
  --batch-size 16 \
  --device cpu \
  --mano-assets-root ./manotorch/assets/mano \
  --save-mano-vectors ./data/humandemonstration/screwdrivermanipulate/mano_vectors.npy
```

## Command-line arguments

- `--input-dir`: directory containing YAML files
- `--output`: output path for the Leap result `.npy`
- `--glob-pattern`: glob used to find files, default `*.yaml`
- `--filename-contains`: optional filename substring filter, default `_0mano_output.yaml`
- `--batch-size`: batch size for YAML loading, default `16`
- `--num-workers`: DataLoader worker count, default `0`
- `--mano-assets-root`: path to MANO assets for `manotorch`
- `--side`: `left` or `right`, default `left`
- `--device`: torch device such as `cpu` or `cuda`
- `--save-mano-vectors`: optional path for the intermediate 58D MANO vectors
- `--log-level`: `DEBUG`, `INFO`, `WARNING`, or `ERROR`

## Output format

### Leap output

The main output file contains one row per YAML file:

```python
(num_files, num_leap_joints)
```

In this script, `num_leap_joints` is `16`.

### Optional MANO vector output

If `--save-mano-vectors` is set, the script also saves:

```python
(num_files, 58)
```

This is:

- 48 MANO pose values
- 10 MANO shape values

## Cleanup improvements in this version

Compared with the original script, this version:

- removes unused imports and dead code,
- adds clear function boundaries and docstrings,
- supports real batching through a `DataLoader`,
- validates shapes and missing YAML keys,
- normalizes mixed 15-joint / 16-joint `hand_pose` inputs so they can be batched together,
- handles device selection properly,
- avoids hardcoded input/output paths,
- adds logging,
- makes output saving explicit,
- preserves the original behavior of using only the first 15 hand joints to build the 48D MANO pose vector.

## Notes

- The script preserves the original left-hand default.
- If your `hand_pose` contains 16 rotation matrices, only the first 15 are used for the 48D MANO pose vector.
- Mixed datasets containing both 15-joint and 16-joint `hand_pose` files are normalized to 15 joints during loading.
- `SciPy` runs the matrix-to-axis-angle conversion on CPU, so tensors are moved to CPU for that step.
- If you need exactly reproducible file selection, keep your naming convention consistent and rely on `--filename-contains`.




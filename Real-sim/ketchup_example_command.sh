#!/usr/bin/env bash

# Fill these in for your machine.
ROBOGS_ROOT="/path/to/robogs"
PLY_FILE="/path/to/final_scene_with_ids.ply"
GT_PATH="/path/to/target_ketchup_20250428_1745.txt"
ROBOGS_DATA_DIR="/path/to/robogs/data_dir"
OUT_DIR="/path/to/output/ketchup_bridge"

python real2sim.py \
  --robogs-root "$ROBOGS_ROOT" \
  --ply-file "$PLY_FILE" \
  --gt-path "$GT_PATH" \
  --result-dir "$OUT_DIR" \
  --object-semantic-id 15 \
  --trajectory-variant used_gt \
  --sync-str 2025-04-28T17:45:38.996320 \
  --offset-vector 0 0 0.75 \
  --sync-with-real-vector 0.095 -0.08 0.02 \
  --max-frames 16 \
  --n-samples 600 \
  --dt 0.002 \
  --position-axis-sign -1 -1 1 \
  --render \
  --robogs-data-dir "$ROBOGS_DATA_DIR" \
  --camera-index 0 \
  --write-video

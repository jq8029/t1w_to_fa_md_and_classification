#!/usr/bin/env bash
set -euo pipefail

# --- EDIT THIS ---
TARGET_ROOT="data"   # contains subject folders
DWI_NAME="data_b0_b1000.nii.gz"
BVAL_NAME="data_b0_b1000.bval"
BVEC_NAME="data_b0_b1000.bvec"
MASK_NAME="nodif_brain_mask.nii.gz"
OUT_SUBDIR="dtifit"
OUT_PREFIX="dtifit"
# -----------------

# Ensure dtifit is available (FSL sourced)
if ! command -v dtifit >/dev/null 2>&1; then
  echo "ERROR: dtifit not found in PATH. Have you sourced FSL? e.g.:"
  echo "  source \$FSLDIR/etc/fslconf/fsl.sh"
  exit 1
fi

echo "Scanning subjects in: $TARGET_ROOT"
shopt -s nullglob
for subj_dir in "$TARGET_ROOT"/*; do
  [[ -d "$subj_dir" ]] || continue
  subj="$(basename "$subj_dir")"
  echo -e "\n==> Subject: $subj"

  dwi="$subj_dir/$DWI_NAME"
  bval="$subj_dir/$BVAL_NAME"
  bvec="$subj_dir/$BVEC_NAME"
  mask="$subj_dir/$MASK_NAME"

  missing=()
  [[ -f "$dwi"  ]] || missing+=("$DWI_NAME")
  [[ -f "$bval" ]] || missing+=("$BVAL_NAME")
  [[ -f "$bvec" ]] || missing+=("$BVEC_NAME")
  [[ -f "$mask" ]] || missing+=("$MASK_NAME")

  if (( ${#missing[@]} > 0 )); then
    echo "  [SKIP] Missing: ${missing[*]}"
    continue
  fi

  out_dir="$subj_dir/$OUT_SUBDIR"
  mkdir -p "$out_dir"
  out_base="$out_dir/$OUT_PREFIX"

  # Skip if already done (FA exists)
  if [[ -f "${out_base}_FA.nii.gz" ]]; then
    echo "  [SKIP] dtifit outputs already present: ${out_base}_FA.nii.gz"
    continue
  fi

  echo "  Running dtifit..."
  set +e
  dtifit \
    -k "$dwi" \
    -o "$out_base" \
    -m "$mask" \
    -r "$bvec" \
    -b "$bval" \
    --save_tensor
  rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    echo "  [ERR] dtifit failed for $subj (exit $rc)"
  else
    echo "  [OK] dtifit complete -> $out_dir"
  fi
done
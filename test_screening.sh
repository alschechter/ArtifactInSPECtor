#!/bin/bash
#SBATCH --job-name=cutouts_screen_test
#SBATCH --output=logs/cutouts_screen_test_%j.out
#SBATCH --error=logs/cutouts_screen_test_%j.err
#SBATCH --partition=msismall
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --mail-type=BEGIN,END,FAIL

# Test the sam-screening version of the cutouts step on ONE detector, from a clean slate.
# Usage:  cd <folder with the code AND Euclid_Images/>;  sbatch test_screening.sh 2681 23
# Old results for that detector are moved to <det>/old_<timestamp>/, not deleted.

set -euo pipefail
FITS=${1:?give a fits number, e.g. 2681}
DET=${2:?give a detector code, e.g. 23}

module load python/3.10.10-gcc-13.1.0-ucftoxt
source ~/envs/AISpector/bin/activate

# Code and data folder: the directory you ran sbatch from (override with BASE_DIR=... sbatch ...).
# Cutouts_Pipeline.py looks for calib/, Official-Roman-Artifact-Detection/ and writes <fits>/<det>/
# next to itself, so the code must live in the same folder as the data.
BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$BASE_DIR"
echo "BASE_DIR = $BASE_DIR   (git branch: $(git branch --show-current 2>/dev/null || echo 'not a git repo'))"
[ -d "$BASE_DIR/Euclid_Images" ] || { echo "ERROR: $BASE_DIR/Euclid_Images not found - run sbatch from the folder that holds the data"; exit 1; }

# the new code must sit next to Cutouts_Pipeline.py in $BASE_DIR
for f in Cutouts_Pipeline.py euclid_mask.py artifact_screen.py; do
    [ -f "$BASE_DIR/$f" ] || { echo "ERROR: $BASE_DIR/$f not found - check out the sam-screening branch there or copy the file in"; exit 1; }
done
grep -q "artifact_screen" "$BASE_DIR/Cutouts_Pipeline.py" \
  || { echo "ERROR: $BASE_DIR/Cutouts_Pipeline.py is the old version (no screening)"; exit 1; }
python -c "import cv2, skimage" \
  || { echo "ERROR: missing packages - run: pip install opencv-python-headless scikit-image"; exit 1; }

matches=("$BASE_DIR"/Euclid_Images/*_${FITS}_*.fits)
fits_file="${matches[0]}"
[ -e "$fits_file" ] || { echo "No FITS file for $FITS"; exit 1; }
target="$BASE_DIR/$FITS/$DET"
mkdir -p "$target"

# move the previous run's outputs aside so every file afterwards comes from this run
stamp=$(date +%Y%m%d_%H%M%S)
old="$target/old_$stamp"
mkdir -p "$old"
for item in cutouts sam_results sam_results_precontsub euclid_masks \
            detections_*_DET${DET}.csv screened_out_*_DET${DET}.csv; do
    for p in "$target"/$item; do if [ -e "$p" ]; then mv "$p" "$old"/; fi; done
done
mkdir -p "$target/sam_results_precontsub"     # PreContSub writes here and expects it to exist

echo "=== Cutouts + screening: $fits_file det $DET -> $target ==="
(cd "$target" && python "$BASE_DIR/Cutouts_Pipeline.py" "$fits_file" --det-code "$DET" --force)

echo "=== Summary ==="
cd "$target"
python - <<EOF
import glob, pandas as pd
d = pd.read_csv(glob.glob('detections_*_DET${DET}.csv')[0])
s = glob.glob('screened_out_*_DET${DET}.csv')
s = pd.read_csv(s[0]) if s else pd.DataFrame(columns=['screen_reason'])
print(f"to SAM: {len(d)}   screened out: {len(s)}")
print("to SAM by reason:\n", d['screen_reason'].value_counts().to_string())
print("screened out by reason:\n", s['screen_reason'].value_counts().to_string())
EOF
echo "QA image: $(ls "$BASE_DIR/$FITS/$DET"/*_DET${DET}_screened.png 2>/dev/null)"
echo "Masks:    $BASE_DIR/$FITS/$DET/euclid_masks/classified_overlay_legend.png"

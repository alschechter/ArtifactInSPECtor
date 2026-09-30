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
# Usage:  sbatch test_screening.sh 2681 23
# Old results for that detector are moved to <det>/old_<timestamp>/, not deleted.

set -euo pipefail
FITS=${1:?give a fits number, e.g. 2681}
DET=${2:?give a detector code, e.g. 23}

module load python/3.10.10-gcc-13.1.0-ucftoxt
source ~/envs/AISpector/bin/activate

BASE_DIR=/users/7/aimees/AI_Inspector
cd "$BASE_DIR"

python -c "import cv2, skimage, euclid_mask, artifact_screen" \
  || { echo "Missing deps: pip install opencv-python-headless scikit-image"; exit 1; }

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

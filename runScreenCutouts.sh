#!/bin/bash
#SBATCH --job-name=screen_cutouts
#SBATCH --output=logs/screen_cutouts_%j.out
#SBATCH --error=logs/screen_cutouts_%j.err
#SBATCH --partition=msismall
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00          # ~1.5-2.5 min per detector -> ~40 min for 16 detectors
#SBATCH --mail-type=BEGIN,END,FAIL

# Run Screen_Cutouts_Pipeline.py on ONE FITS file: all its detectors, or just one.
#
# Usage (from the folder that holds the code and Euclid_Images/):
#   sbatch runScreenCutouts.sh 2681                    -> all detectors of FITS 2681
#   sbatch runScreenCutouts.sh 2681 23                 -> only detector 23
#   sbatch runScreenCutouts.sh 2681 --force            -> all detectors, redo ones already done
#   sbatch runScreenCutouts.sh 2681 23 --aggressive    -> aggressive mode (output in <fits>/<det>/aggressive/)
# Extra flags (--force, --aggressive, --no-rotate, --no-screen, ...) are passed to the pipeline.
# FITS_FILE=/full/path/file.fits sbatch runScreenCutouts.sh 2681   uses that file instead of searching.
#
# Output per detector: <fits>/<det>/ (cutouts/, detections CSV, screened-out CSV, QA images, euclid_masks/).
# Detectors whose detections CSV already exists are skipped unless --force is given, so if a job stops
# part-way, resubmitting the same command continues with the detectors that are left.

set -uo pipefail
FITS=${1:?usage: sbatch runScreenCutouts.sh <fits> [<det>] [pipeline flags]}
shift
DET=""
if [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]{2}$ ]]; then DET=$1; shift; fi
EXTRA=("$@")

module load python/3.10.10-gcc-13.1.0-ucftoxt
source ~/envs/AISpector/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$BASE_DIR"
echo "BASE_DIR = $BASE_DIR   (git branch: $(git branch --show-current 2>/dev/null || echo 'not a git repo'))"

for f in Screen_Cutouts_Pipeline.py euclid_mask.py artifact_screen.py dispersion_frame.py; do
    [ -f "$BASE_DIR/$f" ] || { echo "ERROR: $BASE_DIR/$f not found"; exit 1; }
done

if [ -n "${FITS_FILE:-}" ]; then
    fits_file="$FITS_FILE"
else
    fits_file=$(find -L "$BASE_DIR/Euclid_Images" -type f \( -name "*_${FITS}_*.fits" -o -name "*_${FITS}_*.fits.gz" \
                -o -name "*${FITS}*.fits" -o -name "*${FITS}*.fits.gz" \) 2>/dev/null | sort | head -1)
fi
if [ -z "$fits_file" ] || [ ! -e "$fits_file" ]; then
    echo "ERROR: no FITS file for '$FITS' in $BASE_DIR/Euclid_Images (searched *${FITS}*.fits[.gz], incl. subfolders)."
    echo "       Point to it directly:  FITS_FILE=/full/path/to/file.fits sbatch runScreenCutouts.sh $FITS ${DET}"
    exit 1
fi

if [ -n "$DET" ]; then
    echo "=== $fits_file  detector $DET  ${EXTRA[*]:-} ==="
    python "$BASE_DIR/Screen_Cutouts_Pipeline.py" "$fits_file" --det-code "$DET" ${EXTRA[@]+"${EXTRA[@]}"}
else
    echo "=== $fits_file  ALL detectors  ${EXTRA[*]:-} ==="
    python "$BASE_DIR/Screen_Cutouts_Pipeline.py" "$fits_file" ${EXTRA[@]+"${EXTRA[@]}"}
fi
status=$?

echo "=== Done (exit $status). Per detector: $BASE_DIR/<fits>/<det>/ ==="
ls -d "$BASE_DIR"/"$FITS"/[0-9][0-9] 2>/dev/null | while read dd; do
    n=$(ls "$dd"/cutouts/*.npy 2>/dev/null | grep -Ev '_(bbox|points|precontsub|dark)\.npy$' | wc -l)
    echo "  $(basename "$dd"): $n cutouts"
done
exit $status

#!/bin/bash
#SBATCH --job-name=cutouts
#SBATCH --output=logs/cutouts_%j.out
#SBATCH --error=logs/cutouts_%j.err
#SBATCH --partition=msismall     
#SBATCH --cpus-per-task=2
#SBATCH --mem=3G
#SBATCH --time=00:05:00
#SBATCH --mail-type=BEGIN,END,FAIL


module load python/3.10.10-gcc-13.1.0-ucftoxt

source ~/envs/AISpector/bin/activate

BASE_DIR=/users/7/aimees/AI_Inspector
cd "$BASE_DIR"

# Usage: sbatch runCutouts.sh [fits_number] [det_number] [--force]
#   sbatch runCutouts.sh              -> every FITS in Euclid_Images, all detectors
#                                        (detectors that already have a CSV are skipped,
#                                        so this just picks up newly downloaded files)
#   sbatch runCutouts.sh 2681         -> all detectors for fits 2681
#   sbatch runCutouts.sh 2681 11      -> just fits 2681, detector 11
#   sbatch runCutouts.sh 2681 --force -> all detectors for fits 2681, reprocess existing
# Outputs go to $BASE_DIR/<fits_number>/<det>/ (created if missing).
FORCE_FLAG=""
POSITIONAL=()
for arg in "$@"; do
    if [ "$arg" = "--force" ]; then
        FORCE_FLAG="--force"
    else
        POSITIONAL+=("$arg")
    fi
done
FITS_ARG=${POSITIONAL[0]}
DET_ARG=${POSITIONAL[1]}

DET_OPT=()
[ -n "$DET_ARG" ] && DET_OPT=(--det-code "$DET_ARG")

# EUC_SIR_W-SCIFRM_BKGSUB_<fits_number>_..., so the fits number is field 5
if [ -n "$FITS_ARG" ]; then
    fits_files=("$BASE_DIR"/Euclid_Images/EUC_*_${FITS_ARG}_*.fits)
else
    fits_files=("$BASE_DIR"/Euclid_Images/EUC_*.fits)
fi
if [ ! -e "${fits_files[0]}" ]; then
    echo "WARNING: no FITS files found in $BASE_DIR/Euclid_Images for '${FITS_ARG:-*}'"
    exit 1
fi

# One python call per FITS file, so the file, gelsa frame and ZO catalog are loaded once
for fits_file in "${fits_files[@]}"; do
    echo "=== Processing $fits_file ${DET_ARG:+det $DET_ARG} ==="
    python "$BASE_DIR/ArtifactInSPECtor/Cutouts_Pipeline.py" "$fits_file" "${DET_OPT[@]}" $FORCE_FLAG \
        || echo "WARNING: Cutouts_Pipeline.py failed for $fits_file"
done

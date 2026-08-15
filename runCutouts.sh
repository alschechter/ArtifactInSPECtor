#!/bin/bash
#SBATCH --job-name=cutouts
#SBATCH --output=logs/cutouts_%j.out
#SBATCH --error=logs/cutouts_%j.err
#SBATCH --partition=msismall     
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --mail-type=BEGIN,END,FAIL


module load python/3.10.10-gcc-13.1.0-ucftoxt

source ~/envs/AISpector/bin/activate

BASE_DIR=/users/7/aimees/AI_Inspector
cd "$BASE_DIR"

# Usage: sbatch runCutouts.sh [fits_number] [det_number]
#   sbatch runCutouts.sh              -> all fits, all detectors
#   sbatch runCutouts.sh 2681         -> all detectors for fits 2681
#   sbatch runCutouts.sh 2681 11      -> just fits 2681, detector 11
FITS_ARG=$1
DET_ARG=$2

ALL_FITS=(2681 2682 2683 2684 2685 2686 2687 2688 2689 2690 2691 2692)
ALL_DETS=(11 12 13 14 21 22 23 24 31 32 33 34 41 42 43 44)

fits_list=("${ALL_FITS[@]}")
dets_list=("${ALL_DETS[@]}")
[ -n "$FITS_ARG" ] && fits_list=("$FITS_ARG")
[ -n "$DET_ARG" ] && dets_list=("$DET_ARG")

for f in "${fits_list[@]}"; do
    matches=("$BASE_DIR"/Euclid_Images/*_${f}_*.fits)
    if [ ! -e "${matches[0]}" ]; then
        echo "WARNING: no FITS file found for $f, skipping"
        continue
    fi
    fits_file="${matches[0]}"
    for d in "${dets_list[@]}"; do
        target="$BASE_DIR/$f/$d"
        if [ ! -d "$target" ]; then
            echo "WARNING: $target does not exist, skipping"
            continue
        fi
        echo "=== Processing $fits_file det $d -> $target/cutouts ==="
        (cd "$target" && python "$BASE_DIR/Cutouts_Pipeline.py" "$fits_file" --det-code "$d")
    done
done

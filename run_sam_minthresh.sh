#!/bin/bash
#SBATCH --job-name=sam_pipeline
#SBATCH --output=logs/sam_%j.out
#SBATCH --error=logs/sam_%j.err
#SBATCH --partition=msigpu  #msigpu     #preempt-gpu     # adjust to your cluster's GPU partition name
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --mail-type=BEGIN,END,FAIL
#####SBATCH --constraint=a100|h100

module load python/3.10.10-gcc-13.1.0-ucftoxt

source ~/envs/AISpector-v100/bin/activate

BASE_DIR=/users/7/aimees/AI_Inspector
cd "$BASE_DIR"

# Usage: sbatch run_sam.sh [fits_number] [det_number] [--force]
#   sbatch run_sam.sh                     -> all fits, all detectors
#   sbatch run_sam.sh 2681                -> all detectors for fits 2681
#   sbatch run_sam.sh 2681 11             -> just fits 2681, detector 11
#   sbatch run_sam.sh 2681 --force        -> all detectors for fits 2681, recompute existing masks
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

ALL_FITS=(2681 2682 2683 2684 2685 2686 2687 2688 2689 2690 2691 2692)
ALL_DETS=(11 12 13 14 21 22 23 24 31 32 33 34 41 42 43 44)

fits_list=("${ALL_FITS[@]}")
dets_list=("${ALL_DETS[@]}")
[ -n "$FITS_ARG" ] && fits_list=("$FITS_ARG")
[ -n "$DET_ARG" ] && dets_list=("$DET_ARG")

for f in "${fits_list[@]}"; do
    for d in "${dets_list[@]}"; do
        target="$BASE_DIR/$f/$d"
        if [ ! -d "$target" ]; then
            echo "WARNING: $target does not exist, skipping"
            continue
        fi
        echo "=== Running SAM on $f/$d ==="
        (cd "$target" && python "$BASE_DIR/SAM_GPU_Pipeline_MinMaskThresh.py" $FORCE_FLAG)
    done
done

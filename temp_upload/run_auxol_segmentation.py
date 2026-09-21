"""Self-contained AuxOL segmentation step: run after the cutout+SAM pipeline
(Cutouts_Pipeline.py -> SAM_GPU_Pipeline_MinMaskThresh.py) to produce AuxOL
segmentation masks for every cutout in a pipeline output folder.

Everything this script needs -- the AuxOL wrapper, the vendored UNet
(Nets/Unet), img_utils.py -- lives next to it in this same temp_upload/
folder, so the whole folder can be copied/uploaded and run standalone
without the sibling official_roman_artifact_detection repo. The trained
checkpoint (~370MB) is not stored here; it's downloaded on first use from
https://huggingface.co/BCN001/Artifact-Inspector-AUXOL and cached locally
by huggingface_hub (pass --state to point at a local file instead).

Input directory layout expected (as written by Cutouts_Pipeline.py and
SAM_GPU_Pipeline_MinMaskThresh.py):
    <dir>/cutouts/<stem>.npy            cutout science array
    <dir>/cutouts/<stem>_dark.npy       optional flag: cutout is dark-source (negate)
    <dir>/sam_results/<stem>_mask.npy   SAM binary mask
    <dir>/sam_results/<stem>_logits.npy SAM raw pre-sigmoid logits

For each cutout, the AuxOL auxiliary UNet is run (conditioned on the SAM
binary mask as its 4th input channel, as in AuxOL's design) and its output
is fused with the SAM logits at a *fixed* alpha (default 0.0, i.e. pure
UNet segmentation -- SAM's own logits contribute nothing to the fused
mask). This bypasses AuxOL's online alpha history (which predict_only()
would otherwise use) so the requested alpha is applied exactly, the same
way auxol_alpha_sweep.py does it.

Output, into --out-dir (default: <dir>/auxol_segmentation):
    masks/<stem>_auxol_mask.npy   fused binary mask, uint8
    overlays/<stem>.png           cutout + fused mask overlay (unless --no-overlays)
    results.csv                   per-cutout mask pixel counts + SAM/AuxOL agreement

Usage:
    python run_auxol_segmentation.py --dir /path/to/pipeline_output_dir
    python run_auxol_segmentation.py --dir single_detector_demo_DET11 --alpha 0.0
"""
import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from img_utils import to_rgb  # noqa: E402
from auxol_online import AuxOLOnlineUpdater, calc_dice  # noqa: E402

HF_REPO_ID = "BCN001/Artifact-Inspector-AUXOL"
HF_FILENAME = "auxol_train_test_state.pt"


def resolve_state_path(state_arg):
    """A local path if it exists, else download HF_FILENAME from HF_REPO_ID
    (cached under ~/.cache/huggingface/hub after the first download)."""
    if state_arg is not None:
        if not os.path.exists(state_arg):
            sys.exit(f"{state_arg} not found.")
        return state_arg
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("No --state given and huggingface_hub is not installed "
                  "(pip install huggingface_hub) to fetch it from "
                  f"https://huggingface.co/{HF_REPO_ID}.")
    print(f"Fetching {HF_FILENAME} from https://huggingface.co/{HF_REPO_ID} ...")
    return hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)


def overlay(rgb_u8, mask, color=(30, 144, 255), alpha=0.55):
    out = rgb_u8.astype(np.float32).copy()
    m = mask.astype(bool)
    for c in range(3):
        out[..., c][m] = (1 - alpha) * out[..., c][m] + alpha * color[c]
    return out.astype(np.uint8)


def find_stems(cutout_dir, sam_dir):
    stems = []
    for p in sorted(glob.glob(os.path.join(cutout_dir, "*.npy"))):
        name = os.path.basename(p)[:-4]
        if name.endswith(("_bbox", "_precontsub", "_points", "_dark")):
            continue
        mask_path = os.path.join(sam_dir, f"{name}_mask.npy")
        logits_path = os.path.join(sam_dir, f"{name}_logits.npy")
        if os.path.exists(mask_path) and os.path.exists(logits_path):
            stems.append(name)
    return stems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True, help="Pipeline output directory with cutouts/ and sam_results/ subdirs")
    parser.add_argument("--out-dir", default=None, help="Default: <dir>/auxol_segmentation")
    parser.add_argument("--alpha", type=float, default=0.0,
                         help="Fixed SAM/UNet fusion weight (0=pure AuxOL UNet, 1=pure SAM). Default: 0.0")
    parser.add_argument("--state", default=None,
                         help=f"AuxOL checkpoint path (default: download {HF_FILENAME} from "
                              f"huggingface.co/{HF_REPO_ID})")
    parser.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N cutouts (quick test run)")
    parser.add_argument("--no-overlays", action="store_true", help="Skip saving QC overlay PNGs (faster)")
    args = parser.parse_args()

    cutout_dir = os.path.join(args.dir, "cutouts")
    sam_dir = os.path.join(args.dir, "sam_results")
    if not os.path.isdir(cutout_dir) or not os.path.isdir(sam_dir):
        sys.exit(f"Expected {cutout_dir}/ and {sam_dir}/ -- run Cutouts_Pipeline.py then "
                  f"SAM_GPU_Pipeline_MinMaskThresh.py on {args.dir} first.")
    state_path = resolve_state_path(args.state)

    out_dir = args.out_dir or os.path.join(args.dir, "auxol_segmentation")
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)
    if not args.no_overlays:
        os.makedirs(os.path.join(out_dir, "overlays"), exist_ok=True)

    stems = find_stems(cutout_dir, sam_dir)
    print(f"{len(stems)} cutouts with a matching SAM mask+logits in {cutout_dir}/ + {sam_dir}/")
    if args.limit:
        stems = stems[:args.limit]
        print(f"--limit {args.limit}: processing first {len(stems)}")

    updater = AuxOLOnlineUpdater(device=args.device)
    updater.load_state(state_path)
    print(f"Loaded AuxOL state from {state_path} (alpha fixed at {args.alpha:.2f} for every cutout)")

    import csv
    results_path = os.path.join(out_dir, "results.csv")
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stem", "sam_mask_px", "auxol_mask_px", "alpha", "sam_auxol_agreement_dice"])

        for i, stem in enumerate(stems, 1):
            raw = np.load(os.path.join(cutout_dir, f"{stem}.npy")).astype(np.float32)
            if os.path.exists(os.path.join(cutout_dir, f"{stem}_dark.npy")):
                raw = -raw
            crop_rgb = to_rgb(raw)
            sam_mask = np.load(os.path.join(sam_dir, f"{stem}_mask.npy"))
            sam_logits = np.load(os.path.join(sam_dir, f"{stem}_logits.npy"))

            h, w = sam_logits.shape
            sam_bin = (sam_logits > 0).astype(np.uint8)
            image_t = updater._prep_image(crop_rgb, sam_bin)
            unet_logits = updater._unet_forward_infer(image_t, h, w)
            _, fused_mask = updater._fuse(sam_logits, unet_logits, args.alpha)

            np.save(os.path.join(out_dir, "masks", f"{stem}_auxol_mask.npy"), fused_mask.astype(np.uint8))
            if not args.no_overlays:
                Image.fromarray(overlay(crop_rgb, fused_mask)).save(os.path.join(out_dir, "overlays", f"{stem}.png"))

            writer.writerow([
                stem, int(sam_mask.sum()), int(fused_mask.sum()), args.alpha,
                calc_dice(sam_mask.astype(np.float32), fused_mask.astype(np.float32)),
            ])
            if i % 200 == 0 or i == len(stems):
                print(f"  {i}/{len(stems)}")

    print(f"\nSaved {len(stems)} AuxOL masks to {out_dir}/masks/")
    print(f"Per-cutout results: {results_path}")


if __name__ == "__main__":
    main()

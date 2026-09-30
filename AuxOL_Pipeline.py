import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import glob
import os
import sys
import pandas as pd
import argparse
from concurrent.futures import ThreadPoolExecutor
from astropy.visualization import ZScaleInterval
from MaskEncoder import MaskEncoder

# AuxOL segmentation step: run in a detector folder after SAM_GPU_Pipeline_AuxOL.py.
# Mirrors the SAM script's outputs, with AuxOL's mask in place of SAM's:
#   auxol_results/<stem>_mask.npy     AuxOL binary mask
#   auxol_results/auxol_masks.csv     per-cutout AuxOL mask rows
#   sam_results/<stem>_result.jpg     same two-panel plot as the SAM script, drawn with
#                                     AuxOL's mask; replaces SAM's JPG so the Zooniverse
#                                     upload picks it up unchanged (--image-dir to change)
#   detections_*.csv                  gains auxol_x0/y0/x1/y1, auxol_mask_h/w, auxol_mask_encoded
#                                     (the SAM columns are left untouched)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--no-plots', action='store_true', help='Skip saving result JPGs (faster)')
parser.add_argument('--force', action='store_true', help='Recompute masks even if already present, overwriting existing results')
parser.add_argument('--alpha', type=float, default=0.0, help='Fixed SAM/UNet fusion weight (0=pure AuxOL UNet, 1=pure SAM). Values > 0 need SAM logits')
parser.add_argument('--state', type=str, default=None, help='AuxOL checkpoint path (default: download from huggingface.co/BCN001/Artifact-Inspector-AUXOL)')
parser.add_argument('--device', type=str, default=None, help='cuda | cpu (default: auto)')
parser.add_argument('--sam-dir', type=str, default='sam_results', help='Directory holding SAM <stem>_mask.npy / <stem>_logits.npy')
parser.add_argument('--output-dir', type=str, default='auxol_results', help='Directory to save masks/CSV into')
parser.add_argument('--image-dir', type=str, default='sam_results', help='Directory to save <stem>_result.jpg plots into (default overwrites the SAM JPGs used for upload)')
parser.add_argument('--auxol-code-dir', type=str, default=None,
                    help='Folder containing auxol_online.py (default: ArtifactInSPECtor/temp_upload next to this script, or temp_upload/ if this script is inside the repo)')
args = parser.parse_args()

auxol_code_dir = args.auxol_code_dir
if auxol_code_dir is None:
    for candidate in (os.path.join(SCRIPT_DIR, 'ArtifactInSPECtor', 'temp_upload'),
                      os.path.join(SCRIPT_DIR, 'temp_upload')):
        if os.path.exists(os.path.join(candidate, 'auxol_online.py')):
            auxol_code_dir = candidate
            break
if auxol_code_dir is None:
    sys.exit("Couldn't find auxol_online.py; pass --auxol-code-dir /path/to/ArtifactInSPECtor/temp_upload")
sys.path.insert(0, auxol_code_dir)
from auxol_online import AuxOLOnlineUpdater  # noqa: E402
from run_auxol_segmentation import resolve_state_path  # noqa: E402

updater = AuxOLOnlineUpdater(device=args.device)
state_path = resolve_state_path(args.state)
updater.load_state(state_path)
print(f"Using device: {updater.device}")
print(f"Loaded AuxOL state from {state_path} (alpha fixed at {args.alpha:.2f})")

plot_executor = ThreadPoolExecutor(max_workers=2)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


zscale = ZScaleInterval()

os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.image_dir, exist_ok=True)

imageslist = sorted(
    (f for f in glob.glob('cutouts/*.npy') if '_bbox' not in f \
     and '_precontsub' not in f and '_dark' not in f and '_points' not in f),
    key=os.path.getsize,
    reverse=True,
)
print(f"Running AuxOL on {len(imageslist)} cutouts → saving to {args.output_dir}/")

rows = []
n_no_sam = 0
n_mask_fallback = 0

# Pre-load rows for any masks that were already computed in a prior run
for fname in imageslist:
    stem = os.path.splitext(os.path.basename(fname))[0]
    mask_path = os.path.join(args.output_dir, f'{stem}_mask.npy')
    if os.path.exists(mask_path) and not args.force:
        mask = np.load(mask_path)
        ys, xs = np.where(mask)
        h = np.shape(mask)[0]
        w = np.shape(mask)[1]
        pad = 10
        if len(xs):
            rows.append({
                'stem'              : stem,
                'auxol_x0'          : max(0, int(xs.min()) - pad),
                'auxol_y0'          : max(0, int(ys.min()) - pad),
                'auxol_x1'          : min(w, int(xs.max()) + pad),
                'auxol_y1'          : min(h, int(ys.max()) + pad),
                'auxol_mask_h'      : h,
                'auxol_mask_w'      : w,
                'auxol_mask_encoded': MaskEncoder(w,h).encode(mask),
            })

for fname in imageslist:
    stem = os.path.splitext(os.path.basename(fname))[0]
    if os.path.exists(os.path.join(args.output_dir, f'{stem}_mask.npy')) and not args.force:
        print(f"Skipping {stem} (already done)")
        continue

    sam_mask_path = os.path.join(args.sam_dir, f'{stem}_mask.npy')
    sam_logits_path = os.path.join(args.sam_dir, f'{stem}_logits.npy')
    if not os.path.exists(sam_mask_path):
        print(f"Skipping {stem} (no SAM mask in {args.sam_dir}/)")
        n_no_sam += 1
        continue
    sam_mask = np.load(sam_mask_path).astype(np.uint8)
    if os.path.exists(sam_logits_path):
        sam_logits = np.load(sam_logits_path).astype(np.float32)
        sam_bin = (sam_logits > 0).astype(np.uint8)
    elif args.alpha == 0:
        # at alpha=0 SAM's logits never enter the fusion; the UNet only needs
        # SAM's binary mask as its 4th input channel
        sam_logits = np.zeros(sam_mask.shape, dtype=np.float32)
        sam_bin = sam_mask
        n_mask_fallback += 1
    else:
        print(f"Skipping {stem} (no {stem}_logits.npy; needed for --alpha > 0, rerun SAM_GPU_Pipeline_AuxOL.py)")
        n_no_sam += 1
        continue

    raw = np.load(fname).astype(np.float32)
    if raw.ndim != 2:
        print(f"Skipping {stem} (unexpected shape {raw.shape})")
        continue
    dark_flag = fname.replace('.npy', '_dark.npy')
    if os.path.exists(dark_flag):
        raw = -raw

    finite = raw[np.isfinite(raw)]
    try:
        vmin, vmax = zscale.get_limits(finite)
    except Exception:
        vmin, vmax = np.nanpercentile(raw, 1), np.nanpercentile(raw, 99)
    normed = np.clip((raw - vmin) / (vmax - vmin + 1e-12), 0, 1)
    normed = np.nan_to_num(normed, nan=0.0)
    scaled = (normed * 255).astype(np.uint8)
    image = np.stack([scaled, scaled, scaled], axis=-1)
    h, w = image.shape[:2]
    print(f"{os.path.basename(fname)}  {h}×{w}")

    bbox_file = fname.replace('.npy', '_bbox.npy')
    if os.path.exists(bbox_file):
        bbox_local = np.load(bbox_file)
        x0b, y0b, x1b, y1b = bbox_local
        box = np.array([[max(x0b,0), max(y0b,0), min(x1b,w), min(y1b,h)]], dtype=np.float32)
    else:
        box = np.array([[0, 0, w, h]], dtype=np.float32)

    image_t = updater._prep_image(image, sam_bin)
    unet_logits = updater._unet_forward_infer(image_t, h, w)
    _, fused = updater._fuse(sam_logits, unet_logits, args.alpha)
    mask = fused.astype(np.uint8)

    np.save(os.path.join(args.output_dir, f'{stem}_mask.npy'), mask)

    ys, xs = np.where(mask)
    if len(xs):
        auxol_x0, auxol_y0, auxol_x1, auxol_y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    else:
        auxol_x0, auxol_y0, auxol_x1, auxol_y1 = int(box[0,0]), int(box[0,1]), int(box[0,2]), int(box[0,3])

    if not args.no_plots:
        def save_plot(image, mask, box, out_path):
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(image, origin='upper', cmap='gray')
            x0b, y0b, x1b, y1b = box[0]
            rect = plt.matplotlib.patches.Rectangle(
                xy=(x0b, y0b), width=x1b - x0b, height=y1b - y0b,
                edgecolor='limegreen', facecolor='none', linewidth=1.5, linestyle='--'
            )
            rect.set_path_effects([pe.Stroke(linewidth=1.8, foreground='black'), pe.Normal()])
            axes[0].add_patch(rect)
            axes[0].set_title('Data')
            axes[0].axis('off')
            for spine in axes[0].spines.values():
                spine.set_edgecolor('limegreen'); spine.set_linewidth(2); spine.set_visible(True)
            axes[1].imshow(image, origin='upper', cmap='gray')
            show_mask(mask, axes[1])
            axes[1].set_title('Segmentation Map') #Segmentation Map
            axes[1].axis('off')
            for spine in axes[1].spines.values():
                spine.set_edgecolor('limegreen'); spine.set_linewidth(2); spine.set_visible(True)
            plt.tight_layout()
            plt.savefig(out_path, bbox_inches='tight')
            plt.close(fig)

        out_path = os.path.join(args.image_dir, f'{stem}_result.jpg')
        plot_executor.submit(save_plot, image.copy(), mask.copy(), box.copy(), out_path)

    pad = 10
    rows.append({
        'stem'              : stem,
        'auxol_x0'          : max(0, auxol_x0 - pad),
        'auxol_y0'          : max(0, auxol_y0 - pad),
        'auxol_x1'          : min(w, auxol_x1 + pad),
        'auxol_y1'          : min(h, auxol_y1 + pad),
        'auxol_mask_h'      : h,
        'auxol_mask_w'      : w,
        'auxol_mask_encoded': MaskEncoder(w,h).encode(mask),
    })

plot_executor.shutdown(wait=True)

if n_no_sam:
    print(f"WARNING: {n_no_sam} cutouts skipped for missing SAM output in {args.sam_dir}/")
if n_mask_fallback:
    print(f"Note: {n_mask_fallback} cutouts had no SAM logits; used the SAM binary mask as the UNet's 4th channel")

out_csv = os.path.join(args.output_dir, 'auxol_masks.csv')
auxol_df = pd.DataFrame(rows)
auxol_df.to_csv(out_csv, index=False)
print(f"Saved {len(rows)} mask rows → {out_csv}")

if not len(auxol_df):
    print("No AuxOL masks to merge into detections CSVs.")
    sys.exit(0)

group_ids = set()
for stem in auxol_df['stem']:
    after_cutout = stem[len('cutout_'):]
    fits_id, rest = after_cutout.split('_DET', 1)
    det_code = rest.split('_', 1)[0]
    group_ids.add(f"{fits_id}_DET{det_code}")

for group_id in group_ids:
    in_det_csv = f"detections_{group_id}.csv"
    out_det_csv = in_det_csv
    merge_cols = ['auxol_x0', 'auxol_y0', 'auxol_x1', 'auxol_y1', 'auxol_mask_h', 'auxol_mask_w', 'auxol_mask_encoded']
    mask_cols = auxol_df[['stem'] + merge_cols].rename(columns={'stem': 'cutout_stem'})

    if os.path.exists(in_det_csv):
        det_df = pd.read_csv(in_det_csv)
        det_df = det_df.drop(columns=[c for c in merge_cols if c in det_df.columns])
        det_df = det_df.merge(mask_cols, on='cutout_stem', how='left')
    else:
        det_df = mask_cols

    # keep pixel columns as integers (not 7.0) on rows without an AuxOL mask yet
    int_cols = [c for c in merge_cols if c != 'auxol_mask_encoded']
    det_df[int_cols] = det_df[int_cols].astype('Int64')

    det_df.to_csv(out_det_csv, index=False)
    n_merged = det_df['auxol_mask_encoded'].notna().sum()
    print(f"  Wrote {n_merged}/{len(det_df)} AuxOL masks → {out_det_csv}")

print("Done.")

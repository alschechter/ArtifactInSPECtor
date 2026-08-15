import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import glob
import cv2
import os
import pandas as pd
from astropy.visualization import ZScaleInterval
zscale = ZScaleInterval()

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    rect = plt.Rectangle((x0, y0), w, h, edgecolor='limegreen', facecolor='none', linewidth=1.5, linestyle='--')
    rect.set_path_effects([pe.Stroke(linewidth=1.8, foreground='black'), pe.Normal()])
    ax.add_patch(rect)
    
    
def to_display(raw):
    finite = raw[np.isfinite(raw)]
    try:
        vmin, vmax = zscale.get_limits(finite)
    except Exception:
        vmin, vmax = np.nanpercentile(raw, 1), np.nanpercentile(raw, 99)
    normed = np.clip((raw - vmin) / (vmax - vmin + 1e-12), 0, 1)
    return np.nan_to_num(normed, nan=0.0)
    
det_csvs = glob.glob('detections_*.csv')
if len(det_csvs) != 1:
    raise FileNotFoundError(f"Expected exactly one detections_*.csv in {os.getcwd()}, found {det_csvs}")
table = pd.read_csv(det_csvs[0])

for index, row in table.iterrows():
    stem = str(row['cutout_stem'])
    precontsub = np.load('cutouts/' + stem + '_precontsub.npy')
    cutout = np.load('cutouts/' + stem + '.npy')
    mask = np.load('sam_results/' + stem + '_mask.npy')
    bbox = np.load('cutouts/' + stem + '_bbox.npy')
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(to_display(precontsub), cmap='gray', origin='upper')
    axes[0].set_title('Data Before Continuum Subtraction')
    axes[0].axis('off')
    show_box(bbox, axes[0])
    axes[1].imshow(to_display(cutout), cmap='gray', origin='upper')
    show_mask(mask, axes[1])
    show_box(bbox, axes[1])
    axes[1].set_title(f'Segmentation Map')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('sam_results_precontsub/' + stem + 'precont_AND_sam.jpg', bbox_inches='tight')
    plt.close(fig)

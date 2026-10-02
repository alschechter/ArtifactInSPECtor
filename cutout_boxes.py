"""
cutout_boxes.py -- the per-cutout box prompt, centre points and dark flag for a whole detector in
ONE file, cutouts/boxes_<fits_id>_DET<nn>.npz, instead of three tiny files per cutout
(<stem>_bbox.npy, <stem>_points.npy, <stem>_dark.npy). Saves ~3 files per cutout against file-count
quotas and ~2.5 MB of disk blocks per detector. The same numbers are also in the detections CSV
(local_xmin/ymin/xmax/ymax, local_points, source_type).

    from cutout_boxes import load_boxes
    boxes = load_boxes('cutouts')          # {stem: {'bbox': (4,), 'points': (n, 2), 'dark': bool}}
    rec = boxes.get(stem)                  # None if the stem is unknown

load_boxes also reads the old per-cutout files, so folders from earlier runs keep working.
"""
import os, glob
import numpy as np


def boxes_path(cutout_dir, fits_id, det_code):
    return os.path.join(cutout_dir, f"boxes_{fits_id}_DET{det_code}.npz")


def save_boxes(path, records):
    """records: list of (stem, bbox[4], points[n,2], dark: bool)."""
    stems = np.array([r[0] for r in records], dtype=str)
    bbox = np.array([np.asarray(r[1], np.float32).ravel()[:4] for r in records], np.float32).reshape(-1, 4)
    pts = [np.asarray(r[2], np.float32).reshape(-1, 2) for r in records]
    offs = np.cumsum([0] + [len(p) for p in pts]).astype(np.int64)
    allp = np.concatenate(pts, 0) if pts else np.zeros((0, 2), np.float32)
    dark = np.array([bool(r[3]) for r in records], bool)
    tmp = path + '.tmp.npz'
    np.savez_compressed(tmp, stems=stems, bbox=bbox, points=allp, points_offset=offs, dark=dark)
    os.replace(tmp, path)


def load_boxes(cutout_dir='cutouts'):
    out = {}
    for f in sorted(glob.glob(os.path.join(cutout_dir, 'boxes_*.npz'))):
        z = np.load(f)
        offs = z['points_offset']
        for i, s in enumerate(z['stems']):
            out[str(s)] = {'bbox': z['bbox'][i], 'points': z['points'][offs[i]:offs[i + 1]], 'dark': bool(z['dark'][i])}
    # older runs: one file per cutout
    for f in glob.glob(os.path.join(cutout_dir, '*_bbox.npy')):
        s = os.path.basename(f)[:-len('_bbox.npy')]
        if s in out: continue
        p = os.path.join(cutout_dir, s + '_points.npy')
        out[s] = {'bbox': np.load(f).ravel()[:4],
                  'points': np.load(p).reshape(-1, 2) if os.path.exists(p) else np.zeros((0, 2), np.float32),
                  'dark': os.path.exists(os.path.join(cutout_dir, s + '_dark.npy'))}
    return out


def points_to_str(points):
    """CSV form of the centre points: 'x1:y1;x2:y2' (cutout-local pixels)."""
    return ';'.join(f"{x:.1f}:{y:.1f}" for x, y in np.asarray(points).reshape(-1, 2))

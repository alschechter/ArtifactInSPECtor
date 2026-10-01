#!/usr/bin/env python3
"""
compare_aggressive.py -- put ONLY the cutouts that differ between the normal and the aggressive run
into their own folders, with PNG previews for quick browsing (e.g. in VS Code).

Run from a detector folder that holds both runs (or pass it as an argument):
    python compare_aggressive.py [<fits>/<det>]

Creates in that folder:
  dropped_by_aggressive/   cutouts the NORMAL run sends to SAM but the aggressive run does not
                           (what aggressive mode would miss)
  only_in_aggressive/      the few cutouts only the aggressive run makes (from its own trimming)
Each holds, per cutout: the .npy files (cutout, _precontsub, _bbox, _points, _dark if present),
a <stem>.png preview (continuum-subtracted | pre-subtraction, detection box in red) and an
index.csv with the normal run's reason, the aggressive run's reason and the box coordinates.
"""
import os, re, sys, glob, shutil
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

AUX = r'_(bbox|points|precontsub|dark)\.npy$'


def stems(cut_dir):
    return {f[:-4] for f in os.listdir(cut_dir) if f.endswith('.npy') and not re.search(AUX, f)} \
        if os.path.isdir(cut_dir) else set()


def stretch(a):
    a = np.nan_to_num(np.asarray(a, dtype=float))
    lo, hi = np.percentile(a, 1), np.percentile(a, 99.5)
    v = np.arcsinh((np.clip(a, lo, hi) - lo) / max(hi - lo, 1e-9) * 10) / np.arcsinh(10)
    return (v * 255).astype(np.uint8)


def preview(cut_dir, stem, out_png, lines, scale=6, max_side=900):
    cs = np.load(os.path.join(cut_dir, stem + '.npy'))
    pre_p = os.path.join(cut_dir, stem + '_precontsub.npy')
    panels = [('continuum-subtracted', cs)] + ([('before subtraction', np.load(pre_p))] if os.path.exists(pre_p) else [])
    s = max(1, min(scale, max_side // max(cs.shape)))
    ims = []
    for name, a in panels:
        im = Image.fromarray(stretch(a)).resize((a.shape[1] * s, a.shape[0] * s), Image.NEAREST).convert('RGB')
        bb = os.path.join(cut_dir, stem + '_bbox.npy')
        if os.path.exists(bb):
            x0, y0, x1, y1 = np.load(bb).ravel()[:4]
            ImageDraw.Draw(im).rectangle([x0 * s, y0 * s, x1 * s, y1 * s], outline=(255, 0, 0), width=2)
        ims.append((name, im))
    head = 14 * len(lines) + 20
    W = max(sum(i.width for _, i in ims) + 8 * (len(ims) - 1), 7 * max(len(l) for l in lines) + 8)
    H = max(i.height for _, i in ims) + head
    canvas = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(canvas)
    for k, l in enumerate(lines): d.text((4, 3 + 14 * k), l, fill=(0, 0, 0))
    x = 0
    for name, i in ims:
        d.text((x + 2, head - 16), name, fill=(90, 90, 90)); canvas.paste(i, (x, head)); x += i.width + 8
    canvas.save(out_png)


def reasons(folder):
    """stem -> screen_reason from the detections and screened-out CSVs of one run."""
    out = {}
    for pat, tag in (('detections_*_DET*.csv', 'sent'), ('screened_out_*_DET*.csv', 'held back')):
        for f in glob.glob(os.path.join(folder, pat)):
            d = pd.read_csv(f)
            if 'cutout_stem' not in d.columns:
                fits_id = re.search(r'_(\d+_\d+_\d+)_DET(\d+)\.csv$', f)
                if not fits_id or not {'det_xmin', 'det_xmax', 'det_ymin', 'det_ymax'} <= set(d.columns):
                    continue
                d['cutout_stem'] = [f"cutout_{fits_id.group(1)}_DET{fits_id.group(2)}_{a}_{b}_{c}_{e}"
                                    for a, b, c, e in zip(d.det_xmin, d.det_xmax, d.det_ymin, d.det_ymax)]
            for s, r in zip(d['cutout_stem'], d.get('screen_reason', [''] * len(d))):
                out.setdefault(s, f"{tag}: {r}")
    return out


def collect(src_dir, wanted, dest, why_normal, why_aggr):
    if os.path.isdir(dest): shutil.rmtree(dest)
    os.makedirs(dest)
    rows = []
    for st in sorted(wanted):
        for f in glob.glob(os.path.join(src_dir, st + '*.npy')):
            if os.path.basename(f) == st + '.npy' or re.search(AUX, f):
                shutil.copy2(f, dest)
        rn = why_normal.get(st, '-'); ra = why_aggr.get(st, 'no cutout (tiny on a spectrum, or trimmed away)')
        m = re.search(r'_(\d+)_(\d+)_(\d+)_(\d+)$', st)
        preview(src_dir, st, os.path.join(dest, st + '.png'), [st, f"normal run:     {rn}", f"aggressive run: {ra}"])
        rows.append({'cutout_stem': st, 'normal_run': rn, 'aggressive_run': ra,
                     **(dict(zip(['det_xmin', 'det_xmax', 'det_ymin', 'det_ymax'], map(int, m.groups()))) if m else {})})
    pd.DataFrame(rows).to_csv(os.path.join(dest, 'index.csv'), index=False)
    return len(rows)


def main(det_dir='.'):
    det_dir = os.path.abspath(det_dir)
    normal_cut = os.path.join(det_dir, 'cutouts')
    aggr_dir = os.path.join(det_dir, 'aggressive'); aggr_cut = os.path.join(aggr_dir, 'cutouts')
    if not os.path.isdir(normal_cut) or not os.path.isdir(aggr_cut):
        sys.exit(f"need both {normal_cut} and {aggr_cut} - run the normal and the aggressive cutouts step first")
    n, a = stems(normal_cut), stems(aggr_cut)
    why_n, why_a = reasons(det_dir), reasons(aggr_dir)
    k1 = collect(normal_cut, n - a, os.path.join(det_dir, 'dropped_by_aggressive'), why_n, why_a)
    k2 = collect(aggr_cut, a - n, os.path.join(det_dir, 'only_in_aggressive'), why_n, why_a)
    print(f"normal {len(n)} cutouts, aggressive {len(a)}, in both {len(n & a)}")
    print(f"dropped_by_aggressive/: {k1} cutouts (+ .png previews, index.csv)")
    print(f"only_in_aggressive/:    {k2} cutouts")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')

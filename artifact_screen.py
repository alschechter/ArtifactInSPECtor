"""
artifact_screen.py -- decide which detection boxes go to SAM.

Policy: EVERYTHING goes to SAM except boxes that are confidently continuum only
(poorly subtracted spectra). Zeroth orders, emission-line candidates, stars, snowballs, ghosts,
trails, other artifacts, unclassified and mixed boxes are all sent.

Uses the maps written by euclid_mask.py:
  label_map.npy        0 clean  1 zeroth order  2 emission-line candidate  3 continuum residual
                       4 other artifact (stars, spikes, ghosts, trails, columns, residual blobs)  5 snowball
  zo_candidates.npy    every pair-like blob (two peaks on one row at the zeroth-order separation),
                       including pairs sitting on spectra that label_map calls continuum
  compact_sources.npy  every blob found, whatever its class

Independently of the masks, find_blobs() tests the light profile directly: round blobs (>= 4 px
tall at half maximum, not stretched along the rows) and pairs of them at the zeroth-order
separation. Any box holding one is sent to SAM -- this also rescues tiny detections.

A box is screened out only when ALL of these hold:
  * at most max_keep_px pixels of zeroth order, emission, artifact or snowball, and of zeroth-order
    candidates, inside the box;
  * continuum makes up >= min_drop_frac of the recognised pixels and covers >= min_coverage of the box;
  * no round blob (>= 4 px tall, not stretched along the rows) from compact_sources.npy in the box --
    spectrum segments are 1-3 px tall, a zeroth order or other source on a spectrum is not.
"""
import numpy as np
from scipy import ndimage as ndi

ZO, EM, CONT, ART, SNOW = 1, 2, 3, 4, 5

DEFAULTS = dict(
    pad=2,               # px added around the box when reading the maps
    max_keep_px=2,       # more than this many non-continuum object pixels -> send to SAM
    min_coverage=0.10,   # continuum pixels must cover >= this fraction of the box
    min_drop_frac=0.95,  # and make up >= this fraction of the recognised pixels
    min_blob_h=4,        # a round blob at least this tall inside the box -> send to SAM
)


def _has_round_blob(sub, min_h=4):
    lab, n = ndi.label(sub)
    for sl in ndi.find_objects(lab):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if h >= min_h and w <= 2.5 * h + 2:
            return True
    return False


def _find_blobs_scale(img, pair_range=(6, 16), snr=5.0, min_h=4, max_dark=0.30, valid=None, smooth=1.0, max_wh=2.0):
    """Image-based test for round blobs and zeroth-order pairs, independent of the mask classes
    (which can call a pair sitting on a spectrum 'continuum').
    Peaks of the lightly smoothed image above `snr` x noise are measured at half maximum:
      * round blob: half-max region >= min_h px tall and not stretched along the rows
        (w <= 2h + 3) -- spectrum dashes are thin and long, zeroth orders / snowballs are not;
      * pair: two round blobs on the same row (|dy| <= 2), separated by pair_range px along the
        dispersion, brightness within x8, and a dip between them.
    Poorly subtracted spectra alternate bright and very dark pixels along their rows, so a single
    round blob only counts if the pixels beside it in its own rows (+-14 px, the blob excluded)
    are not mostly dark: fraction below -2 sigma <= max_dark (sky ~0.08, zeroth orders and
    snowballs <= ~0.2-0.3, spectrum pieces ~0.3-0.5). Pairs count regardless.
    Returns (blob_map, pair_map) boolean images (half-max footprints)."""
    a = np.nan_to_num(np.asarray(img, dtype=np.float64))
    bg = np.median(a if valid is None else a[valid])
    sig = 1.4826 * np.median(np.abs((a if valid is None else a[valid]) - bg)) + 1e-12
    z = (a - bg) / sig
    s = ndi.gaussian_filter(a - bg, smooth)
    ref = s if valid is None else s[valid]
    sn = 1.4826 * np.median(np.abs(ref - np.median(ref))) + 1e-12
    pk = (s == ndi.maximum_filter(s, 5)) & (s > snr * sn)
    if valid is not None: pk &= valid
    H, W = a.shape
    blob_map = np.zeros((H, W), bool); pair_map = np.zeros((H, W), bool)
    blobs = []
    for y, x in zip(*np.nonzero(pk)):
        y0, y1, x0, x1 = max(0, y - 15), min(H, y + 16), max(0, x - 15), min(W, x + 16)
        reg = s[y0:y1, x0:x1] > s[y, x] / 2
        lab, _ = ndi.label(reg); m = lab == lab[y - y0, x - x0]
        yy, xx = np.nonzero(m); h = np.ptp(yy) + 1; w = np.ptp(xx) + 1
        if h >= min_h and w <= max_wh * h + 3:
            # darkness beside the blob in its own rows
            ry0, ry1 = y0 + yy.min(), y0 + yy.max() + 1
            rx0, rx1 = max(0, x0 + xx.min() - 14), min(W, x0 + xx.max() + 15)
            own = np.zeros((ry1 - ry0, rx1 - rx0), bool)
            own[:, max(0, x0 + xx.min() - 3 - rx0):x0 + xx.max() + 4 - rx0] = True
            side = z[ry0:ry1, rx0:rx1][~own]
            dark = float((side < -2).mean()) if side.size > 10 else 0.0
            blobs.append((x, y, s[y, x], (y0, y1, x0, x1), m, dark))
            if dark <= max_dark:
                blob_map[y0:y1, x0:x1] |= m
    lo, hi = pair_range
    by_row = sorted(range(len(blobs)), key=lambda i: blobs[i][0])
    xs = np.array([blobs[i][0] for i in by_row]) if blobs else np.array([])
    for ii, i in enumerate(by_row):
        xi, yi, vi = blobs[i][:3]
        for jj in range(ii + 1, len(by_row)):
            j = by_row[jj]; xj, yj, vj = blobs[j][:3]
            if xj - xi > hi: break
            if xj - xi < lo or abs(yj - yi) > 2 or not (0.125 <= vi / vj <= 8): continue
            if s[(yi + yj) // 2, (xi + xj) // 2] > 0.75 * min(vi, vj): continue
            for k in (i, j):
                (y0, y1, x0, x1), m = blobs[k][3], blobs[k][4]; pair_map[y0:y1, x0:x1] |= m
    return blob_map, pair_map


def find_blobs(img, pair_range=(6, 16), snr=5.0, max_dark=0.30, valid=None):
    """Two scales: compact blobs (1 px smoothing, >= 4 px tall at half maximum) and faint, more
    diffuse blobs such as faint snowballs (2 px smoothing, >= 6 px tall)."""
    b1, p1 = _find_blobs_scale(img, pair_range, snr, 4, max_dark, valid, 1.0)
    b2, p2 = _find_blobs_scale(img, pair_range, snr, 6, max_dark, valid, 2.0, 1.4)   # rounder: no faint spectrum dashes
    return b1 | b2, p1 | p2


def screen_box(bbox, label_map, compact=None, zo_cand=None, blob_map=None, pair_map=None, **kw):
    """bbox = (xmin, ymin, xmax, ymax) in detector pixels (same frame as the maps).
    Returns (send_to_sam: bool, reason: str, fractions: dict)."""
    p = {**DEFAULTS, **kw}
    H, W = label_map.shape
    xmin, ymin, xmax, ymax = [int(round(v)) for v in bbox]
    x0, y0 = max(0, xmin - p['pad']), max(0, ymin - p['pad'])
    x1, y1 = min(W, xmax + p['pad'] + 1), min(H, ymax + p['pad'] + 1)
    reg = label_map[y0:y1, x0:x1]
    counts = np.bincount(reg.ravel(), minlength=6)[:6]
    labelled = int(counts[1:].sum())
    fr = {name: float(counts[k] / labelled) if labelled else 0.0
          for k, name in ((ZO, 'zo'), (EM, 'emission'), (CONT, 'continuum'), (ART, 'artifact'), (SNOW, 'snowball'))}
    fr['coverage'] = float(labelled / max(reg.size, 1))

    other = {'zeroth_order': counts[ZO], 'emission_line': counts[EM],
             'artifact': counts[ART], 'snowball': counts[SNOW]}
    if sum(other.values()) > p['max_keep_px']:
        return True, max(other, key=other.get), fr
    if zo_cand is not None and int(zo_cand[y0:y1, x0:x1].sum()) > p['max_keep_px']:
        return True, 'zeroth_order_candidate', fr
    if pair_map is not None and pair_map[y0:y1, x0:x1].any():
        return True, 'zeroth_order_pair', fr
    if blob_map is not None and blob_map[y0:y1, x0:x1].any():
        return True, 'round_blob', fr
    if labelled == 0:
        return True, 'unclassified', fr
    if counts[CONT] / labelled >= p['min_drop_frac'] and counts[CONT] / max(reg.size, 1) >= p['min_coverage']:
        if compact is not None and _has_round_blob(compact[y0:y1, x0:x1], p['min_blob_h']):
            return True, 'blob_on_continuum', fr
        return False, 'continuum', fr
    return True, 'mixed_or_uncertain', fr


def screen_boxes(boxes, label_map, compact=None, zo_cand=None, blob_map=None, pair_map=None, **kw):
    return [screen_box(b, label_map, compact, zo_cand, blob_map, pair_map, **kw) for b in boxes]

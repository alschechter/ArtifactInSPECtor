"""
artifact_screen.py -- decide which SEP detection boxes go to SAM.

Uses the pixel classes from euclid_mask.py (label_map.npy):
    0 clean  1 zeroth order  2 emission-line candidate  3 continuum residual
    4 other artifact (stars, spikes, ghosts, trails, columns, residual blobs)  5 snowball

Policy: EVERYTHING goes to SAM except boxes that are confidently only emission-line and/or
continuum pixels. A box is screened out only when
  * it contains (almost) no zeroth-order, artifact or snowball pixels, and
  * the pixels the mask code recognised inside it are emission/continuum, and
  * those pixels cover enough of the box that the verdict can be trusted.
Unclassified boxes (nothing recognised) are sent to SAM, since they could be artifacts the mask missed.
"""
import numpy as np

ZO, EM, CONT, ART, SNOW = 1, 2, 3, 4, 5

DEFAULTS = dict(
    pad=2,              # px added around the SEP box when reading the label map
    max_keep_px=2,      # more than this many ZO/artifact/snowball pixels -> send to SAM
    min_coverage=0.10,  # emission+continuum pixels must cover >= this fraction of the box
    min_drop_frac=0.95, # and make up >= this fraction of the recognised pixels
    cont_min_aspect=2.5,  # a continuum-only box must be elongated; a compact box on a continuum may be a
                          # zeroth order or other source sitting on the spectrum -> sent to SAM
)


def screen_box(bbox, label_map, compact=None, **kw):
    """bbox = (xmin, ymin, xmax, ymax) in detector pixels (same frame as label_map).
    compact = optional compact_sources.npy map from euclid_mask (every blob, whatever its class):
    used to decide whether a compact box on a continuum holds a real blob (e.g. a zeroth order
    sitting on the spectrum) or is just a short piece of the spectrum itself.
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

    keep_px = int(counts[ZO] + counts[ART] + counts[SNOW])
    if keep_px > p['max_keep_px']:
        dominant = max(('zeroth_order', counts[ZO]), ('artifact', counts[ART]), ('snowball', counts[SNOW]), key=lambda t: t[1])[0]
        return True, dominant, fr
    if labelled == 0:
        return True, 'unclassified', fr
    drop_px = counts[EM] + counts[CONT]
    if drop_px / labelled >= p['min_drop_frac'] and drop_px / max(reg.size, 1) >= p['min_coverage']:
        if counts[EM] > counts[CONT]:
            return False, 'emission_line', fr
        w, h = xmax - xmin + 1, ymax - ymin + 1
        if max(w, h) / max(min(w, h), 1) >= p['cont_min_aspect']:
            return False, 'continuum', fr
        if compact is None or _has_round_blob(compact[y0:y1, x0:x1]):
            return True, 'compact_on_continuum', fr
        return False, 'continuum', fr        # a short piece of the spectrum, no blob on it
    return True, 'mixed_or_uncertain', fr


def _has_round_blob(sub, min_h=4):
    """True if the compact-source map holds a blob at least min_h px tall and not stretched
    along the rows (spectrum segments are 1-3 px tall)."""
    from scipy import ndimage as ndi
    lab, n = ndi.label(sub)
    for sl in ndi.find_objects(lab):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if h >= min_h and w <= 2.5 * h + 2:
            return True
    return False


def screen_boxes(boxes, label_map, compact=None, **kw):
    return [screen_box(b, label_map, compact, **kw) for b in boxes]

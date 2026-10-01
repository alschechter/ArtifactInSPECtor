# Continuum-only screening before SAM

Only one thing is kept out of SAM: detection boxes that are confidently **continuum only**
(poorly subtracted spectra). Everything else -- zeroth orders, emission-line candidates, stars,
snowballs, ghosts, trails, other artifacts, unclassified and mixed boxes -- gets a cutout and goes
to SAM. Cutouts are taken in the dispersion frame (spectra exactly along rows, no interpolation).

Files (all next to `Screen_Cutouts_Pipeline.py`): `euclid_mask.py`, `artifact_screen.py`,
`dispersion_frame.py`. Extra packages: `opencv-python-headless`, `scikit-image`.

## Per detector
1. The pipeline detects and merges with SEP exactly as before and writes the pixel-exact
   `<id>_DET<nn>_noboxes.png`.
2. `euclid_mask.py` runs on that PNG -> `euclid_masks/` (`label_map.npy`, `zo_candidates.npy`,
   `compact_sources.npy`, overlay with legend, `run_info.json`). ~1-1.5 min.
3. A light-profile test runs on the continuum-subtracted image itself (`artifact_screen.find_blobs`),
   independent of the mask classes:
   * round blob: a peak > 5 sigma whose half-maximum region is >= 4 px tall and not stretched along
     the rows (1 px smoothing), or >= 6 px tall and rounder (2 px smoothing, faint diffuse blobs);
     spectra alternate bright and very dark pixels, so a single blob only counts if <= 30 % of the
     pixels beside it in its own rows are below -2 sigma (sky ~8 %, zeroth orders/snowballs ~13 %);
   * zeroth-order pair: two round blobs on one row separated by the measured pair separation
     (+-1 px; ~11-13 px in the test images), with a dip between them -- counted regardless of darkness.
   Any box holding a round blob or pair goes to SAM, **including tiny detections** (< TINY_MAX_DIM px),
   which otherwise never get a cutout.
4. Each merged box is screened (`artifact_screen.screen_box`). It is screened out only if ALL hold:
   * at most 2 pixels of zeroth order, emission line, artifact or snowball, and of zeroth-order
     candidates (pair-like blobs, also those sitting on spectra), inside the box (+2 px);
   * continuum is >= 95 % of the recognised pixels and covers >= 10 % of the box;
   * no round blob >= 4 px tall (not stretched along the rows) inside the box.
5. Long spectrum boxes are trimmed: a merged box that is still sent but is long and thin
   (width >= 80 px and >= 4 x height) and >= 50 % continuum is replaced by compact boxes (+6 px)
   around what actually sits on it -- round blobs, pairs, zeroth orders (and candidates),
   snowballs, star/trail/ghost/column pixels. Long chains along the spectrum itself are not kept,
   and a box sent only for spectrum knots (round-blob / emission reasons with nothing compact on
   it) is dropped. Trimmed boxes appear in `screened_out_*.csv` as `continuum_trimmed` with
   `n_pieces_kept`, and as thin boxes in the QA images.
6. Cutouts are written only for boxes sent to SAM; `detections_<id>_DET<nn>.csv` holds only those
   rows (SAM and PreContSub run unchanged); screened-out rows go to `screened_out_<id>_DET<nn>.csv`.
   Both have `sent_to_sam`, `screen_reason` and `frac_*` columns.

## QA images
* `<id>_DET<nn>_sam_boxes.png` (detector frame): cyan = cutout for SAM, magenta = no cutout
  (continuum only, or tiny < TINY_MAX_DIM px).
* `<id>_DET<nn>_screened.png` (dispersion frame, same as cutouts): red = sent to SAM,
  orange = screened out as continuum, grey = tiny.
* `<id>_DET<nn>_dispersion_frame.png/.json`: the detector in the cutout frame and its parameters.
* `euclid_masks/classified_overlay_legend.png`: the full mask classification.

## Options
`--no-screen` (everything to SAM), `--no-rotate` (cut out in the detector frame),
`--dispersion-angle DEG` (fix the angle instead of measuring it). Thresholds: `artifact_screen.DEFAULTS`.

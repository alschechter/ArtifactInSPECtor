# Artifact screening before SAM

Drop-in additions to ArtifactInSPECtor so that only probable artifacts are cut out and sent to SAM.

Copy `euclid_mask.py`, `artifact_screen.py` and the updated `Cutouts_Pipeline.py` / `runCutouts.sh`
into the repo root (next to the existing scripts). Nothing else changes: `run_sam_minthresh.sh` and
`runPreContSub.sh` run as before.

Extra Python packages: `opencv-python-headless`, `scikit-image` (scipy/numpy/pillow are already used).

## What happens per detector
1. The pipeline writes `<fits_id>_DET<nn>_noboxes.png` as before (pixel-exact, detector coordinates).
2. `euclid_mask.py` runs on that PNG and writes `euclid_masks/` next to it (label map, class masks,
   overlay with legend, run_info.json). ~1–1.5 min per detector.
3. Every merged SEP box is screened with `artifact_screen.screen_box`. **Everything goes to SAM
   except boxes that are confidently only emission-line and/or continuum pixels.** A box is
   screened out only when all of these hold:
   * it contains no more than 2 zeroth-order, star/ghost/trail/other-artifact or snowball pixels;
   * emission + continuum make up >= 95% of the pixels the mask code recognised in the box, and
     cover >= 10% of the box;
   * for continuum: the box is elongated (aspect >= 2.5). A compact box on a continuum could be a
     zeroth order or other source sitting on the spectrum, so it is sent (`compact_on_continuum`).
   Boxes with nothing recognised (`unclassified`) are sent too.
4. Cutouts are written only for SAM-bound boxes. `detections_<id>_DET<nn>.csv` holds only those rows
   (so the SAM and PreContSub steps are unchanged); screened-out rows go to
   `screened_out_<id>_DET<nn>.csv` with the same columns. Both have `sent_to_sam`,
   `screen_reason` (zeroth_order / artifact / snowball / compact_on_continuum / unclassified /
   mixed_or_uncertain / continuum / emission_line) and `frac_*` columns.
5. QA image `<fits_id>_DET<nn>_screened.png`: red = to SAM, orange = screened out as continuum,
   green = screened out as emission line.

## Options (Cutouts_Pipeline.py)
* `--no-screen`          old behaviour, everything to SAM
* `--mask-angle DEG`     force the dispersion angle instead of measuring it

Thresholds live in `artifact_screen.DEFAULTS`.

## Cutouts in the dispersion frame (spectra exactly horizontal)
By default every cutout (and hence every SAM mask) is taken from a copy of the detector image in
which the dispersion direction runs exactly along rows. **No pixel value is interpolated**:
* quarter turns (e.g. RGS270) use `np.rot90` (exact);
* the residual grism tilt is removed by moving whole detector columns up or down by an integer
  number of pixels (a column shear). Every cutout pixel is an original detector pixel, only moved.
  Spectra then follow a +/-0.5 px staircase along rows, and objects are sheared by the tilt angle
  (about 4 deg for the tilted grisms).
* The angle is first measured by `euclid_mask` and then refined from the continuum residuals
  themselves (length-weighted median slope of the long, thin streaks), typically to ~0.03 deg.
  Pixels created outside the detector by the shear are filled with 0 (continuum-subtracted
  cutout) or the image median (pre-subtraction cutout).

Per detector: `<id>_DET<nn>_dispersion_frame.json` (angle, quarter turns, shear, padding) and
`<id>_DET<nn>_dispersion_frame.png` (the whole detector in that frame, for a visual check).
CSV columns: `det_*` stay in detector pixels; `frame_*`, `cutout_*` and `local_*` are in the
dispersion frame; `frame_angle_deg`, `frame_k90`, `frame_shear_tan`, `frame_pad` describe it.
`dispersion_frame.DispersionFrame(shape, angle).to_detector(x, y)` maps frame pixels (e.g. a SAM
mask placed at cutout_x0/y0) back to detector pixels.

Options: `--no-rotate` (cut out in the detector frame, old behaviour);
`--dispersion-angle DEG` (use this angle instead of measuring it; alias `--mask-angle`).

## Screening happens BEFORE merging (update)
Every raw SEP detection (hot, cold and dark passes) is screened on its own, before the 30/10 px
merge, so a spectrum can no longer be merged together with a nearby zeroth order or artifact into
one large box:
* dropped: detections that are only continuum and/or emission-line pixels (rules above). A compact
  detection on a continuum is dropped too **unless** `compact_sources.npy` (every blob found by
  euclid_mask, whatever its class) shows a round blob at least 4 px tall inside it -- e.g. a
  zeroth order sitting on the spectrum. Short spectrum segments are 1-3 px tall and do not count.
* split: a long detection that is >=80 % continuum but touches zeroth-order / artifact / snowball
  pixels is replaced by compact boxes around those pixels (+5 px); the spectrum part is dropped
  (`continuum_split` in the screened-out CSV, with `n_pieces_kept`).
* everything else is kept and merged as before; all merged boxes go to SAM.
`screened_out_<id>_DET<nn>.csv` now lists the raw detections (sep_pass, det bbox, area, reason).
In the QA image red = merged boxes sent to SAM, orange/green = raw detections screened out.

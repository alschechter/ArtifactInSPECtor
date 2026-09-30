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

# ArtifactInSPECtor
zooniverse Artifact InSPECtor code

With a Euclid detector fits file, run:

1. runCutouts.sh
2. run_sam_minthresh.sh
3. run_PreContSub.sh

This runs through background subtraction, pixel masking, source detection, cutouts, segmentation maps, making the zooniverse subject images, and encoding it all into a csv.

To get AuxOL segmentation masks for a folder of cutouts (after steps 1-2 above have produced `cutouts/` and `sam_results/` in it), run the self-contained script in `temp_upload/` (bundles the AuxOL code, no external repo needed; the trained checkpoint is downloaded on first use and cached locally, so it's not stored in this repo):

```
python temp_upload/run_auxol_segmentation.py --dir <folder>
```

**AuxOL weights:** [huggingface.co/BCN001/Artifact-Inspector-AUXOL](https://huggingface.co/BCN001/Artifact-Inspector-AUXOL) hosts `auxol_train_test_state.pt` (~370MB, the AuxOL UNet + fusion state used above). `run_auxol_segmentation.py` fetches it automatically via `huggingface_hub` (`pip install huggingface_hub` if you don't have it) the first time it runs. To download it manually instead:

```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="BCN001/Artifact-Inspector-AUXOL", filename="auxol_train_test_state.pt")
```

and pass the resulting path to the script with `--state <path>`.

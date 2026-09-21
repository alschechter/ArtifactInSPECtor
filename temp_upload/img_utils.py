"""Small shared image helpers with no matplotlib backend side effects,
so scripts that need an interactive backend (correction_gui.py) can import
them without inheriting run_sam.py's matplotlib.use('Agg')."""

import numpy as np
from astropy.visualization import ZScaleInterval

zscale = ZScaleInterval()


def to_rgb(data: np.ndarray) -> np.ndarray:
    """Convert 2-D float32 science array to uint8 H×W×3 RGB via ZScale."""
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros((*data.shape, 3), dtype=np.uint8)
    try:
        vmin, vmax = zscale.get_limits(finite)
    except Exception:
        vmin, vmax = np.nanpercentile(data, 1), np.nanpercentile(data, 99)
    normed = np.clip((data - vmin) / (vmax - vmin + 1e-12), 0, 1)
    normed = np.nan_to_num(normed, nan=0.0)
    gray = (normed * 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)

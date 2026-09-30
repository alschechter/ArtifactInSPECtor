"""
dispersion_frame.py -- put a detector image into a frame where the dispersion direction
(the poorly subtracted spectra) runs exactly along image rows, WITHOUT interpolating pixel values.

How
  1. Multiples of 90 deg are done with np.rot90 (exact).
  2. The remaining small tilt (e.g. the +/-4 deg grism positions) is removed with an integer
     column shear: every detector column is moved up/down by a whole number of pixels,
         y' = y + round(-(x - xc) * tan(delta)) + pad
     so a spectrum at angle delta lands on one row (to within +/-0.5 px everywhere).

Every output pixel is an original pixel value, moved but never resampled or averaged.
The price is geometric: within a cutout, objects are sheared by delta (about 4 deg for the
tilted grisms), and spectra follow a +/-0.5 px staircase instead of a sub-pixel straight line.
Pixels outside the detector (created by the shear) are filled with `fill`.
"""
import numpy as np
from scipy import ndimage as ndi


class DispersionFrame:
    def __init__(self, shape, angle_deg):
        """shape = (H, W) of the detector image; angle_deg = dispersion angle measured in the
        detector image (x right, y down, positive angle = spectra run down to the right)."""
        self.H0, self.W0 = shape
        a = float(angle_deg) % 180.0
        # quarter turns so the dispersion ends up within 45 deg of horizontal
        self.k90 = 1 if 45.0 <= a < 135.0 else 0
        dx, dy = np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))
        if self.k90:                          # np.rot90 (k=1): (x, y) -> (y, W-1-x); direction (dx,dy) -> (dy,-dx)
            dx, dy = dy, -dx
        d = np.degrees(np.arctan2(dy, dx))
        d = (d + 90.0) % 180.0 - 90.0         # residual tilt in (-90, 90]
        self.delta = d
        self.t = np.tan(np.deg2rad(d))
        self.Hr, self.Wr = (self.W0, self.H0) if self.k90 else (self.H0, self.W0)
        self.xc = (self.Wr - 1) / 2.0
        self.shift = np.round(-(np.arange(self.Wr) - self.xc) * self.t).astype(int)   # per column
        self.pad = int(np.abs(self.shift).max()) if self.shift.size else 0
        self.shape = (self.Hr + 2 * self.pad, self.Wr)

    # ---- images ------------------------------------------------------------
    def forward(self, img, fill=np.nan):
        """Detector image -> dispersion frame (pure re-indexing, no interpolation)."""
        a = np.rot90(img, self.k90) if self.k90 else img
        out = np.full(self.shape, fill, dtype=np.result_type(a.dtype, np.float32) if np.isnan(fill) else a.dtype)
        for x in range(self.Wr):
            y0 = self.shift[x] + self.pad
            out[y0:y0 + self.Hr, x] = a[:, x]
        return out

    def inverse(self, img):
        """Dispersion-frame image (e.g. a full-frame mask) -> detector frame."""
        a = np.empty((self.Hr, self.Wr), dtype=img.dtype)
        for x in range(self.Wr):
            y0 = self.shift[x] + self.pad
            a[:, x] = img[y0:y0 + self.Hr, x]
        return np.rot90(a, -self.k90) if self.k90 else a

    # ---- coordinates (pixel centres, integer or float) -----------------------
    def _rot(self, x, y):
        return (y, self.W0 - 1 - x) if self.k90 else (x, y)

    def _unrot(self, x, y):
        return (self.W0 - 1 - y, x) if self.k90 else (x, y)

    def to_frame(self, x, y):
        """Detector (x, y) -> dispersion frame (x', y'). Works on scalars or arrays."""
        xr, yr = self._rot(np.asarray(x), np.asarray(y))
        col = np.clip(np.round(xr).astype(int), 0, self.Wr - 1)
        return xr, yr + self.shift[col] + self.pad

    def to_detector(self, xf, yf):
        """Dispersion frame (x', y') -> detector (x, y)."""
        xf = np.asarray(xf); yf = np.asarray(yf)
        col = np.clip(np.round(xf).astype(int), 0, self.Wr - 1)
        return self._unrot(xf, yf - self.shift[col] - self.pad)

    def box_to_frame(self, xmin, ymin, xmax, ymax):
        """Axis-aligned detector box -> smallest axis-aligned box containing it in the frame."""
        xs = np.array([xmin, xmax, xmin, xmax]); ys = np.array([ymin, ymin, ymax, ymax])
        xr, yr = self._rot(xs, ys)
        x0, x1 = int(np.floor(xr.min())), int(np.ceil(xr.max()))
        y0, y1 = int(np.floor(yr.min())), int(np.ceil(yr.max()))
        cols = self.shift[np.clip(np.arange(x0, x1 + 1), 0, self.Wr - 1)]
        return x0, y0 + int(cols.min()) + self.pad, x1, y1 + int(cols.max()) + self.pad

    def describe(self):
        return dict(k90=int(self.k90), residual_tilt_deg=float(self.delta), shear_tan=float(self.t),
                    pad=int(self.pad), x_centre=float(self.xc), frame_shape=list(self.shape),
                    detector_shape=[int(self.H0), int(self.W0)])


def refine_dispersion_angle(img, label_map, angle0, min_len=200, max_thick=7):
    """Precise dispersion angle measured from the continuum residuals themselves.
    For every long, thin continuum piece in label_map (class 3), fit the |residual|-weighted
    centre row of each column with a line; return the length-weighted median angle.
    Returns (angle_deg, n_streaks, scatter_deg). Falls back to angle0 if < 5 streaks."""
    F = DispersionFrame(img.shape, angle0)
    k = F.k90
    base = angle0 - F.delta                    # quarter-turn part of the angle
    w = np.rot90(np.abs(img.astype(np.float64) - np.nanmedian(img)), k)
    cont = np.rot90(label_map == 3, k)
    lab, n = ndi.label(cont, structure=np.ones((3, 3)))
    ang, wt = [], []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        yy, xx = np.nonzero(lab[sl] == i)
        if np.ptp(xx) < min_len: continue
        th = np.bincount(xx - xx.min()); th = th[th > 0]
        if np.median(th) > max_thick: continue          # several streaks merged
        cols, cys = [], []
        for x in np.unique(xx):
            r = yy[xx == x]; ys = np.arange(r.min() - 1, r.max() + 2) + sl[0].start
            ys = ys[(ys >= 0) & (ys < w.shape[0])]; v = w[ys, x + sl[1].start]
            if v.sum() > 0: cols.append(x); cys.append((v * ys).sum() / v.sum())
        cols, cys = np.array(cols), np.array(cys)
        if len(cols) < min_len // 2: continue
        p = np.polyfit(cols, cys, 1); r = cys - np.polyval(p, cols)
        keep = np.abs(r) < 3 * np.std(r) + 1e-9; p = np.polyfit(cols[keep], cys[keep], 1)
        ang.append(np.degrees(np.arctan(p[0]))); wt.append(np.ptp(cols))
    if len(ang) < 5:
        return float(angle0), len(ang), float('nan')
    ang, wt = np.array(ang), np.array(wt)
    o = np.argsort(ang); cw = np.cumsum(wt[o]); med = ang[o][np.searchsorted(cw, cw[-1] / 2)]
    scatter = 1.4826 * np.median(np.abs(ang - med))
    return float(base + med), len(ang), float(scatter)

#!/usr/bin/env python3
"""
euclid_mask.py -- automatic artifact / source masking for Euclid NISP slitless
(grism) residual images, any grism or tilt (RGS000, RGS180, RGS270, BGS000, +/-4 deg).

Everything image-specific is measured from the data:
  * detector footprint (rotated frames with flat padding)       -> find_footprint
  * background level, noise, saturation level                  -> Frame
  * dispersion direction (0/90 deg +/- tilt)                   -> find_dispersion
  * zeroth-order pair separation                               -> measure_pair_offset
  * bright stars, their diffraction-spike pattern and lengths  -> find_stars
  * trails, ghost arcs, detector column pattern                -> ridge / column search

Classes in label_map.npy
  0 clean   1 zeroth order (paired blobs)   2 emission-line candidate (small single blob)
  3 continuum residual   4 other artifact   5 snowball (large single / ring blob)
artifact_type_map.npy: 1 star (core+spikes+halo), 2 trail, 3 ghost/arc, 4 detector
column pattern, 5 extended/dipole residual, 6 compact artifact on the above.

Usage:  python euclid_mask.py IMAGE.png --out OUTDIR [--angle DEG] [--em-max 60] [--cont-min-len 40]
"""
import argparse, json, os, csv, time, warnings
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.filters import apply_hysteresis_threshold as hyst, sato
from skimage.morphology import remove_small_objects, skeletonize

# rays that leave the image give empty/all-NaN slices; that is expected and handled
warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='All-NaN slice encountered')

E = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
def mad(x): x = x[np.isfinite(x)]; m = np.median(x); return m, 1.4826 * np.median(np.abs(x - m)) + 1e-9
def keep_big(m, n):
    lab, k = ndi.label(m)
    if k == 0: return m
    s = ndi.sum(m, lab, range(1, k + 1)); return np.isin(lab, 1 + np.nonzero(s >= n)[0])
def log(*a): print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)

# --------------------------------------------------------------------------- frame / geometry
def find_footprint(a, edge):
    m = cv2.blur(a, (5, 5)); v = cv2.blur(a * a, (5, 5)) - m * m
    tex = cv2.morphologyEx((v > 1.0).astype(np.uint8), cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    lab, n = ndi.label(tex)
    if n == 0: return np.ones_like(a, bool), np.ones_like(a, bool)
    big = np.argmax(ndi.sum(tex, lab, range(1, n + 1))) + 1
    full = ndi.binary_fill_holes(lab == big)
    return full, cv2.erode(full.astype(np.uint8), np.ones((2 * edge + 1, 2 * edge + 1), np.uint8)) > 0

def line_kernel(ang, L, th=1):
    k = np.zeros((L, L), np.float32); c = L // 2; t = np.deg2rad(ang)
    for s in np.linspace(-c, c, 4 * L):
        for o in range(-(th // 2), th // 2 + 1):
            x = c + s * np.cos(t) - o * np.sin(t); y = c + s * np.sin(t) + o * np.cos(t)
            k[int(round(y)), int(round(x))] = 1
    return k / k.sum()

def find_dispersion(zc, valid):
    """Dispersion axis = direction of the long thin residual streaks. Candidates are the
    two grism axes (0 and 90 deg) with up to +/-6 deg tilt."""
    sub = zc[::2, ::2]; vs = valid[::2, ::2]
    scores = {}
    for base in (0, 90):
        for d in np.arange(-6, 6.5, 1.0):
            r = cv2.filter2D(sub, -1, line_kernel(base + d, 51))[vs]
            thr = np.percentile(r, 99.5); scores[float(base + d)] = float(r[r >= thr].mean() - np.median(r))
    best = max(scores, key=lambda k: (round(scores[k], 3), -abs(((k + 45) % 90) - 45)))
    # refine to 0.25 deg
    fine = {}
    for d in np.arange(best - 1, best + 1.01, 0.25):
        r = cv2.filter2D(sub, -1, line_kernel(d, 51))[vs]; thr = np.percentile(r, 99.5)
        fine[float(d)] = float(r[r >= thr].mean() - np.median(r))
    best = max(fine, key=lambda k: (round(fine[k], 3), -abs(((k + 45) % 90) - 45)))
    return best, scores

class Rot:
    """Maps the image into a frame where the dispersion runs along +x rows, and back."""
    def __init__(self, shape, angle):
        self.H, self.W = shape; self.k90 = 0; a = angle
        if abs(((a - 90 + 90) % 180) - 90) < 20: self.k90 = 1; a = a - 90
        a = ((a + 90) % 180) - 90
        self.small = a if abs(a) >= 0.4 else 0.0
        h, w = (self.W, self.H) if self.k90 else (self.H, self.W)
        self.h0, self.w0 = h, w
        if self.small:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), self.small, 1.0)
            cos, sin = abs(M[0, 0]), abs(M[0, 1]); nw = int(h * sin + w * cos) + 2; nh = int(h * cos + w * sin) + 2
            M[0, 2] += nw / 2 - w / 2; M[1, 2] += nh / 2 - h / 2
            self.M = M; self.Mi = cv2.invertAffineTransform(M); self.size = (nw, nh)
    def fwd(self, x, fill=0):
        x = np.ascontiguousarray(np.rot90(x, self.k90)) if self.k90 else x
        if self.small:
            x = cv2.warpAffine(x.astype(np.float32), self.M, self.size, flags=cv2.INTER_NEAREST, borderValue=float(fill))
        return x
    def inv(self, x):
        if self.small:
            x = cv2.warpAffine(x.astype(np.float32), self.Mi, (self.w0, self.h0), flags=cv2.INTER_NEAREST, borderValue=0)
        return np.ascontiguousarray(np.rot90(x, -self.k90)) if self.k90 else x

# --------------------------------------------------------------------------- main pipeline
class Pipeline:
    def __init__(self, path, out, angle=None, em_max=60, edge=10, seed=0, cont_min_len=40):
        self.path, self.out, self.em_max, self.edge = path, out, em_max, edge
        self.cont_min = cont_min_len
        os.makedirs(out, exist_ok=True)
        raw = np.array(Image.open(path).convert('L')).astype(np.float32)
        self.raw = raw
        full, valid = find_footprint(raw, edge)
        self.full_fp, self.valid0 = full, valid
        bg, sig = mad(raw[valid][::3])
        self.info = dict(image=os.path.basename(path), shape=list(raw.shape), background=float(bg), noise=float(sig),
                         footprint_fraction=float(full.mean()))
        self.sat = float(raw[valid].max()) if raw[valid].max() >= 250 else None
        self.info['saturation_level'] = self.sat
        # fill padding with sky-like values so every statistic sees ordinary noise
        rng = np.random.default_rng(seed)
        fill = raw.copy(); fill[~valid] = rng.choice(raw[valid], (~valid).sum())
        z0 = np.minimum(np.abs(fill - cv2.medianBlur(fill.astype(np.uint8), 31)) / sig, 6.0)
        if angle is None:
            angle, scores = find_dispersion(z0, valid); self.info['dispersion_scores'] = scores
        self.info['dispersion_angle_deg'] = float(angle)
        self.R = Rot(raw.shape, angle)
        self.a = self.R.fwd(fill, fill=bg); self.valid = self.R.fwd(valid.astype(np.float32)) > 0.5
        pad = ~(self.R.fwd(np.ones_like(raw)) > 0.5)
        if pad.any(): self.a[pad] = rng.choice(raw[valid], pad.sum())
        self.H, self.W = self.a.shape; self.bg, self.sig = bg, sig
        log('scores', {k: round(v,3) for k,v in self.info.get('dispersion_scores',{}).items()}); log('frame', raw.shape, 'bg', bg, 'noise', round(sig, 2), 'sat', self.sat, 'angle', angle)

    # ---------------- basic maps
    def maps(self):
        a = self.a; u8 = np.clip(a, 0, 255).astype(np.uint8)
        self.med31 = cv2.medianBlur(u8, 31).astype(np.float32)
        self.med41 = cv2.medianBlur(u8, 41).astype(np.float32)
        self.med101 = cv2.medianBlur(u8, 101).astype(np.float32)
        self.z = np.abs(a - self.med31) / self.sig
        self.zc = np.minimum(self.z, 3.0)
        s = ndi.gaussian_filter(a - self.med41, 1.5)
        _, self.sn = mad(s[self.valid][::7]); self.s15 = s
        self.s12 = ndi.gaussian_filter(a - self.med41, 1.2)
        _, self.sn12 = mad(self.s12[self.valid][::7])

    def rownorm(self, L, zc=None):
        q = cv2.blur(self.zc if zc is None else zc, (L, 1)); m, sd = mad(q[self.valid][::3]); return (q - m) / sd

    # ---------------- continuum residual streaks (along dispersion rows)
    def continuum(self):
        # compact sources (e.g. zeroth-order pairs) must not seed a 'streak' on their own
        zc = self.zc.copy(); zc[cv2.dilate(self._compact.astype(np.uint8), E(5)) > 0] = float(np.median(self.zc[self.valid]))
        q101 = self.rownorm(101, zc); q41 = self.rownorm(41, zc)
        seed = hyst(np.where(self.valid, q101, 0), 3.5, 6.0)
        lab, n = ndi.label(seed); keep = np.zeros(n + 1, bool)
        for i, sl in enumerate(ndi.find_objects(lab), 1):
            hh = sl[0].stop - sl[0].start; ww = sl[1].stop - sl[1].start
            if ww >= 80 and ww >= 8 * hh: keep[i] = True
        streak = keep[lab]
        # follow each streak along its own rows (up to 300 px past the detected ends)
        cand = cv2.dilate(streak.astype(np.uint8), np.ones((3, 601), np.uint8)) > 0
        rows = (np.where(cand & self.valid, q41, 0) > 2.5)
        lab, n = ndi.label(rows); hit = np.unique(lab[streak & rows]); rows = np.isin(lab, hit[hit > 0])
        rows = rows & (np.where(cand, q41, 0) > 2.5)
        strong = hyst(np.where(rows, q41, 0), 2.5, 4.0)
        strong = cv2.morphologyEx(strong.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((1, 9), np.uint8)) > 0
        strong = keep_big(strong, 40)
        strong &= ~(cv2.dilate((self.trail | self.arc).astype(np.uint8), E(5)) > 0)
        # a genuine continuum is exactly along the dispersion; anything tilted is a trail
        lab, n = ndi.label(cv2.dilate(strong.astype(np.uint8), np.ones((5, 15), np.uint8)))
        for i, sl in enumerate(ndi.find_objects(lab), 1):
            cm = (lab[sl] == i) & strong[sl]; yy, xx = np.nonzero(cm)
            if np.ptp(xx) < 120: continue
            p = np.polyfit(xx, yy, 1); ang = np.degrees(np.arctan(p[0]))
            thick = np.bincount(xx - xx.min()).max()
            if abs(ang) > 1.5 and np.std(yy - np.polyval(p, xx)) < 2.5 and thick <= 5:
                self.trail[sl] |= cv2.dilate(cm.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0; strong[sl] &= ~cm
        strong = self.extend_ends(strong, zc)
        self.cont_core = strong
        cont = cv2.dilate(strong.astype(np.uint8), np.ones((3, 1), np.uint8)) > 0
        # a curved arc becomes locally parallel to the dispersion where it is tangent to it; such a
        # piece touches a detected arc/star/trail, is short, and bends -- a real spectrum is straight
        near = cv2.dilate((self.arc | self.star_mask | self.trail).astype(np.uint8), E(9)) > 0
        lab, n = ndi.label(cont, structure=np.ones((3, 3))); self.arc_from_cont = np.zeros(cont.shape, bool); nre = 0
        for i, sl in enumerate(ndi.find_objects(lab), 1):
            cm = lab[sl] == i; L = sl[1].stop - sl[1].start
            if L >= 200 or not (near[sl] & cm).any(): continue
            core = strong[sl] & cm; yy, xx = np.nonzero(core if core.sum() > 10 else cm)
            if np.ptp(xx) < 20: continue
            # centre line (mean row per column): a spectrum's is flat, an arc's bows
            cols = np.unique(xx); cy = np.array([yy[xx == x].mean() for x in cols])
            p1 = np.polyfit(cols, cy, 1); p2 = np.polyfit(cols, cy, 2)
            r1 = np.std(cy - np.polyval(p1, cols)); r2 = np.std(cy - np.polyval(p2, cols)) + 1e-3
            bent = abs(p2[0]) * 1e4 > 2.0 and r1 > 1.0 and r1 > 1.5 * r2
            if bent:
                self.arc_from_cont[sl] |= cm; cont[sl] &= ~cm; nre += 1
        self.arc |= self.arc_from_cont
        self.info['continuum_pieces_reassigned_to_arcs'] = nre
        log('continuum pieces reassigned to arcs/star', nre)
        # minimum length along the dispersion: shorter pieces are blobs (zeroth orders, snowballs,
        # bits of stars/arcs), not spectra -- they are left to the other classes
        lab, n = ndi.label(cont, structure=np.ones((3, 3)))
        L = np.array([0] + [sl[1].stop - sl[1].start for sl in ndi.find_objects(lab)])
        self.cont = (L >= self.cont_min)[lab]
        self.cont_short = cont & ~self.cont
        self.info['continuum_min_length_px'] = self.cont_min
        self.info['continuum_pieces_removed_as_too_short'] = int(((L > 0) & (L < self.cont_min)).sum())
        self.q41 = q41
        log('continuum', round(self.cont[self.valid].mean() * 100, 2), '%')

    def extend_ends(self, strong, zc, step=15, max_ext=400):
        """Walk outward from both ends of every continuum piece along its own rows and keep going while
        the residual stays significant (the faint tapering ends of a spectrum); bridges gaps between
        collinear pieces. Compact sources are already neutralised in zc, so they cannot drive it."""
        med = float(np.median(zc[self.valid])); _, sd = mad(zc[self.valid][::3])
        noise = sd / np.sqrt(3 * step); thr = 3.0 * noise
        blocked = cv2.dilate((self.trail | self.arc | self.star_mask).astype(np.uint8), E(5)) > 0
        out = strong.copy(); added = 0
        lab, n = ndi.label(strong, structure=np.ones((3, 3)))
        for i, sl in enumerate(ndi.find_objects(lab), 1):
            yy, xx = np.nonzero(lab[sl] == i); yy += sl[0].start; xx += sl[1].start
            for side in (-1, 1):
                xe = xx.min() if side < 0 else xx.max()
                sel = (xx <= xe + 15) if side < 0 else (xx >= xe - 15)
                yc = int(round(np.median(yy[sel]))); x = xe; miss = 0; last_good = xe
                while abs(x - xe) < max_ext:
                    xa, xb = (x - step, x) if side < 0 else (x + 1, x + 1 + step)
                    if xa < 0 or xb > self.W: break
                    best, by = -1e9, yc
                    for dy in (-1, 0, 1):                      # follow a small residual tilt
                        y = yc + dy
                        if y - 1 < 0 or y + 2 > self.H: continue
                        if not self.valid[y, xa:xb].all() or blocked[y, xa:xb].any(): best = -1e9; break
                        cmp_ = self._compact[y - 1:y + 2, xa:xb]
                        if cmp_.mean() > 0.5: v = np.nan                 # over a compact blob: judge on what follows
                        else: v = zc[y - 1:y + 2, xa:xb][~cmp_].mean() - med
                        if np.isnan(v): best, by = np.nan, y; break
                        if v > best: best, by = v, y
                    if best == -1e9: break
                    x = xa if side < 0 else xb - 1
                    if np.isnan(best): continue
                    if best > thr: yc = by; last_good = x; miss = 0
                    else:
                        miss += 1
                        if miss >= 2: break
                if last_good != xe:
                    a_, b_ = sorted((xe, last_good))
                    seg = np.zeros(1, bool)
                    new = np.zeros_like(out[yc - 1:yc + 2, a_:b_ + 1]); new[1, :] = True
                    added += int((~out[yc - 1:yc + 2, a_:b_ + 1] & new).sum())
                    out[yc - 1:yc + 2, a_:b_ + 1] |= new
        self.info['continuum_end_extension_px'] = added
        log('continuum ends extended by', added, 'px')
        return out

    # ---------------- compact sources
    def sources(self):
        s, sn = self.s15, self.sn
        seed = hyst(np.where(self.valid, s, 0), 2.5 * sn, 4.5 * sn)
        lab, n = ndi.label(seed); keep = np.zeros(n + 1, bool)
        for i, sl in enumerate(ndi.find_objects(lab), 1):
            hh = sl[0].stop - sl[0].start; ww = sl[1].stop - sl[1].start
            if (lab[sl] == i).sum() >= 5 and not (ww > 80 and ww / hh > 6): keep[i] = True
        src = keep[lab]
        fine = ndi.gaussian_filter(self.a - self.med41, 0.8)
        src = src & (fine > 3.5 * sn)
        src = ndi.binary_fill_holes(cv2.morphologyEx(src.astype(np.uint8), cv2.MORPH_CLOSE, E(3)) > 0)
        self.src = cv2.dilate(src.astype(np.uint8), E(3)) > 0
        self.src &= self.valid
        log('compact sources', ndi.label(self.src)[1])

    # ---------------- pair offset along dispersion
    def measure_pair_offset(self):
        s = self.s12; sn = self.sn12
        pk = (s == ndi.maximum_filter(s, 7)) & (s > 4 * sn) & self.src
        ys, xs = np.nonzero(pk); self.pk = (xs, ys, s[ys, xs])
        T = cKDTree(np.c_[xs, ys]); d = []
        for i, j in T.query_pairs(42):
            if abs(int(ys[i]) - int(ys[j])) <= 2: d.append(abs(int(xs[i]) - int(xs[j])))
        d = np.array(d); h = np.bincount(d, minlength=43)[:43].astype(float)
        base = np.median(h[25:43]) if h[25:43].sum() else 0
        hs = ndi.uniform_filter1d(h, 3); hs[:4] = 0
        mode = int(np.argmax(hs)); excess = hs[mode] - base
        ok = excess > 3 * np.sqrt(base + 1) and 4 <= mode <= 35
        if ok:
            lo = mode; hi = mode
            while lo > 4 and hs[lo - 1] - base > 0.35 * excess: lo -= 1
            while hi < 40 and hs[hi + 1] - base > 0.35 * excess: hi += 1
            self.pair_rng = (max(4, lo - 1), hi + 2)
        else:
            self.pair_rng = (6, 16)
        self.info['pair_separation_px'] = dict(mode=mode if ok else None, accepted_range=list(self.pair_rng))
        log('pair offset range', self.pair_rng, 'mode', mode, 'significant', ok)

    # ---------------- stars: saturated cores + spike pattern
    def stars(self):
        a = self.a; sat = self.sat
        self.star_mask = np.zeros(a.shape, bool); self.star_list = []
        if sat is None: log('no saturation level -> star search skipped'); return
        self._srows = cv2.dilate((self.rownorm(101) > 6).astype(np.uint8), np.ones((5, 1), np.uint8)) > 0
        core = (cv2.blur((a >= sat - 3).astype(np.float32), (3, 3)) > 0.5) & self.valid   # saturated noise pixels are isolated
        # saturated cores can be rings (core subtracted) -> close and fill before measuring
        core = ndi.binary_fill_holes(cv2.morphologyEx(core.astype(np.uint8), cv2.MORPH_CLOSE, E(11)) > 0)
        lab, n = ndi.label(core)
        cands = []
        for i, sl in enumerate(ndi.find_objects(lab), 1):
            m = lab[sl] == i; A = m.sum()
            if A < 150: continue
            hh = sl[0].stop - sl[0].start; ww = sl[1].stop - sl[1].start
            if ww > 2.0 * hh or hh > 2.0 * ww: continue                       # saturated streak segments are not stars
            if A / (hh * ww) < 0.3: continue
            streakrow = self._srows[sl]
            m2 = m & ~streakrow if (m & ~streakrow).sum() > 0.3 * A else m      # centre from the part not on a streak row
            yy, xx = np.nonzero(m2); cy = yy.mean() + sl[0].start; cx = xx.mean() + sl[1].start
            r = np.sqrt(m2.sum() / np.pi)
            cands.append([cx, cy, r, A])
        f = np.clip(a - self.med101, -2 * self.sig, 2.5 * self.sig)
        f[self._srows] = np.nan   # skip streak rows
        def ray(c, ang, r0, r1, half=1):
            t = np.deg2rad(ang); rs = np.arange(r0, r1); v = []
            for o in range(-half, half + 1):
                x = np.round(c[0] + rs * np.cos(t) - o * np.sin(t)).astype(int)
                y = np.round(c[1] + rs * np.sin(t) + o * np.cos(t)).astype(int)
                ok = (x >= 0) & (x < self.W) & (y >= 0) & (y < self.H)
                vv = np.full(len(rs), np.nan); vv[ok] = f[y[ok], x[ok]]; v.append(vv)
            with np.errstate(all='ignore'), warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                return np.nanmean(np.array(v), 0)
        angs = np.arange(0, 360, 1.0)
        prof = {}
        for c in cands:
            r0 = int(c[2] * 1.3 + 6); m = []
            for t in angs:
                v = ray(c, t, r0, r0 + 60); m.append(np.nanmean(v) if np.isfinite(v).any() else np.nan)
            m = np.array(m); b, sc = mad(m); z = (m - b) / sc; prof[id(c)] = z
            c.append(z)
            if False: log('cand', int(c[0]), int(c[1]), 'peaks', [(i, round(float(z[i]),1)) for i in range(360) if np.isfinite(z[i]) and z[i] > 2.5 and z[i] == np.nanmax(z[[(i + k) % 360 for k in range(-6, 7)]])])
        if not cands: log('no star candidates'); return
        # spike template from the candidate with the strongest angular peaks
        def peaks(z, thr):
            out = []
            for i in range(360):
                w = z[[(i + k) % 360 for k in range(-6, 7)]]
                if np.isfinite(z[i]) and z[i] > thr and z[i] == np.nanmax(w): out.append(i)
            return out
        # template from the largest saturated core that shows >=3 spikes (the brightest star)
        tmpl = []
        for c in sorted(cands, key=lambda c: -c[3]):
            pk_ = [p for p in peaks(c[4], 3.0) if min(p % 180, 180 - p % 180) > 8]   # along-dispersion peaks are streaks
            if len(pk_) >= 3: tmpl = pk_; break
        self.info['spike_template_deg'] = tmpl
        for c in cands:
            z = c[4]
            score = np.nanmean([np.nanmax(z[[(t + k) % 360 for k in (-2, -1, 0, 1, 2)]]) for t in tmpl]) if tmpl else 0
            is_star = (len(tmpl) >= 3 and (score > 2.5 or (score > 1.5 and c[3] >= 300))) or c[3] > 3000
            log('star candidate', int(c[0]), int(c[1]), 'area', int(c[3]), 'spike score', round(float(score), 2), '->', is_star)
            if not is_star: continue
            cx, cy, r, A = c[:4]; r0 = int(r * 1.3 + 6)
            m = np.zeros(a.shape, np.uint8)
            cv2.circle(m, (int(cx), int(cy)), int(r * 1.4 + 6), 1, -1)
            spikes = []
            for t in (tmpl or []):
                # refine the angle locally, then find where the spike fades
                best = max(np.arange(t - 2, t + 2.01, 0.25), key=lambda q: np.nanmean(ray((cx, cy), q, r0, r0 + 50)))
                v = ray((cx, cy), best, r0, int(r0 + 40 * r + 400), half=1)
                run = ndi.uniform_filter1d(np.nan_to_num(v), 25)
                noise = self.sig / np.sqrt(3 * 25)
                good = run > 4.0 * noise; end = 0; gap = 0
                for k, g in enumerate(good):
                    if g: end = k; gap = 0
                    else:
                        gap += 1
                        if gap > 25: break
                L = r0 + end + 10
                if end < 5: L = r0 + 15
                tt = np.deg2rad(best)
                w = int(np.clip(r / 3, 3, 7))
                cv2.line(m, (int(cx), int(cy)), (int(cx + L * np.cos(tt)), int(cy + L * np.sin(tt))), 1, w)
                spikes.append((round(float(best), 2), int(L)))
            Rz = int(max(L for _, L in spikes) if spikes else r * 3)
            yy, xx = np.ogrid[:self.H, :self.W]; disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= (1.5 * r + 12) ** 2
            # halo: bright/dark pixels directly around the core
            halo = disk & ((np.abs(self.s12) > 4 * self.sn12))
            m = (m > 0) | halo
            m = ndi.binary_fill_holes(m & ((xx - cx) ** 2 + (yy - cy) ** 2 <= (2 * r + 15) ** 2)) | m
            self.star_mask |= m
            self.star_list.append(dict(x=float(cx), y=float(cy), core_radius=float(r), spikes=spikes, reach=Rz))
        self.star_mask &= self.valid
        log('stars', len(self.star_list), 'template', tmpl)

    # ---------------- ridges: trails, ghost arcs, stray spikes
    def ridges(self):
        pos = np.clip(self.a - self.med101, -self.sig, 3 * self.sig)
        lab_, n_ = ndi.label(self.src); compact = np.zeros(n_ + 1, bool)
        for i, sl in enumerate(ndi.find_objects(lab_), 1): compact[i] = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) < 25
        self._compact = compact[lab_]
        pos[self._compact | self.star_mask | self._srows] = 0            # streak rows are continuum, not ridges
        # oriented matched filter: integrates signal along short straight segments, so it picks up
        # faint straight trails and gently curved ghost arcs alike
        w = (~(cv2.dilate((self._compact | self.star_mask).astype(np.uint8), E(5)) > 0) & ~self._srows & self.valid).astype(np.float32)
        f = pos * w
        R = np.full(f.shape, -99, np.float32)
        for ang in np.arange(0, 180, 5):
            k = line_kernel(ang, 41, 3); k = (k > 0).astype(np.float32)
            num = cv2.filter2D(f, -1, k); den = cv2.filter2D(w, -1, k)
            v = np.where(den > 0.4 * k.sum(), num / np.maximum(den, 1), 0)
            b, sd = mad(v[self.valid][::6]); R = np.maximum(R, (v - b) / sd)
        R[~self.valid] = 0
        m = hyst(R, 4.0, 7.0)
        lab, n = ndi.label(m)
        self.trail = np.zeros(m.shape, bool); self.arc = np.zeros(m.shape, bool)
        nt = na = 0
        for i, sl in enumerate(ndi.find_objects(lab), 1):
            comp = lab[sl] == i
            if comp.sum() < 40: continue
            sk = skeletonize(comp); L = sk.sum()
            if L < 70: continue
            yy, xx = np.nonzero(sk); cov = np.cov(np.vstack([xx, yy]))
            ev, evec = np.linalg.eigh(cov); lin = np.sqrt(ev[1] / max(ev[0], 1e-3))
            ang = np.degrees(np.arctan2(evec[1, 1], evec[0, 1])) % 180
            along = min(ang, 180 - ang) < 2.5
            if along and lin > 15: continue                      # a continuum streak, handled elsewhere
            # residual of a straight-line fit decides trail vs arc
            p = np.polyfit(xx, yy, 1) if min(ang, 180 - ang) < 45 else np.polyfit(yy, xx, 1)
            res = np.std(yy - np.polyval(p, xx)) if min(ang, 180 - ang) < 45 else np.std(xx - np.polyval(p, yy))
            span = np.hypot(np.ptp(xx), np.ptp(yy))
            if res < 2.5 and span > 70: self.trail[sl] |= comp; nt += 1
            elif span > 90 and res >= 2.5 and L > 110: self.arc[sl] |= comp; na += 1
        # pixel-level footprint of each feature: bright pixels hugging the ridge
        near = cv2.dilate((self.trail | self.arc).astype(np.uint8), E(9)) > 0
        lit = ndi.gaussian_filter(self.a - self.med101, 1.0) > 4.0 * self.sig / 2.8
        for name in ('trail', 'arc'):
            base = getattr(self, name)
            grown = (cv2.dilate(base.astype(np.uint8), E(7)) > 0) & lit & ~self._srows | (cv2.dilate(base.astype(np.uint8), E(3)) > 0)
            grown = cv2.morphologyEx(grown.astype(np.uint8), cv2.MORPH_CLOSE, E(5)) > 0
            setattr(self, name, grown & self.valid)
        # straight bright trails via Hough on bright pixels: robust even where a trail runs alongside a streak
        bright = (self.s12 > 3.0 * self.sn12) & self.valid & ~self.star_mask
        L = cv2.HoughLinesP(bright.astype(np.uint8) * 255, 1, np.pi / 720, 50, minLineLength=120, maxLineGap=30)
        hl = np.zeros(bright.shape, np.uint8); nh = 0
        # OpenCV returns (N,1,4) in most versions but (N,4) in some: normalise the shape
        segs = np.asarray(L).reshape(-1, 4) if L is not None else np.zeros((0, 4), int)
        for x1, y1, x2, y2 in segs:
            ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180; d = min(ang, 180 - ang)
            if d < 1.5 or abs(ang - 90) < 1.5: continue                   # along dispersion / columns: other steps
            n_ = int(np.hypot(x2 - x1, y2 - y1)); xs_ = np.linspace(x1, x2, n_).astype(int); ys_ = np.linspace(y1, y2, n_).astype(int)
            near = cv2.dilate(bright.astype(np.uint8), np.ones((5, 5), np.uint8))[ys_, xs_]
            if near.mean() < 0.4: continue
            cv2.line(hl, (int(x1), int(y1)), (int(x2), int(y2)), 1, 3); nh += 1
        if nh:
            hl = hl > 0
            grown = (cv2.dilate(hl.astype(np.uint8), E(7)) > 0) & lit | hl
            self.trail |= cv2.morphologyEx(grown.astype(np.uint8), cv2.MORPH_CLOSE, E(5)) > 0
            self.trail &= self.valid
        self.ridge_R = R
        log('trails', nt, 'arcs', na, 'hough segments', nh)

    # ---------------- detector column pattern / bleeds (perpendicular to dispersion)
    def columns(self):
        q = cv2.blur(self.zc, (1, 101)); b, sd = mad(q[self.valid][::3]); qn = (q - b) / sd
        m = hyst(np.where(self.valid & ~self.cont, qn, 0), 4.0, 8.0)
        lab, n = ndi.label(m); keep = np.zeros(n + 1, bool)
        for i, sl in enumerate(ndi.find_objects(lab), 1):
            hh = sl[0].stop - sl[0].start; ww = sl[1].stop - sl[1].start
            if hh >= 120 and hh >= 6 * ww: keep[i] = True
        col = keep[lab]
        q21 = cv2.blur(self.zc, (1, 21)); b2, sd2 = mad(q21[self.valid][::3])
        col &= (q21 - b2) / sd2 > 2.0
        col = cv2.morphologyEx(col.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 1), np.uint8)) > 0
        # bright bleeds: strongly positive columns
        bl = ndi.gaussian_filter(self.a - self.med101, (6, 1.5)) > 3.5 * self.sig / np.sqrt(20)
        bl = bl & self.valid
        lab, n = ndi.label(bl); kb = np.zeros(n + 1, bool)
        for i, sl in enumerate(ndi.find_objects(lab), 1):
            hh = sl[0].stop - sl[0].start; ww = sl[1].stop - sl[1].start
            if hh >= 60 and hh >= 3 * ww and ww >= 4: kb[i] = True
        self.column = (keep_big(col, 150) | kb[lab]) & self.valid & ~self.cont
        log('column pattern', round(self.column[self.valid].mean() * 100, 3), '%')

    # ---------------- faint extended structure & dipole lobes
    def extended(self):
        base = self.cont | self.src | self.star_mask | self.trail | self.arc | self.column
        f = np.clip(self.a - cv2.medianBlur(np.clip(self.a, 0, 255).astype(np.uint8), 51).astype(np.float32), -2 * self.sig, 2.5 * self.sig)
        w = (~(cv2.dilate(base.astype(np.uint8), E(3)) > 0) & self.valid).astype(np.float32); f = f * w
        num = ndi.gaussian_filter(f, 3.0); den = ndi.gaussian_filter(w, 3.0)
        sm = np.where(den > 0.3, num / np.maximum(den, 1e-3), 0); b, sd = mad(sm[w > 0][::5]); s3 = (sm - b) / sd
        pos = hyst(np.where(self.valid, s3, 0), 3.0, 6.0); neg = hyst(np.where(self.valid, -s3, 0), 3.0, 6.0)
        near_src = cv2.dilate(self.src.astype(np.uint8), E(13)) > 0
        near_art = cv2.dilate((self.star_mask | self.trail | self.arc | self.column).astype(np.uint8), E(21)) > 0
        self.dipole = keep_big(neg, 20)
        lab, n = ndi.label(self.dipole); ok = np.zeros(n + 1, bool)
        for i, sl in enumerate(ndi.find_objects(lab), 1):
            c = lab[sl] == i; ok[i] = (near_src[sl] & c).any() or (near_art[sl] & c).any()
        self.dipole = ok[lab]
        lab, n = ndi.label(keep_big(pos, 20)); self.ext_pos = np.zeros(pos.shape, bool); self.faint_em = np.zeros(pos.shape, bool)
        for i, sl in enumerate(ndi.find_objects(lab), 1):
            c = lab[sl] == i; A = c.sum(); hh = sl[0].stop - sl[0].start; ww = sl[1].stop - sl[1].start
            if (near_art[sl] & c).any(): self.ext_pos[sl] |= c
            elif A <= self.em_max and max(hh, ww) <= 12 and not (near_src[sl] & c).any(): self.faint_em[sl] |= c
            elif A >= 150 and (near_src[sl] & c).any(): self.ext_pos[sl] |= c
        fine = np.abs(ndi.gaussian_filter(f, 1.8)) > 1.8 * self.sig / 3.4
        self.extended_mask = ((self.dipole | self.ext_pos) & (fine | cv2.erode((self.dipole | self.ext_pos).astype(np.uint8), E(5)).astype(bool)))
        self.extended_mask = ndi.binary_fill_holes(cv2.morphologyEx(self.extended_mask.astype(np.uint8), cv2.MORPH_CLOSE, E(5)) > 0)
        self.faint_em = cv2.dilate(self.faint_em.astype(np.uint8), E(3)) > 0
        log('extended', round(self.extended_mask[self.valid].mean() * 100, 3), '%', 'faint emission', ndi.label(self.faint_em)[1])

    # ---------------- classify compact sources
    def classify(self):
        a, s, sn = self.a, self.s12, self.sn12
        lab, n = ndi.label(self.src); objs = ndi.find_objects(lab)
        xs, ys, pv = self.pk; comp = lab[ys, xs]
        lo, hi = self.pair_rng
        T = cKDTree(np.c_[xs, ys]); paired = np.zeros(len(xs), bool); partner = -np.ones(len(xs), int)
        for i, j in T.query_pairs(hi + 1):
            dx = abs(int(xs[j]) - int(xs[i])); dy = abs(int(ys[j]) - int(ys[i]))
            if dy > 2 or not (lo <= dx <= hi): continue
            if not (0.125 <= pv[i] / pv[j] <= 8): continue
            xm = (xs[i] + xs[j]) // 2; ym = (ys[i] + ys[j]) // 2
            if s[ym, xm] > 0.75 * min(pv[i], pv[j]): continue
            paired[i] = paired[j] = True; partner[i] = j; partner[j] = i
        area = np.r_[0, ndi.sum(self.src, lab, range(1, n + 1))]
        npk = np.bincount(comp, minlength=n + 1)
        HG = cv2.dilate(self.cont.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        nmap = cv2.blur(np.abs(a - cv2.medianBlur(np.clip(a, 0, 255).astype(np.uint8), 5)), (21, 9)); nmap /= np.median(nmap[self.valid])
        bgv = self.bg; satl = self.sat or 255
        maxw = 2 * hi + 12
        def is_ring(c):
            sl = objs[c - 1]; cm = lab[sl] == c; g = ndi.gaussian_filter(a[sl], 0.8); pk_ = g[cm].max()
            if pk_ < bgv + 0.8 * self.sig: return False
            brt = (g > bgv + 0.45 * (pk_ - bgv)) & cm
            if brt.sum() < 12: return False
            yy_, xx_ = np.nonzero(brt); cy, cx = yy_.mean(), xx_.mean()
            if g[int(round(cy)), int(round(cx))] > bgv + 0.45 * (pk_ - bgv): return False
            ang = (np.degrees(np.arctan2(yy_ - cy, xx_ - cx)) + 360) % 360
            return len(np.unique((ang // 45).astype(int))) >= 7
        def stripe_ok(c):
            sl = objs[c - 1]; cm = lab[sl] == c
            y0 = max(0, sl[0].start - 8); y1 = min(self.H, sl[0].stop + 8); x0 = max(0, sl[1].start - 8); x1 = min(self.W, sl[1].stop + 8)
            if np.median(nmap[y0:y1, x0:x1]) > 1.25: return False
            if not self.valid[y0:y1, x0:x1].all(): return False
            if is_ring(c): return False
            yc = int(np.round(np.nonzero(cm)[0].mean())) + sl[0].start
            for xa, xb in ((sl[1].start - 18, sl[1].start - 3), (sl[1].stop + 3, sl[1].stop + 18)):
                xa = max(0, xa); xb = min(self.W, xb)
                if xb - xa >= 8 and HG[max(0, yc - 1):yc + 2, xa:xb].mean() >= 0.5: return False
            if (HG[sl] & cm).mean() > 0.25:
                if (sl[0].stop - sl[0].start) < 8 or s[sl][cm].max() < 5 * sn: return False
            return True
        def round_blob(c):
            sl = objs[c - 1]; hh = sl[0].stop - sl[0].start; ww = sl[1].stop - sl[1].start
            if hh > 18 or ww > 20 or ww > 2.2 * hh + 2 or hh > 2.2 * ww + 2: return False
            return stripe_ok(c)
        def valid_pair(c):
            sl = objs[c - 1]; cm = lab[sl] == c; hh = sl[0].stop - sl[0].start; ww = sl[1].stop - sl[1].start
            if hh > 24 or ww > maxw or not stripe_ok(c): return False
            ss = ndi.gaussian_filter(a[sl], 0.8)
            for bm0 in [(s[sl] > q * sn) for q in (3, 4.5, 6, 7.5)] + [ss >= satl - q for q in (15, 5, 1)]:
                bl, bn = ndi.label(bm0 & cm)
                if bn < 2: continue
                info = []
                for k, ssl in enumerate(ndi.find_objects(bl), 1):
                    b = bl[ssl] == k
                    if b.sum() < 4: continue
                    by, bx = np.nonzero(b); info.append((by.mean() + ssl[0].start, bx.mean() + ssl[1].start, np.ptp(by) + 1, np.ptp(bx) + 1))
                for i in range(len(info)):
                    for j in range(i + 1, len(info)):
                        A, B = info[i], info[j]
                        if abs(A[0] - B[0]) > 3 or not (lo - 1 <= abs(A[1] - B[1]) <= hi + 6): continue
                        if max(A[3], B[3]) > 20 or max(A[2], B[2]) > 18 or min(A[2], B[2]) < 3: continue
                        if A[3] > 2.2 * A[2] + 2 or B[3] > 2.2 * B[2] + 2: continue
                        return True
            return False
        comp_paired = np.zeros(n + 1, bool); comp_paired[comp[paired]] = True
        # merged bright pairs (figure-eight): two distance-transform maxima on one row
        for c in range(1, n + 1):
            if comp_paired[c] or area[c] <= self.em_max: continue
            sl = objs[c - 1]; cm = np.pad(lab[sl] == c, 3)
            for m_ in (cm, np.pad(a[sl] >= satl - 20, 3) & cm):
                if m_.sum() < 20: continue
                dt = ndi.distance_transform_edt(m_); pk_ = (dt == ndi.maximum_filter(dt, 5)) & (dt >= 2.0)
                py, px = np.nonzero(pk_)
                for i in range(len(px)):
                    for j in range(i + 1, len(px)):
                        if abs(py[i] - py[j]) <= 3 and lo - 1 <= abs(px[i] - px[j]) <= hi + 6:
                            if dt[(py[i] + py[j]) // 2, (px[i] + px[j]) // 2] < 0.8 * min(dt[py[i], px[i]], dt[py[j], px[j]]): comp_paired[c] = True
        # every component that LOOKS like a pair (two peaks on one row at the measured separation),
        # before the stricter shape/stripe checks: used downstream so no zeroth-order candidate is lost
        self.zo_cand = comp_paired[lab]
        art_lines = self.trail | self.arc | self.column
        art_zone = cv2.dilate(art_lines.astype(np.uint8), E(3)) > 0
        cls = np.zeros(n + 1, np.uint8); sub = np.zeros(n + 1, np.uint8)
        # the zone just beyond each end of every continuum (12 px along, +-4 rows)
        endzone = np.zeros(self.cont.shape, bool)
        for sl_ in ndi.find_objects(ndi.label(self.cont, structure=np.ones((3, 3)))[0]):
            yy0, xx0 = np.nonzero(self.cont[sl_]); yy0 += sl_[0].start; xx0 += sl_[1].start
            for xe, sd_ in ((xx0.min(), -1), (xx0.max(), 1)):
                yc = int(np.median(yy0[np.abs(xx0 - xe) <= 10]))
                xa, xb = (xe - 14, xe + 21) if sd_ < 0 else (xe - 20, xe + 15)
                endzone[max(0, yc - 4):yc + 5, max(0, xa):max(0, xb)] = True
        clab, _ = ndi.label(self.cont, structure=np.ones((3, 3)))
        def rowat(c):                                              # row of the continuum the blob touches
            sl = objs[c - 1]; y0 = max(0, sl[0].start - 2); x0 = max(0, sl[1].start - 5)
            ys_, xs_ = np.nonzero(clab[y0:sl[0].stop + 2, x0:sl[1].stop + 5] > 0)
            return np.median(ys_) + y0 if len(ys_) else -99
        for c in range(1, n + 1):
            sl = objs[c - 1]; cm = lab[sl] == c; hh = sl[0].stop - sl[0].start; ww = sl[1].stop - sl[1].start
            if (self.star_mask[sl] & cm).mean() > 0.3: cls[c] = 4; sub[c] = 1; continue
            if (art_zone[sl] & cm).mean() > 0.15: cls[c] = 4; sub[c] = 6; continue
            if comp_paired[c]:
                mates = [comp[partner[i]] for i in np.nonzero((comp == c) & paired)[0]]
                ok = any(valid_pair(c) if mc == c else (round_blob(c) and round_blob(mc)) for mc in mates) or valid_pair(c)
                if ok: cls[c] = 1; continue
            yy_, xx_ = np.nonzero(cm); sx = xx_.std() + 0.5; sy = yy_.std() + 0.5
            oncont = (cv2.dilate(self.cont.astype(np.uint8), np.ones((5, 5), np.uint8))[sl] > 0)[cm].mean()
            aligned = abs(np.median(yy_ + sl[0].start) - rowat(c)) <= 4
            if sx >= 1.6 * sy and oncont > 0.2 and aligned: cls[c] = 3; continue      # elongated piece lying along a continuum
            if (endzone[sl] & cm).any() and aligned and hh <= 14:
                cls[c] = 3; continue                                   # bright cap at a continuum's end: part of it
            if sx >= 1.6 * sy and 10 <= ww < self.cont_min: cls[c] = 4; sub[c] = 6; continue   # too short for a spectrum: residual/artifact
            if (sx >= 1.6 * sy and ww >= self.cont_min) or (oncont > 0.35 and area[c] > self.em_max): cls[c] = 3; continue   # a blob sitting on a long continuum is part of it
            if is_ring(c): cls[c] = 5; continue
            if area[c] <= self.em_max and npk[c] <= 1: cls[c] = 2; continue
            cls[c] = 5
        self.cls_map = cls[lab]; self.sub_map = sub[lab]; self.src_lab = lab
        log('classes: ZO', (cls == 1).sum(), 'EM', (cls == 2).sum(), 'CONT', (cls == 3).sum(), 'ART', (cls == 4).sum(), 'SNOW', (cls == 5).sum())

    # ---------------- assemble
    def assemble(self):
        zo = self.cls_map == 1; em = (self.cls_map == 2) | (self.faint_em & ~self.src)
        snow = self.cls_map == 5
        art_type = np.zeros(self.a.shape, np.uint8)
        art_type[self.extended_mask] = 5; art_type[self.column] = 4; art_type[self.arc] = 3
        art_type[self.trail] = 2; art_type[self.star_mask] = 1
        art_type[(self.cls_map == 4) & (art_type == 0)] = 6
        nearart = cv2.dilate((art_type > 0).astype(np.uint8), E(7)) > 0
        lab, n = ndi.label(self.cont_short & ~self.src)
        if n:
            touch = np.r_[False, ndi.maximum(nearart, lab, range(1, n + 1)).astype(bool)]
            art_type[touch[lab] & (art_type == 0)] = 6
        art = art_type > 0
        cont = (self.cont | (self.cls_map == 3)) & ~art
        # priorities: star > ZO > emission > snowball > trail/arc/column > continuum > extended
        lm = np.zeros(self.a.shape, np.uint8)
        lm[(art_type == 5) | (art_type == 6)] = 4
        lm[cont] = 3
        lm[(art_type >= 2) & (art_type <= 4)] = 4
        lm[snow] = 5; lm[em] = 2; lm[zo] = 1
        lm[art_type == 1] = 4
        lm[~self.valid] = 0
        # final length check: other classes can cut a continuum into fragments; drop fragments that are too short
        joined = cv2.dilate((lm == 3).astype(np.uint8), np.ones((3, 25), np.uint8)) > 0     # gaps < 24 px along the row
        jl, _ = ndi.label(joined, structure=np.ones((3, 3)))
        lab = np.where(lm == 3, jl, 0); n = int(lab.max())
        if n:
            L = np.array([0] + [(sl[1].stop - sl[1].start) if sl is not None else 0 for sl in ndi.find_objects(lab)])
            short = (L > 0) & (L < self.cont_min); sm = short[lab]
            nearart = cv2.dilate((lm == 4).astype(np.uint8), E(7)) > 0
            touch = np.r_[False, ndi.maximum(nearart, lab, range(1, n + 1)).astype(bool)]
            lm[sm] = 0; lm[sm & touch[lab]] = 4; art_type[sm & touch[lab]] = 6
        art_type[lm != 4] = 0
        self.lm = self.R.inv(lm).astype(np.uint8); self.art_type = self.R.inv(art_type).astype(np.uint8)
        # rotation back can leave single-pixel holes: close them per class
        if self.R.small:
            for k in range(1, 6):
                m = cv2.morphologyEx((self.lm == k).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8)) > 0
                self.lm[m & (self.lm == 0)] = k
        self.lm[~self.valid0] = 0

    def write(self):
        o = self.out; lm = self.lm; V = self.valid0
        names = {1: 'zeroth_order_mask', 2: 'emission_candidate_mask', 3: 'continuum_residual_mask', 4: 'artifact_mask', 5: 'snowball_mask'}
        stats = {}
        for k, nm in names.items():
            m = lm == k; np.save(f'{o}/{nm}.npy', m); Image.fromarray((m * 255).astype(np.uint8)).save(f'{o}/{nm}.png')
            stats[nm] = dict(percent_of_footprint=round(float(m.sum() / V.sum() * 100), 3), objects=int(ndi.label(m)[1]))
        comb = lm > 0; np.save(f'{o}/combined_mask.npy', comb); Image.fromarray((comb * 255).astype(np.uint8)).save(f'{o}/combined_mask.png')
        np.save(f'{o}/label_map.npy', lm); np.save(f'{o}/artifact_type_map.npy', self.art_type)
        np.save(f'{o}/footprint.npy', V)
        # compact-source footprint (every blob, whatever its class) for downstream screening
        np.save(f'{o}/compact_sources.npy', (self.R.inv(self.src.astype(np.uint8)) > 0) & V)
        np.save(f'{o}/zo_candidates.npy', (self.R.inv(self.zo_cand.astype(np.uint8)) > 0) & V)
        self.info['classes'] = stats
        self.info['stars'] = [dict(s, x=None, y=None) for s in []]
        # star positions back in image coordinates
        st = []
        for s_ in self.star_list:
            m = np.zeros(self.a.shape, np.uint8); m[int(s_['y']), int(s_['x'])] = 1
            yy, xx = np.nonzero(self.R.inv(cv2.dilate(m, np.ones((3, 3), np.uint8))))
            st.append(dict(s_, x=float(xx.mean()) if len(xx) else None, y=float(yy.mean()) if len(yy) else None))
        self.info['stars'] = st
        # emission catalogue (image coordinates)
        el, en = ndi.label(lm == 2); src = self.raw
        with open(f'{o}/emission_candidates.csv', 'w', newline='') as fh:
            wr = csv.writer(fh); wr.writerow(['id', 'x', 'y', 'area_px', 'peak_snr', 'on_continuum'])
            contimg = cv2.dilate((lm == 3).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            for i, sl in enumerate(ndi.find_objects(el), 1):
                cm = el[sl] == i; yy, xx = np.nonzero(cm)
                pk = (ndi.gaussian_filter(src[sl], 1.2)[cm].max() - self.bg) / (self.sig / 2.6)
                wr.writerow([i, round(xx.mean() + sl[1].start, 1), round(yy.mean() + sl[0].start, 1), int(cm.sum()), round(float(pk), 1), bool(contimg[sl][cm].any())])
        # overlay + legend
        cols = {1: (40, 200, 255), 2: (60, 255, 60), 3: (255, 160, 0), 4: (255, 40, 40), 5: (200, 80, 255)}
        rgb = np.stack([src] * 3, -1).astype(np.float32)
        for k, c in cols.items(): m = lm == k; rgb[m] = 0.4 * rgb[m] + 0.6 * np.array(c)
        rgb[~self.full_fp] *= 0.35
        ov = Image.fromarray(rgb.astype(np.uint8)); ov.save(f'{o}/classified_overlay.png')
        self.legend(ov, stats, cols)
        with open(f'{o}/run_info.json', 'w') as fh: json.dump(self.info, fh, indent=1, default=float)

    def legend(self, ov, stats, cols):
        W, H = ov.size; sc = W / 2040
        def font(sz):
            for p in ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf']:
                try: return ImageFont.truetype(p, int(sz * sc))
                except Exception: pass
            return ImageFont.load_default()
        F, Fb, Fs = font(34), font(40), font(26)
        items = [(1, 'zeroth_order_mask', 'Zeroth-order sources (paired blobs)'), (2, 'emission_candidate_mask', 'Emission-line candidates (small single blobs)'),
                 (5, 'snowball_mask', 'Snowballs (large single / ring blobs)'), (3, 'continuum_residual_mask', 'Poorly subtracted continuum residuals'),
                 (4, 'artifact_mask', 'Other artifacts (stars, spikes, ghosts, trails, columns)')]
        lh = int(58 * sc); pad = int(30 * sc); ph = pad * 2 + int(60 * sc) + lh * len(items) + int(80 * sc)
        cv = Image.new('RGB', (W, H + ph), (250, 250, 250)); cv.paste(ov, (0, 0)); d = ImageDraw.Draw(cv)
        y = H + pad; d.text((pad, y), f"Mask classes  —  {self.info['image']}", fill=(0, 0, 0), font=Fb); y += int(60 * sc)
        for i, (k, nm, lab) in enumerate(items):
            c = cols[k]; st = stats[nm]; yy = y + i * lh
            d.rectangle([pad, yy + 6, pad + int(44 * sc), yy + int(44 * sc)], fill=tuple(int(0.4 * 128 + 0.6 * v) for v in c), outline=(0, 0, 0), width=2)
            t = f"{lab}  —  {st['percent_of_footprint']:.2f}% of footprint" + (f",  {st['objects']} objects" if k in (1, 2, 5) else '')
            d.text((pad + int(60 * sc), yy + 4), t, fill=(0, 0, 0), font=F)
        inf = self.info
        d.text((pad, H + ph - int(75 * sc)), f"dispersion {inf['dispersion_angle_deg']:.2f} deg   pair separation {inf['pair_separation_px']['accepted_range']} px   "
               f"stars {len(inf['stars'])}   noise {inf['noise']:.1f}   background {inf['background']:.0f}", fill=(60, 60, 60), font=Fs)
        d.text((pad, H + ph - int(40 * sc)), 'Unmasked pixels in greyscale; outside the detector footprint is darkened.', fill=(90, 90, 90), font=Fs)
        cv.save(f'{self.out}/classified_overlay_legend.png')

    def run(self):
        for step in (self.maps, self.sources, self.measure_pair_offset, self.stars,
                     self.ridges, self.continuum, self.columns, self.extended, self.classify, self.assemble, self.write):
            t = time.time(); step(); log(step.__name__, f'{time.time() - t:.1f}s')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('image'); ap.add_argument('--out', required=True)
    ap.add_argument('--angle', type=float, default=None, help='dispersion angle in degrees (default: measured)')
    ap.add_argument('--em-max', type=int, default=60, help='largest emission-candidate area in px (default 60)')
    ap.add_argument('--edge', type=int, default=10, help='px trimmed from the footprint edge (default 10)')
    ap.add_argument('--cont-min-len', type=int, default=40, help='shortest continuum kept, px along the dispersion (default 40)')
    a = ap.parse_args()
    Pipeline(a.image, a.out, a.angle, a.em_max, a.edge, cont_min_len=a.cont_min_len).run()

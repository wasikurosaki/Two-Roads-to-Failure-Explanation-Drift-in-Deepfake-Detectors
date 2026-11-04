"""Degradation module: blur, down/up-scaling, noise, JPEG, social-media chain, and region-restricted degradation.

All functions take/return uint8 RGB numpy arrays (H, W, 3). Randomness is seeded from the image id and the
condition name, so every model sees exactly the same degraded pixels for a given image and condition.
"""
import hashlib
import cv2
import numpy as np


def stable_seed(*parts) -> int:
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------- primitive degradations
def gaussian_blur(img, sigma, rng=None):
    return cv2.GaussianBlur(img, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma),
                            borderType=cv2.BORDER_REFLECT_101)


def down_up(img, scale, rng=None):
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def gaussian_noise(img, sigma, rng):
    noisy = img.astype(np.float32) + rng.normal(0.0, float(sigma), img.shape).astype(np.float32)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def jpeg(img, quality, rng=None):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def unsharp(img, amount, sigma=1.0):
    if amount <= 0:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)


def social_media(img, params, rng=None):
    """Upload pipeline: downscale -> sharpen -> JPEG, then the viewer scales it back up."""
    scale, quality, sharpen = params
    h, w = img.shape[:2]
    x = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    x = unsharp(x, sharpen)
    x = jpeg(x, quality)
    return cv2.resize(x, (w, h), interpolation=cv2.INTER_CUBIC)


# ---------------------------------------------------------------- registry
DEGRADATIONS = {
    "blur":   {"fn": gaussian_blur,  "levels": [1.0, 2.0, 3.0],      "fmt": "σ={}"},
    "resize": {"fn": down_up,        "levels": [0.5, 0.25],          "fmt": "×{}"},
    "noise":  {"fn": gaussian_noise, "levels": [5, 10, 20],          "fmt": "σ={}"},
    "jpeg":   {"fn": jpeg,           "levels": [90, 70, 50, 30],     "fmt": "Q={}"},
    "social": {"fn": social_media,   "levels": [(0.75, 85, 0.3), (0.5, 75, 0.6), (0.35, 60, 1.0)],
               "fmt": "s/Q/sh={}"},
}


def all_conditions(include_clean=True):
    """List of dicts: name ('blur@2'), type, level value, severity (1..k), relative severity (0..1], label."""
    out = [dict(name="clean", type="clean", level=None, severity=0, rel_severity=0.0, label="clean")] if include_clean else []
    for t, spec in DEGRADATIONS.items():
        k = len(spec["levels"])
        for i, lv in enumerate(spec["levels"], start=1):
            out.append(dict(name=f"{t}@{i}", type=t, level=lv, severity=i, rel_severity=i / k,
                            label=f"{t} {spec['fmt'].format(lv)}"))
    return out


def parse_condition(cond):
    if cond == "clean":
        return None, None
    t, i = cond.split("@")
    return t, DEGRADATIONS[t]["levels"][int(i) - 1]


def apply_degradation(img, cond, uid):
    t, level = parse_condition(cond)
    if t is None:
        return img
    rng = np.random.default_rng(stable_seed(uid, cond))
    return DEGRADATIONS[t]["fn"](img, level, rng)


# ---------------------------------------------------------------- region-restricted degradation (Trial 4)
def random_region_mask(face_mask, rng, max_overlap=0.25, tries=60):
    """Translate the face-shaped region to a random location with as little face overlap as possible.
    Same shape and area as the face region, so it controls for 'how many pixels were degraded'."""
    ys, xs = np.nonzero(face_mask > 0.5)
    if len(ys) == 0:
        return face_mask.copy(), 0.0
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    patch = face_mask[y0:y1 + 1, x0:x1 + 1]
    h, w = face_mask.shape
    ph, pw = patch.shape
    area = float(patch.sum()) + 1e-6
    best = None
    for _ in range(tries):
        ty = int(rng.integers(0, h - ph + 1))
        tx = int(rng.integers(0, w - pw + 1))
        m = np.zeros_like(face_mask)
        m[ty:ty + ph, tx:tx + pw] = patch
        overlap = float((m * face_mask).sum()) / area
        if best is None or overlap < best[0]:
            best = (overlap, m)
        if overlap <= max_overlap:
            break
    return best[1], best[0]


def region_mask(face_mask, region, uid):
    if region == "full":
        return np.ones_like(face_mask)
    if region == "face":
        return face_mask
    if region == "background":
        return 1.0 - face_mask
    if region == "random":
        return random_region_mask(face_mask, np.random.default_rng(stable_seed(uid, "random-region")))[0]
    raise ValueError(region)


def apply_region_degradation(img, cond, uid, face_mask=None, region="full", feather=3.0):
    """Degrade the whole image, then keep the degraded pixels only inside `region` (soft-edged blend)."""
    deg = apply_degradation(img, cond, uid)
    if region == "full" or cond == "clean":
        return deg
    m = region_mask(face_mask.astype(np.float32), region, uid)
    if feather > 0:
        m = cv2.GaussianBlur(m, (0, 0), feather)
    m = m[..., None]
    return np.clip(m * deg.astype(np.float32) + (1 - m) * img.astype(np.float32), 0, 255).astype(np.uint8)

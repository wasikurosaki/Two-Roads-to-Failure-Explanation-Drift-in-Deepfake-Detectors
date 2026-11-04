"""Face detection (MTCNN) helpers, face cropping, and the landmark-aligned face ROI mask."""
import math
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:          # lets the numpy parts be used/tested without torch
    Dataset = object


def read_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def letterbox(img, size):
    """Resize keeping aspect ratio and pad to size x size, so images of any shape can be batched for MTCNN."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), np.uint8)
    py, px = (size - nh) // 2, (size - nw) // 2
    canvas[py:py + nh, px:px + nw] = resized
    return canvas, scale, px, py


class DetectDataset(Dataset):
    def __init__(self, paths, size=512):
        self.paths, self.size = list(paths), size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = read_rgb(self.paths[i])
            h, w = img.shape[:2]
            canvas, scale, px, py = letterbox(img, self.size)
            return canvas, (i, scale, px, py, w, h, 1)
        except Exception:
            return np.zeros((self.size, self.size, 3), np.uint8), (i, 1.0, 0, 0, 0, 0, 0)


def detect_collate(batch):
    imgs = np.stack([b[0] for b in batch])
    meta = np.array([b[1] for b in batch], dtype=np.float64)
    return imgs, meta


def mtcnn_detect(mtcnn, imgs):
    """Batched MTCNN on uint8 images (B, H, W, 3). Returns one (boxes, probs, points) tuple per image,
    (None, None, None) when nothing is found. Calls facenet-pytorch's detect_face directly, because
    MTCNN.detect() packs batch results into ragged arrays that recent NumPy versions reject."""
    import torch
    try:
        from facenet_pytorch.models.utils.detect_face import detect_face
        with torch.no_grad():
            batch_boxes, batch_points = detect_face(imgs, mtcnn.min_face_size, mtcnn.pnet, mtcnn.rnet, mtcnn.onet,
                                                    mtcnn.thresholds, mtcnn.factor, mtcnn.device)
        out = []
        for box, pts in zip(batch_boxes, batch_points):
            box = np.asarray(box, dtype=np.float64).reshape(-1, 5) if len(box) else np.zeros((0, 5))
            out.append((None, None, None) if len(box) == 0 else
                       (box[:, :4], box[:, 4], np.asarray(pts, dtype=np.float64).reshape(-1, 5, 2)))
        return out
    except ImportError:
        boxes, probs, points = mtcnn.detect(imgs, landmarks=True)
        return list(zip(boxes, probs, points))


def pick_face(boxes, probs, points, min_prob=0.9):
    """Return (box[4], landmarks[5,2], prob) of the largest confident face, or None."""
    if boxes is None or probs is None or len(boxes) == 0:
        return None
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 5, 2)
    ok = probs >= min_prob
    if not ok.any():
        return None
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area[~ok] = -1
    j = int(np.argmax(area))
    return boxes[j], points[j], float(probs[j])


def square_crop_params(box, W, H, margin):
    """Square crop centred on the face box, `margin` x the box size, shifted (not padded) to stay inside."""
    if box is None or not np.all(np.isfinite(box)):
        side = min(W, H)
        return (W - side) / 2, (H - side) / 2, float(side)
    x0, y0, x1, y1 = box
    side = min(margin * max(x1 - x0, y1 - y0), W, H)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    cx0 = float(np.clip(cx - side / 2, 0, W - side))
    cy0 = float(np.clip(cy - side / 2, 0, H - side))
    return cx0, cy0, float(side)


def make_crop(img, cx0, cy0, cside, target_side=None, crop_size=320, jpeg_q=95):
    """Cut the square crop, optionally downsample to `target_side` (paired real's native resolution),
    resize to crop_size, and pass through one JPEG(q) round so every class carries the same compression."""
    xa, ya = int(round(cx0)), int(round(cy0))
    sa = max(1, int(round(cside)))
    crop = img[ya:ya + sa, xa:xa + sa]
    t = int(round(target_side)) if target_side is not None and np.isfinite(target_side) else crop.shape[0]
    if 0 < t < crop.shape[0]:
        crop = cv2.resize(crop, (t, t), interpolation=cv2.INTER_AREA)
    interp = cv2.INTER_AREA if crop.shape[0] > crop_size else cv2.INTER_CUBIC
    crop = cv2.resize(crop, (crop_size, crop_size), interpolation=interp)
    if jpeg_q:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_q)])
        crop = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return crop


def crop_geometry(box, landmarks, cx0, cy0, cside):
    """Face box and 5 landmarks expressed in normalised crop coordinates (0..1)."""
    g = {"fx0": np.nan, "fy0": np.nan, "fx1": np.nan, "fy1": np.nan}
    g.update({f"l{i}{a}": np.nan for i in range(5) for a in "xy"})
    if box is not None:
        g.update(fx0=(box[0] - cx0) / cside, fy0=(box[1] - cy0) / cside,
                 fx1=(box[2] - cx0) / cside, fy1=(box[3] - cy0) / cside)
    if landmarks is not None:
        for i in range(5):
            g[f"l{i}x"] = (landmarks[i][0] - cx0) / cside
            g[f"l{i}y"] = (landmarks[i][1] - cy0) / cside
    return g


class CropDataset(Dataset):
    """Runs make_crop for every record in DataLoader workers and saves the crop as PNG (skips existing files)."""

    def __init__(self, records, crop_size=320, jpeg_q=95):
        self.records, self.crop_size, self.jpeg_q = list(records), crop_size, jpeg_q

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        out = Path(r["crop_path"])
        if out.exists():
            return i, 1
        try:
            crop = make_crop(read_rgb(r["path"]), r["cx0"], r["cy0"], r["cside"], r.get("target_side"),
                             self.crop_size, self.jpeg_q)
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            return i, 1
        except Exception:
            return i, 0


# ---------------------------------------------------------------- face ROI mask
def face_mask(meta, size, margin=1.7):
    """Filled ellipse over the face, in crop coordinates (0..size).
    Centre from the box and the 5 landmarks, size from the MTCNN box, rotation from the eye line.
    `meta` holds normalised crop-space values: fx0, fy0, fx1, fy1, and l0x..l4y (left eye, right eye, nose, mouth L, mouth R)."""
    m = np.zeros((size, size), np.float32)
    detected = bool(meta.get("detected", False)) and np.isfinite(meta.get("fx0", np.nan))
    if detected:
        x0, y0, x1, y1 = (meta[k] * size for k in ("fx0", "fy0", "fx1", "fy1"))
        lm = np.array([[meta[f"l{i}x"], meta[f"l{i}y"]] for i in range(5)], dtype=np.float64) * size
        bcx, bcy = (x0 + x1) / 2, (y0 + y1) / 2
        if np.all(np.isfinite(lm)):
            lcx, lcy = lm[:, 0].mean(), lm[:, 1].mean()
            cx, cy = 0.5 * (bcx + lcx), 0.5 * (bcy + lcy)
            angle = math.degrees(math.atan2(lm[1, 1] - lm[0, 1], lm[1, 0] - lm[0, 0]))
        else:
            cx, cy, angle = bcx, bcy, 0.0
        ax, ay = 0.5 * (x1 - x0), 0.55 * (y1 - y0)
    else:                                    # no detection: centred default ellipse of the expected size
        cx = cy = size / 2
        ax = size / (2 * margin)
        ay = 1.2 * ax
        angle = 0.0
    cv2.ellipse(m, (int(round(cx)), int(round(cy))), (max(1, int(round(ax))), max(1, int(round(ay)))),
                angle, 0, 360, 1.0, -1)
    return m

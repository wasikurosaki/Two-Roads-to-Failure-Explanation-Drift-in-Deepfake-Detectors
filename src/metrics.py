"""Metrics and statistics: fast AUC, bootstrap CIs, paired bootstrap tests, Holm correction, ESS, face energy."""
import numpy as np
from scipy.stats import norm, rankdata


def fast_auc(y, s):
    """ROC-AUC via the Mann-Whitney U statistic (ties handled with mid-ranks)."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def best_threshold(y, s):
    """Threshold maximising balanced accuracy (Youden's J) – chosen once on the clean validation set."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    cand = np.unique(np.quantile(s, np.linspace(0.005, 0.995, 199)))
    best_t, best_j = 0.5, -1
    for t in cand:
        pred = s >= t
        j = pred[y].mean() + (~pred[~y]).mean() - 1
        if j > best_j:
            best_t, best_j = float(t), j
    return best_t


def classification_metrics(y, s, thr=0.5):
    y = np.asarray(y).astype(bool)
    pred = np.asarray(s) >= thr
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return dict(auc=fast_auc(y, s), acc=(tp + tn) / max(1, len(y)), precision=prec, recall=rec,
                f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                fpr=fp / (fp + tn) if fp + tn else 0.0, n=int(len(y)), n_fake=int(y.sum()))


def _strat_boot_idx(y, rng):
    y = np.asarray(y).astype(bool)
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    return np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def bootstrap_auc_ci(y, s, n_boot=1000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    vals = [fast_auc(y[i], s[i]) for i in (_strat_boot_idx(y, rng) for _ in range(n_boot))]
    return float(np.nanpercentile(vals, 100 * alpha / 2)), float(np.nanpercentile(vals, 100 * (1 - alpha / 2)))


def delong_paired_p(y, s_a, s_b):
    """Two-sided p-value of DeLong's test for AUC(b) = AUC(a), two correlated scores on the same images
    (fast midrank version, Sun & Xu 2014). Analytic, so it is not floored at 1/n_boot like a bootstrap p-value –
    needed when hundreds of tests go through Holm."""
    y = np.asarray(y).astype(bool)
    m, n = int(y.sum()), int((~y).sum())
    if m < 2 or n < 2:
        return np.nan
    order = np.r_[np.flatnonzero(y), np.flatnonzero(~y)]
    X = np.vstack([np.asarray(s_a, np.float64), np.asarray(s_b, np.float64)])[:, order]
    tx, ty, tz = rankdata(X[:, :m], axis=1), rankdata(X[:, m:], axis=1), rankdata(X, axis=1)
    aucs = tz[:, :m].sum(1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 1e-15:
        return 1.0
    return float(2 * norm.sf(abs(aucs[1] - aucs[0]) / np.sqrt(var)))


def paired_bootstrap_delta(y, s_a, s_b, n_boot=1000, seed=0, alpha=0.05):
    """AUC(b) - AUC(a) on the same images. Returns delta, a stratified-bootstrap CI, the paired DeLong p-value (p)
    and the bootstrap p-value (p_boot, resolution 1/n_boot – kept for reference only)."""
    rng = np.random.default_rng(seed)
    y, s_a, s_b = np.asarray(y), np.asarray(s_a), np.asarray(s_b)
    delta = fast_auc(y, s_b) - fast_auc(y, s_a)
    d = np.empty(n_boot)
    for k in range(n_boot):
        i = _strat_boot_idx(y, rng)
        d[k] = fast_auc(y[i], s_b[i]) - fast_auc(y[i], s_a[i])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return dict(delta=float(delta), lo=float(np.percentile(d, 100 * alpha / 2)),
                hi=float(np.percentile(d, 100 * (1 - alpha / 2))), p=delong_paired_p(y, s_a, s_b),
                p_boot=float(min(1.0, max(p, 1.0 / n_boot))))


def holm(pvals):
    """Holm–Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=np.float64)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    idx = np.flatnonzero(ok)[np.argsort(p[ok])]
    m, running = len(idx), 0.0
    for rank, j in enumerate(idx):
        running = max(running, min(1.0, (m - rank) * p[j]))
        out[j] = running
    return out


def rowwise_spearman(A, B):
    """Spearman correlation between matching rows of A and B (N, D). Constant rows -> NaN.
    Used for the Explanation Stability Score (ESS) between clean and degraded Grad-CAM maps."""
    A = rankdata(np.asarray(A, dtype=np.float64).reshape(len(A), -1), axis=1)
    B = rankdata(np.asarray(B, dtype=np.float64).reshape(len(B), -1), axis=1)
    A -= A.mean(1, keepdims=True)
    B -= B.mean(1, keepdims=True)
    den = np.sqrt((A ** 2).sum(1) * (B ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (A * B).sum(1) / den
    r[den == 0] = np.nan
    return r


def face_energy(cams, masks):
    """Share of Grad-CAM energy inside the face ROI, per image. cams, masks: (N, h, w)."""
    cams = np.asarray(cams, dtype=np.float64)
    masks = np.asarray(masks, dtype=np.float64)
    tot = cams.reshape(len(cams), -1).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        e = (cams * masks).reshape(len(cams), -1).sum(1) / tot
    e[tot <= 0] = np.nan
    return e

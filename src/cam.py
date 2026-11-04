"""Grad-CAM for timm classifiers with a single 'fake' logit.

Uses model.forward_features / model.forward_head (available on every timm model), so the target layer is the
last convolutional feature map: ResNet-18 -> layer4, Xception -> the final separable conv block (act4).
"""
import numpy as np
import torch
import torch.nn.functional as F


def gradcam_batch(model, x, out_res=32):
    """Returns (cams[B,out_res,out_res] in [0,1], probs[B]). CAM is for the 'fake' logit."""
    model.eval()
    req = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            feats = model.forward_features(x)
            logits = model.forward_head(feats).float().reshape(-1)
            grads, = torch.autograd.grad(logits.sum(), feats)
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * feats).sum(1, keepdim=True)).float()
    finally:
        for p, r in zip(model.parameters(), req):
            p.requires_grad_(r)
    cam = F.interpolate(cam, size=(out_res, out_res), mode="bilinear", align_corners=False)[:, 0]
    mx = cam.flatten(1).amax(1).clamp_min(1e-8)[:, None, None]
    return (cam / mx).detach().cpu().numpy(), torch.sigmoid(logits).detach().cpu().numpy()


def gradcam_loader(model, loader, device, out_res=32):
    n = len(loader.dataset)
    cams = np.zeros((n, out_res, out_res), np.float16)
    probs = np.zeros(n, np.float32)
    for x, _, idx in loader:
        c, p = gradcam_batch(model, x.to(device), out_res)
        cams[idx.numpy()] = c.astype(np.float16)
        probs[idx.numpy()] = p
    return cams, probs

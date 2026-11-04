"""Model construction (timm), training loop with early stopping, prediction, checkpoint I/O."""
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .data import FaceCropDataset, make_loader
from .metrics import fast_auc, best_threshold

# Each architecture: list of timm names to try (first one that exists in the installed timm wins).
ARCH_SPECS = {
    "resnet18": ["resnet18.tv_in1k", "resnet18"],
    "xception": ["legacy_xception.tf_in1k", "legacy_xception", "xception41.tf_in1k"],
}


def build_model(arch, pretrained=True):
    last_err = None
    for name in ARCH_SPECS[arch]:
        try:
            model = timm.create_model(name, pretrained=pretrained, num_classes=1)
            cfg = timm.data.resolve_model_data_config(model)
            return model, dict(timm_name=name, input_size=int(cfg["input_size"][-1]),
                               mean=tuple(cfg["mean"]), std=tuple(cfg["std"]))
        except Exception as e:      # unknown name / pretrained tag in this timm version
            last_err = e
    raise RuntimeError(f"Could not build {arch}: {last_err}")


def autocast_ctx(device, enabled=True):
    if device.type == "cuda" and enabled:
        return torch.autocast("cuda", dtype=torch.float16)
    return torch.autocast("cpu", enabled=False)


@torch.no_grad()
def predict(model, loader, device, amp=True):
    model.eval()
    out = np.zeros(len(loader.dataset), np.float32)
    for x, _, idx in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with autocast_ctx(device, amp):
            logits = model(x).float().squeeze(1)
        out[idx.numpy()] = torch.sigmoid(logits).cpu().numpy()
    return out


def train_model(tag, arch, train_df, val_df, ckpt_dir, device, epochs=8, lr=1e-4, weight_decay=1e-4,
                batch_size=64, patience=2, num_workers=4, amp=True, aug_degrade_p=0.0, crop_size=320,
                margin=1.7, pretrained=True, seed=0, extra=None):
    ckpt_path = Path(ckpt_dir) / f"{tag}.pt"
    partial_path = Path(ckpt_dir) / f"{tag}.partial.pt"     # best epoch so far, while training is still running
    if ckpt_path.exists():
        print(f"[{tag}] checkpoint exists – skipping training")
        return ckpt_path
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model, spec = build_model(arch, pretrained)
    model = model.to(device).to(memory_format=torch.channels_last)
    common = dict(input_size=spec["input_size"], mean=spec["mean"], std=spec["std"], crop_size=crop_size, margin=margin)
    tr_loader = make_loader(FaceCropDataset(train_df, train=True, aug_degrade_p=aug_degrade_p, **common),
                            batch_size, num_workers, shuffle=True, drop_last=True)
    va_loader = make_loader(FaceCropDataset(val_df, **common), batch_size, num_workers)

    n_pos = float(train_df["label"].sum())
    n_neg = float(len(train_df) - n_pos)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / max(n_pos, 1.0), device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total = epochs * len(tr_loader)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best_auc, best_state, bad, best_val_p, best_ep = [], -1.0, None, 0, None, 0

    def save_ckpt(path):
        thr = best_threshold(val_df["label"].to_numpy(), best_val_p)
        ckpt = dict(state_dict=best_state, arch=arch, spec=spec, threshold=thr, val_auc=best_auc, best_epoch=best_ep,
                    history=history, tag=tag, aug_degrade_p=aug_degrade_p, **(extra or {}))
        tmp = path.with_suffix(".tmp")
        torch.save(ckpt, tmp)
        tmp.replace(path)                                   # atomic: an interrupted save never leaves a broken file
        return thr

    for ep in range(1, epochs + 1):
        model.train()
        t0, run_loss, n = time.time(), 0.0, 0
        for x, y, _ in tqdm(tr_loader, desc=f"[{tag}] epoch {ep}/{epochs}", leave=False):
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast_ctx(device, use_amp):
                loss = crit(model(x).float().squeeze(1), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * len(y)
            n += len(y)
        val_p = predict(model, va_loader, device, use_amp)
        val_auc = fast_auc(val_df["label"].to_numpy(), val_p)
        history.append(dict(epoch=ep, train_loss=run_loss / max(n, 1), val_auc=val_auc, sec=time.time() - t0))
        print(f"[{tag}] epoch {ep}: loss {run_loss / max(n, 1):.4f}  val AUC {val_auc:.4f}  ({time.time() - t0:.0f}s)")
        if best_state is None or val_auc > best_auc + 1e-4:
            best_auc, bad = val_auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val_p, best_ep = val_p, ep
            save_ckpt(partial_path)
            print(f"[{tag}] new best epoch {ep} saved -> {partial_path.name}")
        else:
            bad += 1
            if bad >= patience:
                print(f"[{tag}] early stop")
                break

    thr = save_ckpt(ckpt_path)
    partial_path.unlink(missing_ok=True)
    pd.DataFrame(history).to_csv(Path(ckpt_dir) / f"{tag}_history.csv", index=False)
    print(f"[{tag}] best val AUC {best_auc:.4f} (epoch {best_ep}), threshold {thr:.3f} -> {ckpt_path.name}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ckpt_path


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = timm.create_model(ckpt["spec"]["timm_name"], pretrained=False, num_classes=1)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).to(memory_format=torch.channels_last).eval()
    return model, ckpt

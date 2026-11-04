"""PyTorch dataset over cached face crops, with optional (region-restricted) degradation."""
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from . import degrade, faces

MASK_COLS = ["detected", "fx0", "fy0", "fx1", "fy1"] + [f"l{i}{a}" for i in range(5) for a in "xy"]


class FaceCropDataset(Dataset):
    """
    condition : 'clean' or e.g. 'jpeg@3' (applied at crop resolution, before resizing to the model input)
    region    : 'full' | 'face' | 'background' | 'random'   (Trial 4)
    train     : random horizontal flip
    aug_degrade_p : probability of a random degradation during training (optional remedy experiment)
    """

    def __init__(self, df, input_size, mean, std, condition="clean", region="full", train=False,
                 aug_degrade_p=0.0, crop_size=320, margin=1.7, feather=3.0):
        self.paths = df["crop_path"].tolist()
        self.labels = df["label"].to_numpy(np.float32)
        self.uids = df["uid"].tolist()
        self.meta = df[MASK_COLS].to_dict("records") if region != "full" else None
        self.input_size, self.condition, self.region = int(input_size), condition, region
        self.train, self.aug_p = train, aug_degrade_p
        self.mean = np.asarray(mean, np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, np.float32).reshape(1, 1, 3)
        self.crop_size, self.margin, self.feather = crop_size, margin, feather
        self.aug_conds = [c["name"] for c in degrade.all_conditions(include_clean=False)]

    def __len__(self):
        return len(self.paths)

    def load_rgb(self, i):
        return faces.read_rgb(self.paths[i])

    def __getitem__(self, i):
        img = self.load_rgb(i)
        if self.train and self.aug_p > 0 and random.random() < self.aug_p:
            img = degrade.apply_degradation(img, random.choice(self.aug_conds), f"{self.uids[i]}#{random.random()}")
        elif self.condition != "clean":
            fm = faces.face_mask(self.meta[i], img.shape[0], self.margin) if self.region != "full" else None
            img = degrade.apply_region_degradation(img, self.condition, self.uids[i], fm, self.region, self.feather)
        if self.train and random.random() < 0.5:
            img = img[:, ::-1]
        if img.shape[0] != self.input_size:
            interp = cv2.INTER_AREA if img.shape[0] > self.input_size else cv2.INTER_CUBIC
            img = cv2.resize(np.ascontiguousarray(img), (self.input_size, self.input_size), interpolation=interp)
        x = (img.astype(np.float32) / 255.0 - self.mean) / self.std
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))), torch.tensor(self.labels[i]), i


def make_loader(ds, batch_size, num_workers, shuffle=False, drop_last=False):
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=False,
        prefetch_factor=4 if num_workers > 0 else None)

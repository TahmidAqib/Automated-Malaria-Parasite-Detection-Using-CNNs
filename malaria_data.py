# Dataset and transform helpers for the CSE366 malaria project (written from Submission2_Pretrained_Models.ipynb).
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class PadToSquare:
    # Pad a PIL image to a square canvas with black, keeping it centred (preserves cell shape).
    def __call__(self, img):
        w, h = img.size
        s = max(w, h)
        canvas = Image.new("RGB", (s, s), (0, 0, 0))
        canvas.paste(img, ((s - w) // 2, (s - h) // 2))
        return canvas


class MalariaCellDataset(Dataset):
    # Loads the cell images listed in a split DataFrame (columns: relpath, label).
    # class_to_idx maps label names to integers, e.g. {"Parasitized": 0, "Uninfected": 1}.

    def __init__(self, frame, root, class_to_idx, transform=None):
        mapped = frame["label"].map(class_to_idx)
        if mapped.isna().any():
            raise ValueError(f"Unknown labels in splits.csv: {set(frame['label'][mapped.isna()])}")
        self.paths = [Path(root) / p for p in frame["relpath"]]
        self.labels = mapped.astype(int).to_numpy()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as im:
            img = im.convert("RGB")            # guarantees 3 channels
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class BaseThermalDataset(Dataset):
    def __init__(self, segment_df, img_size=(32, 32), augment=False):
        self.segment_df = segment_df.reset_index(drop=True)
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.segment_df)

    def _load_frame(self, path):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Could not read image: {path}")

        image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.0
        return image

    def _augment_clip(self, clip):
        if self.augment and np.random.rand() < 0.5:
            clip = np.flip(clip, axis=2).copy()
        return clip


class ThermalDataset(BaseThermalDataset):
    def __getitem__(self, index):
        row = self.segment_df.iloc[index]
        frames = [self._load_frame(path) for path in row["paths"]]
        clip = np.stack(frames, axis=0)
        clip = self._augment_clip(clip)

        clip = torch.from_numpy(clip).unsqueeze(1).float()
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return clip, label


class MotionThermalDataset(BaseThermalDataset):
    def __getitem__(self, index):
        row = self.segment_df.iloc[index]
        frames = [self._load_frame(path) for path in row["paths"]]
        clip = np.stack(frames, axis=0)
        clip = self._augment_clip(clip)

        diffs = np.zeros_like(clip)
        diffs[1:] = np.abs(clip[1:] - clip[:-1])

        raw = torch.from_numpy(clip).unsqueeze(1).float()
        motion = torch.from_numpy(diffs).unsqueeze(1).float()
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return raw, motion, label

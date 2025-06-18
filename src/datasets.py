import os
from typing import Optional, Callable

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset

from src.utils import preprocess_features


class SequenceDataset(Dataset):
    """
    Dataset class for loading and preprocessing classification sequence data.

    Args:
        datadir (str): Directory containing the sequence data.
        sequence_length (int): Length of the sequence to be returned.
    """

    def __init__(self, datadir: str, sequence_length: int = 450):
        self.datadir = datadir
        self.sequence_length = sequence_length
        self.files = [f for f in os.listdir(datadir) if f.endswith(".npy")]
        if not self.files:
            raise RuntimeError(f"Found 0 files in {datadir}. ")

        file_path = os.path.join(self.datadir, self.files[0])
        sampled_item = np.load(file_path)
        self.len_features = sampled_item.shape[1]
        self.classes = sorted((set([f.split("_")[0] for f in self.files])))
        self.dict_class_indices = {cls: i for i, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.datadir, self.files[idx])
        features = np.load(file_path)
        label = self.dict_class_indices[self.files[idx].split("_")[0]]
        features = preprocess_features(features, self.sequence_length)
        return torch.from_numpy(features).float(), torch.tensor(label, dtype=torch.long)


class SSLVideoFramesDataset(Dataset):
    """
    Dataset class for loading and preprocessing video frames. Used for SSL pretraining.

    Args:
        datadir (str): Directory containing the video frames.
        transform (Optional[Callable]): Transformations to apply to the video frames.
    """

    def __init__(self, datadir: str, transform: Optional[Callable] = None):
        self.datadir = datadir
        self.transform = transform
        self.files = [
            os.path.join(datadir, f)
            for f in sorted(os.listdir(datadir))
            if f.endswith((".png", ".jpg", ".jpeg"))
        ]
        if not self.files:
            raise RuntimeError(f"Found 0 images in {datadir}. ")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        frame = cv2.imread(img_path)[:, :, ::-1].copy()

        if self.transform:
            aug_frame1 = self.transform(frame)
            aug_frame2 = self.transform(frame)
            return aug_frame1, aug_frame2
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
        frame = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0
        return frame, frame


class OneVideoFramesDataset(Dataset):
    """
    Dataset class for generating video frames. Used for feature extraction.

    Args:
        video_path (str): Path to the video file.
        transform (Optional[Callable]): Transformations to apply to the video frames.
    """

    def __init__(self, video_path: str, transform: Optional[Callable] = None):
        self.transform = transform
        self.success = True
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count <= 0:
            print(f"Could not read frames from {video_path}, skipping.")
            self.success = False

        self.frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frames.append(frame)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        if self.transform:
            frame = self.transform(frame)
            return frame
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
        frame = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0
        return frame

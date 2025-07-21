import os
from pathlib import Path
import argparse
import sys

import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

FILE = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(FILE))
if ROOT not in sys.path:
    sys.path.append(ROOT)
from src.datasets import SSLVideoFramesDataset
from src.models import SSLFeatureExtractor
from src.losses import SimSiamLoss
from src.train import train
from src.utils import extract_frames


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, default="data/raw_videos")
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--save_model_path",
        type=str,
        default="data/weights/ssl_feature_extractor.pth",
    )
    args = parser.parse_args()
    return args


def main(
    datadir: str,
    num_epochs: int,
    batch_size: int,
    num_workers: int,
    lr: float,
    save_model_path: str,
):
    # Extract frames
    extract_frames_dir = os.path.join(os.path.dirname(datadir), "raw_video_frames")
    os.makedirs(extract_frames_dir, exist_ok=True)
    print("\n--------------------------------")
    print("Extracting frames...\n")
    for dir in Path(datadir).iterdir():
        if (not dir.is_dir()) or (dir.name not in ["train", "val"]):
            continue
        video_files = [f for f in dir.iterdir() if f.suffix in [".mp4", ".avi", ".mov"]]
        for video_file in video_files:
            save_frame_dir = Path(extract_frames_dir) / dir.name
            os.makedirs(save_frame_dir, exist_ok=True)

            video_name = video_file.name
            frame_files = save_frame_dir.glob(f"{video_name}_f-*.jpg")
            if next(frame_files, None):
                print(f"Frames for {video_name} already extracted. Skipping.")
                continue

            video_path = video_file.as_posix()
            extract_frames(video_path, save_frame_dir.as_posix())
    print("Done extracting frames!")
    print("--------------------------------\n")

    # load data
    torch.cuda.empty_cache()
    torch.manual_seed(9)
    torch.cuda.manual_seed_all(9)

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = SSLVideoFramesDataset(
        datadir=os.path.join(extract_frames_dir, "train"), transform=transform
    )
    val_dataset = SSLVideoFramesDataset(
        datadir=os.path.join(extract_frames_dir, "val"), transform=transform
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_config = {}
    data_config = {"transform": transform}
    model = SSLFeatureExtractor()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = SimSiamLoss(device=device)

    print("\n--------------------------------")
    print("Training model...\n")
    train(
        mode="ssl",
        model=model,
        model_and_data_config={
            "model_config": model_config,
            "data_config": data_config,
        },
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=num_epochs,
        device=device,
        save_model_path=save_model_path,
    )
    print("\nDone training model!")
    print("--------------------------------\n")


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))

import os
from pathlib import Path
import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

FILE = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(FILE))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.train import train
from src.datasets import OneVideoFramesDataset, SequenceDataset
from src.models import FeatureExtractor, VideoMAEPoseClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_videos_dir", type=str, default="data/raw_videos")
    parser.add_argument(
        "--extracted_features_dir",
        type=str,
        default="data/processed/extracted_features",
    )
    parser.add_argument(
        "--ssl_model_path", type=str, default="data/weights/ssl_feature_extractor.pth"
    )
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--use_pretrained_transformer", type=bool, default=True)
    parser.add_argument(
        "--save_model_path",
        type=str,
        default="data/weights/transformer_pose_classifier.pth",
    )
    args = parser.parse_args()
    return args


def main(
    raw_videos_dir: str,
    extracted_features_dir: str,
    ssl_model_path: str,
    num_epochs: int,
    batch_size: int,
    num_workers: int,
    lr: float,
    use_pretrained_transformer: bool,
    save_model_path: str,
):
    os.makedirs(extracted_features_dir, exist_ok=True)
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)

    torch.manual_seed(9)
    torch.cuda.manual_seed_all(9)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}\n")

    # Extract features from videos
    print("\n--------------------------------")
    print("Extracting features from videos...\n")
    feature_extractor = FeatureExtractor(ssl_model_path=ssl_model_path)
    feature_extractor.to(device)
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    for dir in Path(raw_videos_dir).iterdir():
        if (not dir.is_dir()) or (dir.name not in ["train", "val"]):
            continue
        video_files = [f for f in dir.iterdir() if f.suffix in [".mp4", ".avi", ".mov"]]
        for video_file in video_files:
            video_name = video_file.stem
            save_feature_dir = Path(extracted_features_dir) / dir.name
            os.makedirs(save_feature_dir, exist_ok=True)

            output_feature_path = save_feature_dir / f"{video_name}.npy"
            if output_feature_path.exists():
                print(f"Features for {video_name} already extracted. Skipping.")
                continue

            video_path = video_file.as_posix()
            video_frames_dataset = OneVideoFramesDataset(video_path, transform)
            video_frames_dataloader = DataLoader(
                video_frames_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
            )
            pbar = tqdm(
                video_frames_dataloader,
                desc=f"Extracting features from {video_file.name}",
            )
            video_features = []
            for frames in pbar:
                features = feature_extractor(frames.to(device))
                features = features.cpu().numpy()
                video_features.append(features)
            video_features = np.concatenate(video_features, axis=0)
            np.save(
                output_feature_path,
                video_features,
            )
            del video_frames_dataset, video_frames_dataloader, pbar, video_features
            torch.cuda.empty_cache()
    print("Done extracting features from videos!")
    print("--------------------------------\n")

    # Load data for transformer training
    train_dataset = SequenceDataset(os.path.join(extracted_features_dir, "train"))
    val_dataset = SequenceDataset(os.path.join(extracted_features_dir, "val"))

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    assert (
        train_dataset.classes == val_dataset.classes
    ), "Train and val classes must have the same classes and order"

    model_config = {
        "cnn_embedding_dim": train_dataset[0][0].shape[-1],
        "num_classes": train_dataset.num_classes,
        "pretrained": use_pretrained_transformer,
    }
    data_config = {
        "sequence_length": train_dataset.sequence_length,
        "dict_class_indices": train_dataset.dict_class_indices,
    }
    # Train model
    model = VideoMAEPoseClassifier(**model_config)
    model.to(device)

    # Use different learning rates for pretrained and remaining parameters
    if use_pretrained_transformer:
        pretrained_params = []
        remaining_params = []
        for name, param in model.named_parameters():
            if name.startswith("encoder"):
                pretrained_params.append(param)
            else:
                remaining_params.append(param)

        optimizer = optim.Adam(
            [
                {"params": pretrained_params, "lr": lr * 0.01},
                {"params": remaining_params, "lr": lr},
            ],
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    print("\n--------------------------------")
    print("Training model...\n")
    train(
        mode="sl",
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

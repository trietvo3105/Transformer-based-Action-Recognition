import os
import argparse
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np

FILE = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(FILE))
if ROOT not in sys.path:
    sys.path.append(ROOT)
from src.models import VideoMAEPoseClassifier, FeatureExtractor
from src.datasets import OneVideoFramesDataset
from src.utils import preprocess_features


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference script for VideoMAEPoseClassifier"
    )
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument(
        "--ssl_model_path",
        type=str,
        default="data/weights/ssl_feature_extractor.pth",
        help="Path to the trained SSL feature extractor model file.",
    )
    parser.add_argument(
        "--transformer_model_path",
        type=str,
        default="data/weights/transformer_pose_classifier.pth",
        help="Path to the trained Transformer model file.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for the feature extractor.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for the feature extractor.",
    )

    return parser.parse_args()


def main(
    video_path: str,
    ssl_model_path: str,
    transformer_model_path: str,
    batch_size: int = 8,
    num_workers: int = 4,
):
    # Extract features from the video
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    video_frames_dataset = OneVideoFramesDataset(video_path, transform)
    video_frames_dataloader = DataLoader(
        video_frames_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    print("\n--------------------------------")
    pbar = tqdm(video_frames_dataloader, desc=f"Extracting features from {video_path}")
    video_features = []
    for frames in pbar:
        features = feature_extractor(frames.to(device))
        video_features.append(features.cpu().numpy())
    video_features = np.concatenate(video_features, axis=0)

    print("\nDone extracting features from video!")
    print("--------------------------------\n")

    # Load model configuration and weights
    ckpt = torch.load(transformer_model_path, weights_only=False, map_location="cpu")
    training_config = ckpt["config"]
    data_config = training_config["data_config"]
    model = VideoMAEPoseClassifier(**training_config["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    # Preprocess features
    features_tensor = (
        torch.from_numpy(
            preprocess_features(video_features, data_config["sequence_length"])
        )
        .float()
        .unsqueeze(0)
        .to(device)
    )

    # Perform inference
    with torch.no_grad():
        outputs = model(features_tensor)
        predicted_class = torch.argmax(outputs, dim=1).item()

    # Map predicted class index to class name
    idx_to_class = {v: k for k, v in data_config["dict_class_indices"].items()}
    predicted_label = idx_to_class[predicted_class]

    print(f"Predicted class for the video is: {predicted_label}")


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))

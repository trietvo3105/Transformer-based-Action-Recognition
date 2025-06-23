import os
import re

import cv2
import numpy as np


def increment_path(path: str) -> str:
    """
    Finds the next available path by incrementing a number at the end.
    For example, if '/path/to/exp' exists, it will return '/path/to/exp1'.
    If '/path/to/exp1' also exists, it will return '/path/to/exp2', and so on.
    If the path does not exist, it returns the path itself.
    Args:
        path (str): Path to the directory.

    Returns:
        str: Path to the next available directory.
    """
    if not os.path.exists(path):
        return path
    match = re.search(r"(\d+)$", path)
    if match:
        print(match)
        base = path[: match.start()]
        num = int(match.group(1))
    else:
        base = path
        num = 0

    i = num + 1
    while True:
        new_path = f"{base}{i}"
        if not os.path.exists(new_path):
            os.makedirs(new_path)
            return new_path
        i += 1


def draw_video_landmarks(video_path: str, landmarks: np.ndarray) -> list:
    """
    Draws landmarks on a video.

    Args:
        video_path (str): Path to the input video file.
        landmarks (np.ndarray): Array of shape (num_frames, num_landmarks, 2) containing the normalized keypoints for each frame.

    Returns:
        list: List of frames with landmarks drawn on them.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        for landmark in landmarks[i]:
            cv2.circle(frame, (int(landmark[0]), int(landmark[1])), 5, (0, 0, 255), -1)
        frames.append(frame)
        i += 1
    cap.release()
    return frames


def preprocess_features(features: np.ndarray, sequence_length: int) -> np.ndarray:
    """
    Preprocesses the features to a fixed sequence length.
    """
    features = features.reshape(features.shape[0], -1)
    if features.shape[0] < sequence_length:
        features = np.pad(
            features,
            ((0, sequence_length - features.shape[0]), (0, 0)),
        )
    else:
        features = features[:sequence_length]
    return features


def extract_frames(video_path: str, output_dir: str):
    """
    Extracts frames from a video and saves them to a directory.
    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the extracted frames.
    """
    video_name = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count <= 0:
        print(f"Could not read frames from {video_name}, skipping.")
        return

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_dir, f"{video_name}_f-{i:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

    cap.release()

# Deep Learning-based Video Action Recognition Project

## Project Overview

This project implements a video deep learning pipeline for **human action recognition**, exploring the approach of using a **self-supervised pre-trained frame feature extractor** combined with a **Transformer** encoder. The goal is to demonstrate the feasibility of processing sequential video data for classification tasks.

## Setup

Python 3.11.13 was used to develop this project. So it is the recommended Python version.

Setting up with _anaconda_/_miniconda_ is recommended. To do so:

```
conda create --name dl_video_understanding python=3.11.13
conda activate dl_video_understanding
pip install -r requirements.txt
```

If you don't like using anaconda, just simply run:

```
pip install -r requirements.txt
```

## Usage

### Data Preparation:

Place some short video files (can download from internet, e.g. https://www.pexels.com/) into the `data/raw_videos/{train,val}` directories (for training and validation purpose). These videos should be of simple actions like waving, clapping, or walking.

### Self-Supervised Pre-trained Frame Encoder

This task explores an approach, in which a feature extractor is pre-trained on raw video frames using a self-supervised learning (SSL) method (SimSiam). This allows the model to learn meaningful visual representations without needing any labels. 

These learned features are then extracted and fed into the Transformer pose classifier for the final action classification.

```bash
python scripts/_02_ssl_based_pretrain_encoder.py
```

By default, the trained model will be saved to `data/weights/ssl_feature_extractor.pth`.

### Transformer-Based Video Model

This task uses powerful Transformer-based model (inspired by VideoMAE), which takes the features extracted from the SSL-pre-trained encoder and leverages the Transformer architecture to better capture complex temporal patterns across video frames.

**Training:**
This single script handles feature extraction and trains the Transformer model.

```bash
python scripts/_03_transformer_based_train.py
```

You can customize **training parameters** by modifying the arguments in the script. The trained model, by default, will be saved to `data/weights/transformer_pose_classifier.pth`.

**Inference:**
To run inference with the Transformer-based model:

```bash
python scripts/_03_transformer_based_infer.py path/to/your/video.mp4
```

## Possible future improvements

There are several things that can be explored and enhanced in future work, such as:

-   Use a true video database for human action recognition instead of just some videos.
-   Experimenting with different model for SSL feature extract model (e.g. bigger ResNet, EfficientNet).
-   Add some more good data augmentations to SSL training.
-   Apply data augmentation to videos: Use a sliding window to randomly sample sequences from videos, rather than using the entire padded or trimmed video, as a single input to the models.
-   Use the entire VideoMAE architecture or other Transformers instead of CNN feature extractor + Transformer encoder, to leverage the powerful pretrained embedding layer of the transformer that comes with the corresponding encoder.
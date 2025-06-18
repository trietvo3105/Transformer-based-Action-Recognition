# Deep Learning-based Video Action Recognition Project

## Project Overview

This project implements a video deep learning pipeline for **human action recognition**, exploring the approach of using a **self-supervised pre-trained frame feature extractor** combined with a **Transformer** encoder. The goal is to demonstrate the feasibility of processing sequential video data for classification tasks.

## Setup

Python 3.11.13 was used to develop this project. So it is the recommended Python version.

Setting up with _anaconda_/_miniconda_ is recommended. To do so:

```
conda create --name dl_video_understading python=3.11.13
conda activate dl_video_understading
pip install -r requirements.txt
```

If you don't like using anaconda, just simply run:

```
pip install -r requirements.txt
```

## Usage

### Data Preparation:

Place some short video files (can download from internet, e.g. https://www.pexels.com/) into the `data/raw_videos/{train,val}` directories (for training and validation purpose). These videos should be of simple actions like waving, clapping, or walking.

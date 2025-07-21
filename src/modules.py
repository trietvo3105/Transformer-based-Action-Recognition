import torch.nn as nn
from torchvision import models


class ResnetEncoder(nn.Module):
    """
    ResNet encoder that extracts features from a video frame. Adapted from https://github.com/facebookresearch/simsiam/blob/main/simsiam/builder.py
    """

    def __init__(self):
        super().__init__()
        self.encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        fc_output_dim, fc_input_dim = self.encoder.fc.weight.shape
        self.dim = fc_output_dim
        self.encoder.fc = nn.Sequential(
            nn.Linear(fc_input_dim, fc_input_dim, bias=False),
            nn.BatchNorm1d(fc_input_dim),
            nn.ReLU(inplace=True),  # first layer
            nn.Linear(fc_input_dim, fc_input_dim, bias=False),
            nn.BatchNorm1d(fc_input_dim),
            nn.ReLU(inplace=True),  # second layer
            self.encoder.fc,
            nn.BatchNorm1d(fc_output_dim, affine=False),
        )  # output layer
        self.encoder.fc[6].bias.requires_grad = (
            False  # hack: not use bias as it is followed by BN
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class PredictionHead(nn.Module):
    """
    Prediction head for the SSL feature extractor. Adapted from https://github.com/facebookresearch/simsiam/blob/main/simsiam/builder.py
    Args:
        dim (int): Dimension of the input features.
        hidden_dim (int): Dimension of the MLP hidden layer.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),  # hidden layer
            nn.Linear(hidden_dim, dim),
        )  # output layer

    def forward(self, x):
        x = self.predictor(x)
        return x

import torch
import torch.nn as nn
from transformers import VideoMAEConfig, VideoMAEModel

from src.modules import ResnetEncoder, PredictionHead


class SSLFeatureExtractor(nn.Module):
    """
    SSL feature extractor. Adapted from https://github.com/facebookresearch/simsiam/blob/main/main_simsiam.py
    Used for SSL pretraining only.
    """

    def __init__(self):
        super().__init__()
        self.encoder = ResnetEncoder()
        self.predictor = PredictionHead(
            dim=self.encoder.dim, hidden_dim=self.encoder.dim // 2
        )

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()


class FeatureExtractor(nn.Module):
    """
    SSL Feature extractor. Used for feature extraction (inference only).
    Args:
        ssl_model_path (str): Path to the SSL model.
        freeze (bool): Whether to freeze the encoder.
    """

    def __init__(self, ssl_model_path: str, freeze: bool = True):
        super().__init__()
        ssl_model = SSLFeatureExtractor()
        checkpoint = torch.load(ssl_model_path, weights_only=False, map_location="cpu")
        ssl_model.load_state_dict(checkpoint["state_dict"])
        self.encoder = ssl_model.encoder.encoder
        self.encoder.fc = nn.Identity()

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

    def forward(self, x):
        with torch.no_grad():
            return self.encoder(x)

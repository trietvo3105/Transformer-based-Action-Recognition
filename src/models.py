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


class VideoMAEPoseClassifier(nn.Module):
    """
    A custom Transformer-based classification architecture that uses the VideoMAE encoder followed by a layer-normalization layer and a linear layer to produce logits.
    Args:
        cnn_embedding_dim (int): Dimension of the CNN embeddings produced by the CNN feature extractor.
        num_classes (int): Number of classes.
        pretrained (bool): Whether to use a pretrained model.
    """

    def __init__(
        self, cnn_embedding_dim: int, num_classes: int, pretrained: bool = True
    ):
        super().__init__()

        if pretrained:
            model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
            config = VideoMAEConfig.from_pretrained(model_name)
            videomae_base = VideoMAEModel.from_pretrained(model_name)
        else:
            config = VideoMAEConfig()
            videomae_base = VideoMAEModel(config)

        self.position_embeddings = videomae_base.embeddings.position_embeddings
        self.cls_token = nn.Parameter(torch.randn((1, 1, config.hidden_size)))

        if cnn_embedding_dim != config.hidden_size:
            self.projection = nn.Linear(cnn_embedding_dim, config.hidden_size)
        else:
            self.projection = nn.Identity()

        self.encoder = videomae_base.encoder
        self.fc_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.classifier = nn.Linear(config.hidden_size, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): A tensor of shape
                (batch_size, num_frames, cnn_embedding_dim) containing the
                features extracted by your CNN.
        """
        batch_size, num_frames, _ = x.shape
        x = self.projection(x)  # (batch_size, num_frames, hidden_size)
        cls_tokens = self.cls_token.expand(
            batch_size, -1, -1
        )  # (batch_size, 1, hidden_size)
        x = torch.cat(
            [cls_tokens, x], dim=1
        )  # (batch_size, num_frames + 1, hidden_size)

        if num_frames + 1 > self.position_embeddings.shape[1]:
            raise ValueError(
                f"The number of frames must be less than or equal to the size of the position embeddings: {self.position_embeddings.shape[1]}. "
                f"Try to trim the video to have maximum of {self.position_embeddings.shape[1]} frames"
            )

        x = x + self.position_embeddings[:, : num_frames + 1].to(
            x.device
        )  # (batch_size, num_frames + 1, hidden_size)
        x = self.encoder(x)[0]  # (batch_size, num_frames + 1, hidden_size)
        x = self.fc_norm(x)  # (batch_size, num_frames + 1, hidden_size)
        logits = self.classifier(x[:, 0])  # (batch_size, num_classes)
        return logits

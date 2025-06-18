import torch.nn as nn


class SimSiamLoss(nn.Module):
    """
    SimSiam loss function. Adapted from https://github.com/facebookresearch/simsiam/blob/main/main_simsiam.py
    Args:
        device (str): cuda or cpu.
    """

    def __init__(self, device: str):
        super().__init__()
        self.criterion = nn.CosineSimilarity(dim=1).to(device)

    def forward(self, p1, p2, z1, z2):
        loss = -(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5
        return loss

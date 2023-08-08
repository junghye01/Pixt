import torch
import torch.nn as nn


class MultiLabelSoftMarginLoss(nn.Module):
    def __init__(self, base_loss_weight) -> None:
        super().__init__()
        self._base_loss_weight = base_loss_weight
        self._base_loss_fn = torch.nn.MultiLabelSoftMarginLoss()

    def forward(self, similarity: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._base_loss_weight * self._base_loss_fn(similarity, target)


class MSELoss(nn.Module):
    def __init__(self, base_loss_weight) -> None:
        super().__init__()
        self._base_loss_weight = base_loss_weight
        self._base_loss_fn = torch.nn.MultiLabelSoftMarginLoss()

    def forward(self, similarity: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        similarity = similarity.float()
        target = target.float()
        return self._base_loss_weight * self._base_loss_fn(similarity, target)

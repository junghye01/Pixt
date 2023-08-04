import torch
import torch.nn as nn


class BaseLoss(nn.Module):
    def __init__(self, base_loss_weight) -> None:
        super().__init__()
        self._ce_loss_weight = base_loss_weight
        self._ce_loss_fn = torch.nn.MSELoss()

    def forward(
        self, similarity: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        loss = self._ce_loss_weight * self._ce_loss_fn(similarity,target)
        return loss

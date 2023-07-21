import torch
import torch.nn as nn


class BaseLoss(nn.Module):
    def __init__(self, ce_loss_weight) -> None:
        super().__init__()
        self._ce_loss_weight = ce_loss_weight
        self._ce_loss_fn = torch.nn.CrossEntropyLoss()

    def forward(
        self, logits_per_image: torch.Tensor, logits_per_text: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        probs_per_image = logits_per_image.softmax(dim=-1)
        probs_per_text = logits_per_text.softmax(dim=-1)

        loss = (
            self._ce_loss_fn(probs_per_image, target) + self._ce_loss_fn(probs_per_text, target.T)
        ) / 2
        return loss

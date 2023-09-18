import torch
import torch.nn as nn
import torch.nn.functional as F



def cross_entropy(preds,targets,reduction='none'):
    log_softmax=nn.LogSoftmax(dim=-1)
    loss=(-targets * log_softmax(preds)).sum(1)
    if reduction =='none':
        return loss
    elif reduction =='mean':
        return loss.mean()


class BaseLoss(nn.Module):
    def __init__(self, base_loss_weight,batch_size) -> None:
        super().__init__()
        
        self._ce_loss_weight = base_loss_weight
        #self._ce_loss_fn
        self._batch_size=batch_size
        #self._ce_loss_fn=torch.nn.BCELoss()

    def forward(
        self, images_similarity: torch.Tensor, texts_similarity: torch.Tensor, logits: torch.Tensor
    ) -> torch.Tensor:
        targets=F.softmax(
            (images_similarity + texts_similarity) /2 ,dim=-1
        )
        texts_loss=cross_entropy(logits,targets,reduction='none')
        images_loss=cross_entropy(logits.T,targets.T,reduction='none')
        loss=(images_loss+texts_loss)/self._batch_size
        
        return loss.mean()



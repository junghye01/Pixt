import torch
import torch.nn as nn
import numpy as np


class Accuracy(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, similarity: torch.Tensor, text_en: list[list], text_input: list
    ) -> torch.Tensor:
        batch_accuracy = []
        similarity = (100.0 * similarity).softmax(dim=-1)
        for index in range(len(text_en)):
            true_labels = text_en[index]
            predicted_labels = []
            values, indices = similarity[index].topk(len(true_labels))
      
            
            for _, index in zip(values, indices):
                predicted_labels.append(text_input[index])
            intersection = list(set(true_labels) & set(predicted_labels))
            batch_accuracy.append(round((len(intersection) / len(true_labels)) * 100, 6))
            
        return np.mean(batch_accuracy)





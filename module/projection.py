import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim,projection_dim=256,dropout=0.1):
        super().__init__()
        self.projection=nn.Linear(embedding_dim,projection_dim,dtype=torch.float16)
        self.gelu=nn.GELU()
        self.fc=nn.Linear(projection_dim,projection_dim,dtype=torch.float16)
        self.dropout=nn.Dropout(dropout)
        self.layer_norm=nn.LayerNorm(projection_dim,dtype=torch.float16)

    def forward(self,x):
        projected=self.projection(x)
        x=self.gelu(projected)
        x=self.fc(x)
        x=self.dropout(x)
        x=x+projected
        x=self.layer_norm(x)
        return x
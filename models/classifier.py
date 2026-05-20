import torch
import torch.nn as nn


class MedicalMLP(nn.Module):
    """
    MLP classifier for tabular medical features.
    用于医学表格特征的 MLP 分类器。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        embed_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x, return_embedding: bool = False):
        embedding = self.feature_extractor(x)
        logits = self.classifier(embedding).squeeze(1)

        if return_embedding:
            return logits, embedding
        return logits

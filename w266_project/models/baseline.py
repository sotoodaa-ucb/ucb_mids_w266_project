import torch as torch
import torch.nn as nn
from transformers import AutoModel


class MarkdownModel(nn.Module):
    def __init__(self, model_path):
        super(MarkdownModel, self).__init__()
        self.model_name = 'markdown-model-v1'
        self.model = AutoModel.from_pretrained(model_path)

        # Bert embeddings are 768-d + 1 for code cell percentage.
        self.top = nn.Linear(769, 1)

    def forward(self, ids, mask, features):
        # Embeddings
        x = self.model(ids, mask)[0]

        # Concatenate with features.
        x = torch.cat((x[:, 0, :], features), 1)
        return self.top(x)

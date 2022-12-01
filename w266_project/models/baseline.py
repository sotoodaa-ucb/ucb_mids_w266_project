import torch as torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class MarkdownModel(nn.Module):
    def __init__(self, tokenizer_name: str = 'microsoft/codebert-base'):
        super(MarkdownModel, self).__init__()
        self.model_name = 'markdown-model-v1'
        self.model = AutoModel.from_pretrained(tokenizer_name)
        self.code_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Bert embeddings are 768-d + 1 for code cell percentage.
        self.top = nn.Linear(769, 1)

    def forward(self, ids, mask, features):
        # Embeddings
        x = self.model(ids, mask)[0]

        # Concatenate with features.
        x = torch.cat((x[:, 0, :], features), 1)
        return self.top(x)

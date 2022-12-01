import torch as torch
import torch.nn as nn
from transformers import AutoModel
from transformers import AutoTokenizer


class MarkdownModel(nn.Module):
    def __init__(self):
        super(MarkdownModel, self).__init__()
        self.model_name = 'markdown-model-v1'
        self.code_tokenizer_name = 'microsoft/codebert-base'
        self.model = AutoModel.from_pretrained(self.code_tokenizer_name)
        self.code_tokenizer = AutoTokenizer.from_pretrained(self.code_tokenizer_name)

        # Bert embeddings are 768-d + 1 for code cell percentage.
        self.top = nn.Linear(769, 1)

    def forward(self, ids, mask, features):
        # Embeddings
        x = self.model(ids, mask)[0]

        # Concatenate with features.
        x = torch.cat((x[:, 0, :], features), 1)
        return self.top(x)

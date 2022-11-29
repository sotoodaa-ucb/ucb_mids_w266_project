import torch as torch
import torch.nn as nn
from transformers import AutoModel


class MarkdownModel(nn.Module):
    def __init__(self, code_model: str, markdown_model: str):
        super(MarkdownModel, self).__init__()
        self.code_model = AutoModel.from_pretrained(code_model)
        self.markdown_model = AutoModel.from_pretrained(markdown_model)

        # Bert embeddings are 768-d + 1 for code cell percentage.
        self.top = nn.Linear(1536, 768)
        self.out = nn.Linear(768, 1)

    def forward(self, code_ids, code_mask, markdown_ids, markdown_mask):
        # Embeddings
        code_embeddings = self.code_model(code_ids, code_mask)[0]
        markdown_embeddings = self.markdown_model(markdown_ids, markdown_mask)[0]

        # Concatenate code embeddings with markdown.
        x = torch.cat((code_embeddings[:, 0, :], markdown_embeddings[:, 0, :]), 1)

        return self.out(self.top(x))

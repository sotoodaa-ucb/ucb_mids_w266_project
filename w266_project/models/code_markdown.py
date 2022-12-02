import torch as torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class CodeMarkdownModel(nn.Module):
    def __init__(
        self,
        code_tokenizer_name: str = 'microsoft/codebert-base',
        markdown_tokenizer_name: str = 'bert-base-uncased'
    ):
        super(CodeMarkdownModel, self).__init__()
        self.code_model = AutoModel.from_pretrained(code_tokenizer_name)
        self.markdown_model = AutoModel.from_pretrained(markdown_tokenizer_name)
        self.code_tokenizer = AutoTokenizer.from_pretrained(code_tokenizer_name)
        self.markdown_tokenizer = AutoTokenizer.from_pretrained(markdown_tokenizer_name)

        # Bert embeddings are 1536 (768 x 2) codebert + bert.
        self.top = nn.Linear(1536, 1)

    def forward(self, code_ids, code_mask, markdown_ids, markdown_mask):
        # Embeddings
        code_embeddings = self.code_model(code_ids, code_mask)[0]
        markdown_embeddings = self.markdown_model(markdown_ids, markdown_mask)[0]

        # Concatenate code embeddings with markdown.
        x = torch.cat((code_embeddings[:, 0, :], markdown_embeddings[:, 0, :]), 1)

        return self.top(x)


class CodeMarkdownModelV2(nn.Module):
    def __init__(
        self,
        code_tokenizer_name: str = 'microsoft/codebert-base',
        markdown_tokenizer_name: str = 'bert-base-uncased'
    ):
        super(CodeMarkdownModelV2, self).__init__()
        self.code_model = AutoModel.from_pretrained(code_tokenizer_name)
        self.markdown_model = AutoModel.from_pretrained(markdown_tokenizer_name)
        self.code_tokenizer = AutoTokenizer.from_pretrained(code_tokenizer_name)
        self.markdown_tokenizer = AutoTokenizer.from_pretrained(markdown_tokenizer_name)

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

from typing import List, Union

import torch


class Preprocessor:
    """ Compatible with Baseline model. """
    def __init__(
        self,
        markdown_tokenizer,
        code_tokenizer,
        md_max_len: int = 200,
        total_max_len: int = 400,
        with_features: bool = False
    ):
        self.markdown_tokenizer = markdown_tokenizer
        self.code_tokenizer = code_tokenizer
        self.with_features = with_features
        self.md_max_len = md_max_len
        self.total_max_len = total_max_len

    def preprocess(
        self,
        markdown_inputs: Union[str, List[str]],
        code_inputs: List[List[str]]
    ):
        if self.with_features and isinstance(self.markdown_inputs, str):
            raise ValueError('markdown_input must be List[str] if with_features=True')

        if isinstance(markdown_inputs, list) and markdown_inputs:
            markdown_input = markdown_inputs[0]
        else:
            markdown_input = markdown_inputs

        # Encode markdown into embedding.
        inputs = self.code_tokenizer.encode_plus(
            markdown_input,
            None,
            add_special_tokens=True,
            max_length=self.md_max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True
        )

        try:
            # Encode code into embedding.
            # Batch encode does not like empty lists!
            code_inputs = self.code_tokenizer.batch_encode_plus(
                [str(cell) for cell in code_inputs] if len(code_inputs) > 0 else [''],
                add_special_tokens=True,
                max_length=23,
                padding="max_length",
                truncation=True
            )
        except Exception as e:
            raise ValueError(e)

        n_md = len(markdown_inputs)
        n_code = len(code_inputs)

        # Get percentage of markdown relative to total cells.
        if n_md + n_code == 0:
            features = torch.FloatTensor([0])
        else:
            features = torch.FloatTensor([n_md / (n_md + n_code)])

        # Get markdown embedding tokens.
        ids = inputs['input_ids']
        for x in code_inputs['input_ids']:
            # Exclude separator token.
            ids.extend(x[:-1])

        # Trim to max length.
        ids = ids[:self.total_max_len]

        # Apply padding if code + markdown tokens is less than max.
        if len(ids) < self.total_max_len:
            ids = ids + [self.markdown_tokenizer.pad_token_id, ] * (self.total_max_len - len(ids))

        # Concatenated embeddings input as a tensor.
        ids = torch.LongTensor(ids)

        # Do the same for the attention mask.
        mask = inputs['attention_mask']
        for x in code_inputs['attention_mask']:
            # Remove mask for separator token.
            mask.extend(x[:-1])

        mask = mask[:self.total_max_len]
        if len(mask) != self.total_max_len:
            mask = mask + [self.code_tokenizer.pad_token_id, ] * (self.total_max_len - len(mask))
        mask = torch.LongTensor(mask)

        # Tokens should be equal to the maximum length.
        assert len(ids) == self.total_max_len

        # Tokens, attention mask, markdown percentage feature, and label.
        return ids, mask, features

import re
from typing import List, Union

import numpy as np
import torch
from nltk.stem import WordNetLemmatizer


def links_to_word(text):
    return re.sub("https?:\/\/[^\s]+", " link ", text)


def no_char(text):
    text = re.sub(r"\s+[a-zA-Z]\s+", " ", text)
    text = re.sub(r"\^[a-zA-Z]\s+", " ", text)
    text = re.sub(r"\s+[a-zA-Z]$", " ", text)
    return text


def no_html_tags(text):
    return re.sub("<.*?>", " ", text)


def no_multi_spaces(text):
    return re.sub(r"\s+", " ", text, flags=re.I)


def lemmatize(text):
    stemmer = WordNetLemmatizer()

    tokens = text.split()
    tokens = [stemmer.lemmatize(word) for word in tokens]
    return " ".join(tokens)


def underscore_to_space(text: str):
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    return text


def no_markdown_special(text: str):
    try:
        text = text[0] + re.sub(r"(?<!\n)[\*\+\-\>]", " ", text[1:])
        text = re.sub(r"\(\)\[\]\{\}\<\>\~\|\`\.", " ", text)
    except IndexError:
        return ''
    return text


def code_preprocess(code):
    code = links_to_word(code)
    code = lemmatize(code)
    return code


def markdown_preprocess(code: str):
    """
    1. Replace new lines with unused token.
    2. Remove HTML Tags and special markdown symbols.
    3. Clear html tags first, then markdown...
    """
    code = code.replace("\n", "[unused1]")
    code = links_to_word(code)
    code = no_html_tags(code)
    code = no_markdown_special(code)
    code = no_multi_spaces(code)
    code = lemmatize(code)
    return code


def preprocessor(text: str, cell_type: str):
    return dict(code=code_preprocess, markdown=markdown_preprocess)[cell_type](text)


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


class PreprocessorV2:
    """ Compatible with CodeMarkdown model. """
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
        markdown_inputs = self.markdown_tokenizer.encode_plus(
            markdown_input,
            None,
            add_special_tokens=True,
            max_length=self.md_max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True
        )

        # Encode code into embedding.
        # Batch encode does not like empty lists!
        code_inputs = self.code_tokenizer.batch_encode_plus(
            [str(cell) for cell in code_inputs] if len(code_inputs) > 0 else [''],
            add_special_tokens=True,
            max_length=23,
            padding="max_length",
            truncation=True
        )

        # Get markdown embedding tokens.
        markdown_ids = markdown_inputs['input_ids']
        markdown_ids = markdown_ids[:self.total_max_len]

        # Apply padding if code + markdown tokens is less than max.
        if len(markdown_ids) < self.total_max_len:
            markdown_ids = markdown_ids + \
                [self.markdown_tokenizer.pad_token_id, ] * (self.total_max_len - len(markdown_ids))

        markdown_ids = torch.LongTensor(markdown_ids)

        # Get code embedding tokens.
        code_ids = list(np.array(code_inputs['input_ids']).flatten())
        code_ids = code_ids[:self.total_max_len]

        # Apply padding if code + markdown tokens is less than max.
        if len(code_ids) < self.total_max_len:
            code_ids = code_ids + [self.code_tokenizer.pad_token_id, ] * (self.total_max_len - len(code_ids))

        code_ids = torch.LongTensor(code_ids)

        # Markdown masks
        markdown_mask = markdown_inputs['attention_mask']
        markdown_mask = markdown_mask[:self.total_max_len]

        if len(markdown_mask) != self.total_max_len:
            markdown_mask = markdown_mask + \
                [self.markdown_tokenizer.pad_token_id, ] * (self.total_max_len - len(markdown_mask))
        markdown_mask = torch.LongTensor(markdown_mask)

        # Do the same for the code attention mask.
        code_mask = markdown_inputs['attention_mask']
        code_mask = code_mask[:self.total_max_len]

        if len(code_mask) != self.total_max_len:
            code_mask = code_mask + [self.code_tokenizer.pad_token_id, ] * (self.total_max_len - len(code_mask))
        code_mask = torch.LongTensor(code_mask)

        # Tokens should be equal to the maximum length.
        assert len(markdown_ids) == self.total_max_len
        assert len(code_ids) == self.total_max_len

        # Tokens, attention mask, markdown percentage feature, and label.
        return code_ids, code_mask, markdown_ids, markdown_mask

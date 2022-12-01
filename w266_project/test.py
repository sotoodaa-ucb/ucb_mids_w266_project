import pandas as pd

from w266_project.inference import PyTorchEngine
from w266_project.models.core import ModelType
from w266_project.utils import download_nltk_resources


def main():
    download_nltk_resources()

    # markdown_content = '# This imports numpy library.'
    markdown_content = '# This adds two numbers and returns the sum.'

    code_cells = [
        'import numpy as np',
        'def add(a: int, b: int) -> int: return a + b',
        'def subtract(a: int, b: int) -> int: return a - b',
        'def multiply(a: int, b: int) -> int: return a * b',
        'def divide(a: int, b: int) -> int: return a // b',
    ]

    notebook_content = [
        markdown_content,
        *code_cells
    ]

    model_type = ModelType.CODE_MARKDOWN

    # Get inference engine based on selected model type.
    engine = PyTorchEngine(model_type)

    if model_type == ModelType.BASELINE:
        # Preprocess to obtain ids, masks, and feature tensors based on input.
        ids, mask, features = engine.preprocess(markdown_content, code_cells)

        # Get ranking prediction score.
        prediction = engine.predict(ids, mask, features).detach().numpy()
    elif model_type == ModelType.CODE_MARKDOWN or model_type == ModelType.CODE_MARKDOWN_V2:
        # Preprocess to obtain ids, masks, and feature tensors based on input.
        code_ids, code_mask, markdown_ids, markdown_mask = engine.preprocess(markdown_content, code_cells)

        # Get ranking prediction score.
        prediction = engine.predict(code_ids, code_mask, markdown_ids, markdown_mask).detach().numpy()

    prediction_df = pd.DataFrame(data=notebook_content, columns=['content'])

    # Using pandas rank function, compute the actual rank.
    prediction_df['actual_pct_rank'] = prediction_df.reset_index()['index'].rank(pct=True)
    prediction_df['predicted_pct_rank'] = prediction_df.reset_index()['index'].rank(pct=True)

    # Replace the input markdown row's predicted percentile rank.
    prediction_df.loc[prediction_df['content'] == markdown_content, 'predicted_pct_rank'] = prediction

    # Sort based on predicted percentile rank to get final notebook prediction.
    prediction_df = prediction_df.sort_values('predicted_pct_rank')

    print(prediction_df.to_markdown(index=False))


if __name__ == '__main__':
    main()

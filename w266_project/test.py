import nltk
import pandas as pd

from w266_project.inference import PyTorchEngine
from w266_project.models.core import ModelType

nltk.download('wordnet')
nltk.download('omw-1.4')


def main():
    markdown_content = '# This adds two numbers and returns the sum.'
    # markdown_content = '# This imports numpy library.'

    code_cells = [
        'import numpy as np',
        'def add(a: int, b: int) -> int: return a + b',
        'def multiply(a: int, b: int) -> int: return a * b'
    ]
    correct_order = [
        'import numpy as np',
        markdown_content,
        'def add(a: int, b: int) -> int: return a + b',
        'def multiply(a: int, b: int) -> int: return a * b'
    ]

    # Get inference engine based on selected model type.
    engine = PyTorchEngine(ModelType.BASELINE)

    # Preprocess to obtain ids, masks, and feature tensors based on input.
    ids, mask, features = engine.preprocess(markdown_content, code_cells)

    # Get ranking prediction score.
    prediction = engine.predict(ids, mask, features).detach().numpy()

    prediction_df = pd.DataFrame(data=correct_order, columns=['content'])

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

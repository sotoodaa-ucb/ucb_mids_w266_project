import nltk

from w266_project.dataset.core import MarkdownDatasetModule
from w266_project.models.baseline import MarkdownModel
from w266_project.train import train


def main():
    nltk.download('wordnet')
    nltk.download('omw-1.4')

    dataset = MarkdownDatasetModule('./w266_project/data/train_all.parquet', './w266_project/data/train_orders.parquet')

    train_loader, val_loader, _test_loader = dataset.get_loaders()

    model = MarkdownModel('microsoft/codebert-base')
    model, _y_pred = train(model, train_loader, val_loader, dataset.val_df, dataset.order_df, epochs=1)


if __name__ == '__main__':
    main()

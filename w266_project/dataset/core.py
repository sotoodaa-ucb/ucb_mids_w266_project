import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer


class MarkdownDataset(Dataset):
    def __init__(
        self,
        markdown_rows: pd.DataFrame,
        features: dict,
        total_max_len: int = 400,
        md_max_len: int = 200,
        model_name: str = 'microsoft/codebert-base'
    ):
        super().__init__()
        self.markdown_rows = markdown_rows.reset_index(drop=True)
        self.features = features
        self.md_max_len = md_max_len
        self.total_max_len = total_max_len
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            do_lower_case=True,
            use_fast=True
        )

    def __getitem__(self, index):
        row = self.markdown_rows.iloc[index]

        # Encode markdown into embedding.
        inputs = self.tokenizer.encode_plus(
            row.source,
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
            code_cells = self.features[row.id]["codes"]
            code_inputs = self.tokenizer.batch_encode_plus(
                [str(cell) for cell in code_cells] if len(code_cells) > 0 else [''],
                add_special_tokens=True,
                max_length=23,
                padding="max_length",
                truncation=True
            )
        except Exception as e:
            # print(self.features[row.id]["codes"])
            code_cells = self.features[row.id]["codes"]
            print(len(code_cells))
            # print([str(cell) for cell in code_cells] if len(code_cells) > 0 else [''])
            raise ValueError(e)

        # Other features (number of markdown cells, number of code cells)
        n_md = self.features[row.id]["total_md"]
        n_code = self.features[row.id]["total_code"]

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
            ids = ids + [self.tokenizer.pad_token_id, ] * (self.total_max_len - len(ids))

        # Concatenated embeddings input as a tensor.
        ids = torch.LongTensor(ids)

        # Do the same for the attention mask.
        mask = inputs['attention_mask']
        for x in code_inputs['attention_mask']:
            # Remove mask for separator toekn.
            mask.extend(x[:-1])

        mask = mask[:self.total_max_len]
        if len(mask) != self.total_max_len:
            mask = mask + [self.tokenizer.pad_token_id, ] * (self.total_max_len - len(mask))
        mask = torch.LongTensor(mask)

        # Tokens should be equal to the maximum length.
        assert len(ids) == self.total_max_len

        # Tokens, attention mask, markdown percentage feature, and label.
        return ids, mask, features, torch.FloatTensor([row.pct_rank])

    def __len__(self):
        return self.markdown_rows.shape[0]


class MarkdownDatasetModule:
    """Encapsulates Markdown dataset into a single object.

    :param markdown_rows: Pandas dataframe containing markdown content.
    :param features: Extra features (number code cells,
    :param md_max_len: Maximum length of markdown tokenized embedding.
    :param total_max_len: Maximum Length of the tokenized input to bert.
    :param model_name: Name of pretrained bert base model.

    :attr tokenizer: Tokenizer class based on provided model_name.
    """
    def __init__(
        self,
        all_path: str,
        order_path: str,
        total_max_len: int = 400,
        md_max_len: int = 200,
        valid_ratio: float = 0.3,
        test_ratio: float = 0.1,
        model_name: str = 'microsoft/codebert-base'
    ):
        super().__init__()
        self.all_path = all_path,
        self.order_path = order_path,
        self.md_max_len = md_max_len
        self.total_max_len = total_max_len
        self.model_name = model_name
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio

        self.all_df = pd.read_parquet(all_path)
        self.order_df = pd.read_parquet(order_path)

        # Orders dataframe currently contains cell orders as a string, i.e "a b c"
        # We want to convert that into a list of strings: ["a", "b", "c"]
        self.order_df['cell_order'] = self.order_df['cell_order'].str.split(' ').tolist()

        # Create label.
        self.all_df['pct_rank'] = self.all_df['order'] / self.all_df.groupby("id")["cell"].transform("count")

        train_splitter = GroupShuffleSplit(n_splits=1, test_size=self.valid_ratio + self.test_ratio, random_state=0)
        val_splitter = GroupShuffleSplit(n_splits=1, test_size=self.test_ratio, random_state=0)

        # Split into train, (val + test) - 60% - 40%.
        train_ind, val_ind = next(train_splitter.split(self.all_df, groups=self.all_df["ancestor_id"]))

        self.train_df = self.all_df.loc[train_ind].reset_index(drop=True)
        self.train_features = self.get_features(self.train_df)

        val_test_df = self.all_df.loc[val_ind].reset_index(drop=True)

        # Split val into val, test - 90% - 10%.
        val_ind, test_ind = next(val_splitter.split(val_test_df, groups=val_test_df["ancestor_id"]))

        self.val_df = val_test_df.loc[val_ind].reset_index(drop=True)
        self.val_features = self.get_features(self.val_df)

        self.test_df = val_test_df.loc[test_ind].reset_index(drop=True)
        self.test_features = self.get_features(self.test_df)

        self.markdown_train = self.train_df[self.train_df['cell_type'] == 'markdown']
        self.markdown_val = self.val_df[self.val_df['cell_type'] == 'markdown']
        self.markdown_test = self.test_df[self.test_df['cell_type'] == 'markdown']

    def get_features(self, df):
        features = dict()

        # Group by notebook and loop through unique notebooks.
        for idx, sub_df in tqdm(df.groupby("id")):
            features[idx] = dict()

            # Get count of markdown cells in current notebook.
            total_md = sub_df[sub_df.cell_type == "markdown"].shape[0]

            # Get count of code cells in current notebook.
            code_sub_df = sub_df[sub_df.cell_type == "code"]
            total_code = code_sub_df.shape[0]

            # Sample 20 code cells.
            # codes = sample_cells(code_sub_df.source.values, 20)
            codes = code_sub_df.source.values
            features[idx]["total_code"] = total_code
            features[idx]["total_md"] = total_md
            features[idx]["codes"] = codes
        return features

    def get_loaders(self):

        train_ds = MarkdownDataset(
            self.markdown_train,
            features=self.train_features,
            total_max_len=self.total_max_len,
            md_max_len=self.md_max_len,
            model_name=self.model_name
        )

        val_ds = MarkdownDataset(
            self.markdown_val,
            features=self.val_features,
            total_max_len=self.total_max_len,
            md_max_len=self.md_max_len,
            model_name=self.model_name
        )

        test_ds = MarkdownDataset(
            self.markdown_test,
            features=self.test_features,
            total_max_len=self.total_max_len,
            md_max_len=self.md_max_len,
            model_name=self.model_name
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
            shuffle=True
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
            shuffle=False
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
            shuffle=False
        )

        return train_loader, val_loader, test_loader

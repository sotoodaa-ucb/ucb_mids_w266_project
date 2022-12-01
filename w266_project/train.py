import gc
import os
import sys
from datetime import datetime

import nltk
import numpy as np
import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from w266_project.dataset.core import MarkdownDatasetModule
from w266_project.metrics import kendall_tau
from w266_project.models.baseline import MarkdownModel

nltk.download('wordnet')
nltk.download('omw-1.4')


def main():
    # Load dataset from disk.
    dataset = MarkdownDatasetModule('./w266_project/data/train_all.parquet', './w266_project/data/train_orders.parquet')

    # Obtain PyTorch data loaders.
    train_loader, val_loader, _test_loader = dataset.get_loaders()

    # Instantiate the model.
    model = MarkdownModel()

    # Begin training process.
    model, _ = train(model, train_loader, val_loader, dataset.val_df, dataset.order_df, epochs=1)

    # Clear resources.
    cleanup(model)


def cleanup(model):
    if model:
        model.cpu()
        del model
    gc.collect()
    torch.cuda.empty_cache()


def train(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_df: pd.DataFrame,
    order_df: pd.DataFrame,
    epochs: int,
    use_wandb: bool = False
):
    if use_wandb:
        wandb.init(project="w266-project", entity="sotoodaa", name='markdown-model-baseline-early-stopping')
        wandb.watch(model, log_freq=1000)

    if not os.path.exists('./tmp'):
        os.mkdir('./tmp')

    np.random.seed(0)

    if torch.cuda.is_available():
        model = model.cuda()

    early_stop_count = 0
    patience = 5
    best_loss = 1_000_000

    # Creating optimizer and lr schedulers
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    num_train_optimization_steps = int(epochs * len(train_loader) / 4)

    # To reproduce BertAdam specific behavior set correct_bias=False
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=3e-5,
        correct_bias=False
    )

    # PyTorch scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.05 * num_train_optimization_steps,
        num_training_steps=num_train_optimization_steps
    )

    criterion = torch.nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler()

    for e in range(epochs):
        model.train()
        tbar = tqdm(train_loader, file=sys.stdout)
        loss_list = []
        preds = []
        labels = []

        # Train
        for idx, data in enumerate(tbar):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            ids, mask, features, target = [dp.cuda() if torch.cuda.is_available() else dp.cpu() for dp in data]

            # Compute loss
            with torch.cuda.amp.autocast():
                pred = model(ids, mask, features)
                loss = criterion(pred, target)

            # Backprop
            scaler.scale(loss).backward()

            # Update optimizer and scheduler.
            if idx % 4 == 0 or idx == len(tbar) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            loss_list.append(loss.detach().cpu().item())
            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

            # Compute mean loss.
            avg_loss = np.round(np.mean(loss_list), 4)

            if idx % 25_000 == 0 and avg_loss < best_loss:
                # Track best performance, and save the model's state
                best_loss = avg_loss
                model_path = 'tmp/{}_{}_{}_{}'.format(model.model_name, timestamp, e, best_loss)
                torch.save(model.state_dict(), model_path)

            if idx % 1000 == 0 and avg_loss < best_loss and use_wandb:
                wandb.log({
                    'avg_loss': avg_loss,
                    'best_loss': best_loss
                })
                early_stop_count = 0

            # Early stopping
            if avg_loss > best_loss:
                early_stop_count += 1

                if early_stop_count > patience:
                    model_path = 'tmp/{}_{}_{}_{}'.format(model.model_name, timestamp, e, best_loss)
                    torch.save(model.state_dict(), model_path)
                    break

            # Update progress bar.
            tbar.set_description(f"Epoch {e + 1} Loss: {avg_loss} lr: {scheduler.get_last_lr()}")

    # Evaluation
    model.eval()

    tbar = tqdm(val_loader, file=sys.stdout)

    preds = []
    labels = []

    with torch.no_grad():
        for idx, data in enumerate(tbar):
            ids, mask, features, target = [dp.cuda() if torch.cuda.is_available() else dp.cpu() for dp in data]

            with torch.cuda.amp.autocast():
                pred = model(ids, mask, features)

            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

    _, y_pred = np.concatenate(labels), np.concatenate(preds)

    # Create a placeholder prediction.
    val_df["pred"] = val_df.groupby(["id", "cell_type"])["order"].rank(pct=True)

    # Replace pred column with predictions (only markdown cells since only markdown cells
    # are randomized).
    val_df.loc[val_df["cell_type"] == "markdown", "pred"] = y_pred

    # Sort based on the predicted ranks, then obtain the order of cells as a list.
    y_dummy = val_df.sort_values("pred").groupby('id')['cell'].apply(list)

    # Get predictions in the same format as actuals.
    prediction_cell_orders = y_dummy.to_frame()['cell']

    # Based on the notebook index, obtain the actual order from orders dataframe.
    actual_cell_orders = order_df.set_index('id').loc[y_dummy.index]['cell_order']

    # Compute metric.
    kendall_tau_score = kendall_tau(actual_cell_orders, prediction_cell_orders)
    print("Preds score", kendall_tau_score)

    return model, y_pred


if __name__ == '__main__':
    main()

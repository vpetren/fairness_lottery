import os
import logging
import json
import argparse
import sys
import copy
import numpy as np
import utils
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter, defaultdict

import torch
import torchtext
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.datasets import text_classification
from sklearn.metrics import f1_score

from models import TextSentiment, LSTMSentiment
from modules import DRO_loss
from robust_losses import RobustLoss
from prepare_data import load_demographic_torchtext, imdb_data


#torch.manual_seed(1234)


def train_step(model, batch, opt, criterion, device, robust=None):
    opt.zero_grad()
    text, text_lengths = batch.text

    predictions = model(text, text_lengths)
    if robust:
        loss = robust(criterion(predictions, batch.label))
    else:
        loss = criterion(predictions, batch.label)
    acc = accuracy(predictions, batch.label)

    loss.backward()
    opt.step()

    return loss, acc


def calc_f1(preds, labels):
    preds = preds.argmax(1).cpu()
    labels = labels.cpu()
    return f1_score(y_true=labels, y_pred=preds, average='macro')


def accuracy(preds, labels):
    if len(preds.shape) == 1:
        preds = preds.unsqueeze(0)
    accuracy = (preds.argmax(1) == labels).float().mean().item()
    return accuracy


def get_optimizer(model, lr):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    #return torch.optim.SGD(model.parameters(), lr=lr)


def label_distribution(dset):
    labels = []
    for batch in dset:
        lbls = batch.label.tolist()
        for lab in lbls:
            labels.append(lab)
    counter = Counter(labels)
    print(counter)
    dists = [counter[i] / float(len(labels)) for i in range(len(counter))]
    print(dists)


def plot(values):
    x = [i + 1 for i in range(len(values))]
    plt.plot(x, values)
    plt.show()


def test(model, data):
    total_accuracy = []
    total_f1 = []
    model.eval()
    num_batches = len(data)
    preds, labs = [], []
    for batch in data:
        with torch.no_grad():
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze()
            acc = accuracy(predictions, batch.label)
            f1 = calc_f1(predictions, batch.label)
            total_accuracy.append(acc)
            total_f1.append(f1)
            preds.extend(predictions.argmax(1).tolist())
            labs.extend(batch.label.tolist())

    # In case that nothing in the dataset
    final_acc, final_f1 = 0.0, 0.0
    if not total_accuracy == []:
        final_acc = sum(total_accuracy) / len(total_accuracy)
    if not total_f1 == []:
        final_f1 = sum(total_f1) / len(total_f1)

    return final_acc, final_f1, preds, labs

def hparam_search():
    pass

def train_and_valid(model, lr_, num_pruning_iter, num_epochs, train_data, val_data, device, pruning, pruning_method, pruning_structure, test_data=None, dro=False, early=False):
    reinit = False
    optimizer = get_optimizer(model, lr_)
    criterion = nn.CrossEntropyLoss(reduction='none') if dro else nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    robust_loss = RobustLoss(geometry='chi-square', size=1.0, reg=0.01).to(device) if dro else None
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)

    initial_state_dict = copy.deepcopy(model.state_dict())
    patience = 5
    pruning_metrics = defaultdict(list)
    for prune_iter in range(num_pruning_iter):
        if not prune_iter == 0:
            model.prune_by_percentile(amount=pruning, pruning_structure=pruning_structure, pruning_method=pruning_method)

            if not reinit:
                utils.reset_initialization(model, initial_state_dict)

            optimizer = get_optimizer(model, lr_)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)

        remaining_weights = 1.0 * ((1.0 - pruning)**(prune_iter)) * 100
        print(f"--- Pruning Level [RUN 0:{prune_iter}/{num_pruning_iter}]: {remaining_weights:.1f}% of weights remaining ---")
        utils.print_model_sparsity(model)

        # Train the model
        losses, accs = [], []
        n_non_improvements = 0
        best_score = 0.
        best_model = copy.deepcopy(model.state_dict())
        for epoch in range(num_epochs):
            epoch_loss, epoch_accs = 0., 0.
            model.train()
            for batch in tqdm(train_data, desc="Epoch {}, Pruning iteration {}:".format(epoch, prune_iter)):
                loss, acc = train_step(model, batch, optimizer, criterion, device, robust=robust_loss)
                losses.append(loss.item())
                epoch_loss += loss.item()
                epoch_accs += acc

            # Adjust the learning rate
            scheduler.step()

            # Early stopping
            if early:
                _, val_f1, _, _ = test(model, val_data)
                if val_f1 > best_score:
                    best_model = copy.deepcopy(model.state_dict())
                    n_non_improvements = 0
                    best_score = val_f1
                else:
                    n_non_improvements += 1

                if n_non_improvements >= patience:
                    print(f'F1 score did not improve for {patience} epochs, stopping training')
                    break

            print(f"total epoch loss: {epoch_loss}")
        # Load best_model if stopping early
        if n_non_improvements > 0 and early:
            model.load_state_dict(best_model)

        val_acc, val_f1, _, _ = test(model, val_data)
        train_acc, train_f1, _, _ = test(model, train_data)
        print(f"train acc: {train_acc:.4f}, f1: {train_f1:.4f}")
        print(f"val acc: {val_acc:.4f}, f1: {val_f1:.4f}")

        pruning_metrics['rem_w'].append(remaining_weights)
        pruning_metrics['val_acc'].append(val_acc)
        pruning_metrics['val_f1'].append(val_f1)

        if test_data is not None:
            for demographic, test_dataset in test_data:
                dem_acc, dem_f1, preds, labs = test(model, test_dataset)
                pruning_metrics[f"{demographic}_acc"].append(dem_acc)
                pruning_metrics[f"{demographic}_f1"].append(dem_f1)
                pruning_metrics[f"{demographic}_preds"].append(preds)
                pruning_metrics[f"{demographic}_labs"].append(labs)
                print(f"{demographic} acc: {dem_acc:.4f}, f1: {dem_f1:.4f}")

    return pruning_metrics


def save_scores(scores, m_args):
    country = m_args.data.split('/')[-1]
    dro = 'dro_' if args.dro else ''
    directory = f"pruning_{country}/{m_args.model}_{dro}{m_args.pruning_structure}_{m_args.pruning_method}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    base_fp = f"pruning_metrics_{country}_{m_args.model}_{m_args.num_pruning_iter}_{m_args.pruning_structure}"
    scores['args'] = vars(m_args)
    for i in range(10):
        fn = f"{directory}/{base_fp}_run{i}.json"
        if not os.path.isfile(fn):
            with open(fn, 'w') as fp:
                json.dump(scores, fp)
            break


def main(args):
    num_epochs = args.num_epochs
    num_pruning_iter = args.num_pruning_iter
    embed_dim = args.embed_dim
    hidden_dim = args.hidden_dim
    batch_size = args.batch_size
    pruning_method = args.pruning_method
    pruning_structure = args.pruning_structure
    lr = args.lr
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # args.device
    data = args.data
    pruning = args.pruning_pct

    logging.basicConfig(level=getattr(logging, args.logging_level))

    train_dataset, val_dataset, test_dataset, text_vocab, labels = load_demographic_torchtext(args.data, batch_size, device)

    #import pdb; pdb.set_trace()

    if args.model == 'lstm':
        num_output_nodes = len(labels)
        num_layers = args.lstm_layers
        bidirection = args.bidirectional
        dropout = args.dropout
        model = LSTMSentiment(len(text_vocab), embed_dim, hidden_dim, num_output_nodes, num_layers,
                              bidirectional=bidirection, dropout=dropout).to(device)
    elif args.model == 'mlp':
        model = TextSentiment(len(text_vocab), embed_dim, hidden_dim, len(labels)).to(device)

    metrics = train_and_valid(model, lr, num_pruning_iter, num_epochs, train_dataset, val_dataset,
                              device, pruning, pruning_method, pruning_structure, test_data=test_dataset,
                              dro=args.dro)

    save_scores(metrics, args)
    #if not isinstance(test_dataset, list):
    #    print("\nTest - Accuracy: {}".format(test(best_model, test_dataset)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a text classification model on text classification datasets.')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='num epochs (default=5)')
    parser.add_argument('--num-pruning-iter', type=int, default=1,
                        help='number of pruning iterations (default=10)')
    parser.add_argument('--embed-dim', type=int, default=64,
                        help='embed dim. (default=64)')
    parser.add_argument('--hidden-dim', type=int, default=32,
                        help='embed dim. (default=32)')
    parser.add_argument('--lstm-layers', type=int, default=2,
                        help='n lstm layers. (default=2)')
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='batch size (default=16)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default=)')
    parser.add_argument('--lr-gamma', type=float, default=0.8,
                        help='gamma value for lr (default=0.8)')
    parser.add_argument('--data', default='data/trustpilot',
                        help='data directory (default=.data)')
    parser.add_argument('--save-model-path',
                        help='path for saving model')
    parser.add_argument('--logging-level', default='WARNING',
                        help='logging level (default=WARNING)')
    parser.add_argument('--pruning_pct', type=float, default=0.1,
                        help='fraction of units pruned per epoch (default=0.1)')
    parser.add_argument('--model', default='mlp',
                        choices=['lstm', 'mlp'])
    parser.add_argument('--pruning_structure', default='layerwise',
                        choices=['layerwise', 'global'])
    parser.add_argument('--pruning_method', default='magnitude',
                        choices=['magnitude', 'random'])
    parser.add_argument('--hparam_search', action='store_true')
    parser.add_argument('--dro', action='store_true')

    args = parser.parse_args()

    main(args)

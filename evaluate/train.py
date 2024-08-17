import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    cohen_kappa_score
)

import copy
from sklearn import preprocessing
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
import torch.optim as optim
import torch
import torch.nn as nn
import collections

import sys
sys.path.append('../')
from diffusion import NormalDataset, worker_init_fn

from inception import InceptionForSignal

class Logger:
    def __init__(self, filename):
        self.console = sys.stdout
        self.file = open(filename, 'w')
    def write(self, message):
        self.console.write(message)
        self.file.write(message)
    def flush(self):
        self.console.flush()
        self.file.flush()

def get_class_weights(y):
    # obtain inverse of frequency as weights for the loss function
    counter = collections.Counter(y)
    for i in range(len(counter)):
        if i not in counter.keys():
            counter[i] = 1

    num_samples = len(y)
    weights = [0] * len(counter)
    for idx in counter.keys():
        weights[idx] = 1.0 / (counter[idx] / num_samples)
    print("Weight tensor: ")
    print(weights)
    return weights

def train_model(model, train_loader, valid_loader, w, cfg, device):
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, amsgrad=True)
    loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor(w).to(device))

    train_losses, train_acces = [], []
    valid_losses, valid_acces = [], []
    best_acc = 0.0
    for epoch in range(cfg.epochs):
        model.train()
        running_loss, running_acc = [], []
        for i, (X, Y) in enumerate(train_loader):
            X, Y = Variable(X), Variable(Y)
            x = X.to(device, dtype=torch.float)
            y = Y.to(device, dtype=torch.long)

            pred, _ = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pred_y = torch.argmax(pred, dim=1)
            train_acc = torch.sum(pred_y == y) / (pred_y.size()[0])

            running_loss.append(loss.cpu().detach().numpy())
            running_acc.append(train_acc.cpu().detach().numpy())
        
        val_loss, val_acc, _, _ = evaluate_model(model, valid_loader, device, nn.CrossEntropyLoss(), cfg)
        
        train_losses.append(np.mean(running_loss))
        train_acces.append(np.mean(running_acc))
        valid_losses.append(val_loss)
        valid_acces.append(val_acc)
        
        if best_acc < val_acc:
            best_acc = val_acc
            best = {'epoch': epoch, 'model': copy.deepcopy(model)}
        
        epoch_len = len(str(cfg.epochs))
        print_msg = (
            f'[{epoch:>{epoch_len}}/{cfg.epochs:>{epoch_len}}] '
            + f'train_loss: {np.mean(running_loss):.5f} '
            + f'valid_loss: {val_loss:.5f} '
            + f'train_acc: {np.mean(running_acc):5f} '
            + f'valid_acc: {val_acc:.5f}'
        )
        print(print_msg)
    log = {
        'loss': {'train': train_losses, 'valid': valid_losses},
        'acc' : {'train': train_acces, 'valid': valid_acces}
    }
    return log, best

def evaluate_model(model, data_loader, device, loss_fn, cfg):
    model.eval()
    losses, acces = [], []
    trues, preds = [], []
    for i, (X, Y) in enumerate(data_loader):
        with torch.no_grad():
            X, Y = Variable(X), Variable(Y)
            x = X.to(device, dtype=torch.float)
            y = Y.to(device, dtype=torch.long)
            pred = model(x)
            loss = loss_fn(pred, y)
            pred_y = torch.argmax(pred, dim=1)
            acc = torch.sum(pred_y == y) /(list(pred_y.size())[0])
            
            losses.append(loss.cpu().detach().numpy())
            acces.append(acc.cpu().detach().numpy())
            trues.append(y)
            preds.append(pred_y)
    losses = np.array(losses)
    acces = np.array(acces)
    return np.mean(losses), np.mean(acces), trues, preds

def plot_fig(log):
    fig = plt.figure(figsize=(16,4))

    fig.add_subplot(1,2,1)
    plt.plot(log['loss']['train'], label='train')
    plt.plot(log['loss']['valid'], label='valid')
    plt.title('loss')
    plt.legend()

    fig.add_subplot(1,2,2)
    plt.plot(log['acc']['train'], label='train')
    plt.plot(log['acc']['valid'], label='valid')
    plt.title('accuracy')
    plt.legend()
    return plt

def test_evaluation(trues, preds, best_epoch):
    true = torch.cat(trues).cpu().numpy()
    pred = torch.cat(preds).cpu().numpy()
    print('best epoch: {}'.format(best_epoch))
    
    acc = accuracy_score(true, pred)
    rec = recall_score(true, pred, average='macro')
    pre = precision_score(true, pred, average='macro')
    f1s = f1_score(true, pred, average='macro')
    print('accuracy :', acc)
    print('recall   :', rec)
    print('precision:', pre)
    print('f1       :', f1s)
    
    fig = plt.figure(figsize=(8,8))
    sns.heatmap(confusion_matrix(true, pred), square=True, cbar=True, annot=True, cmap='Blues')
    plt.xlabel('pred')
    plt.ylabel('true')
    return (
        acc,
        rec,
        pre,
        f1s,
        plt
    )


def main(cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    os.makedirs(cfg.results_dir, exist_ok=True)
    sys.stdout = Logger(cfg.results_dir+'/output.log')
    
    ##############
    # setup data #
    ##############
    x = np.load(cfg.data_path+'/X.npy').transpose(0, 2, 1)
    y_dict = {'sleep':0, 'sedentary':1, 'light':2, 'moderate-vigorous':3}
    y_str = np.load(cfg.data_path+'/Y.npy')
    y = np.array([y_dict[i] for i in y_str])
    
    x_train, x_, y_train, y_, = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)
    x_valid, x_test, y_valid, y_test = train_test_split(x_, y_, test_size=0.5, random_state=0, stratify=y_)
    
    w_train = get_class_weights(y_train)

    train_dataset = NormalDataset(x_train, y_train, name="train")
    valid_dataset = NormalDataset(x_valid, y_valid, name="val")
    test_dataset  = NormalDataset(x_test, y_test, name="test")

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    
    ###############
    # setup model #
    ###############
    model = InceptionForSignal(
        output_blocks=[3],
        num_classes=cfg.num_classes,
    ).to(device)
    
    #######
    # run #
    #######
    log, best = train_model(model, train_loader, valid_loader, w_train, cfg, device)
    torch.save(best['model'].state_dict(), cfg.results_dir+'/bests.pt')
    ##########
    # figure #
    ##########
    import matplotlib.pyplot as plt
    plt = plot_fig(log)
    plt.savefig(cfg.results_dir+'/log.png')
    #############
    # test eval #
    #############
    import matplotlib.pyplot as plt
    loss, acc, preds, trues = evaluate_model(best['model'], test_loader, device, nn.CrossEntropyLoss(), cfg)
    acc, rec, pre, f1, plt = test_evaluation(preds, trues, best['epoch'])
    plt.savefig(cfg.results_dir+'/confusion_matrix.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--results-dir", type=str, default='results')
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.set_defaults(
        data_path='<data path>',
    )
    args = parser.parse_args()
    main(args)
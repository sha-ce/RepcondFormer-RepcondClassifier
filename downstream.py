import os
import sys
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
# from torch.autograd import Variable
from torchvision import transforms
import torch.optim as optim
import torch
import torch.nn as nn
import collections
from diffusion import diffusion_transformer as dits
import diffusion.models.transformer as sits
from diffusion import create_diffusion, Dataset, NormalDataset, worker_init_fn
from diffusion.diffusion_utils import bool_flag

import warnings
warnings.simplefilter('ignore', UserWarning)


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
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-3, amsgrad=True)
    scheduler = None
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=cfg.dtype).to(device))

    train_losses, train_acces, train_f1s = [], [], []
    valid_losses, valid_acces, valid_f1s = [], [], []
    best_acc, best_f1 = 0., 0.
    for epoch in range(cfg.epochs):
        model.train()
        running_loss, running_acc, running_f1 = [], [], []
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device, dtype=cfg.dtype)
            y = y.to(device, dtype=torch.uint8)

            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

            pred_y = torch.argmax(pred, dim=1)
            train_acc = accuracy_score(y.cpu(), pred_y.cpu())
            train_f1 = f1_score(y.cpu(), pred_y.cpu(), average='macro')

            running_loss.append(loss.item())
            running_acc.append(train_acc)
            running_f1.append(train_f1)
        
        val_loss, val_acc, val_f1, _, _ = evaluate_model(model, valid_loader, nn.CrossEntropyLoss(), device=device, dtype=cfg.dtype)
        
        train_losses.append(np.mean(running_loss))
        train_acces.append(np.mean(running_acc))
        train_f1s.append(np.mean(running_f1))
        valid_losses.append(val_loss)
        valid_acces.append(val_acc)
        valid_f1s.append(val_f1)
        
        if best_f1 <= val_f1:
            best_f1 = val_f1
            best = {'epoch': epoch, 'model': copy.deepcopy(model)}
            update_msg = 'Update: f-score'
            if best_acc <= val_acc:
                best_acc = val_acc
                update_msg += ' & accuracy'
        elif best_acc <= val_acc:
            best_acc = val_acc
            best = {'epoch': epoch, 'model': copy.deepcopy(model)}
            update_msg = 'Update: accuracy'
        else:
            update_msg = ''
        
        epoch_len = len(str(cfg.epochs))
        print_msg = (
            f'[{epoch:>{epoch_len}}/{cfg.epochs:>{epoch_len}}] '
            + f'Loss: [train: {np.mean(running_loss):.4f}, valid: {val_loss:.4f}] '
            + f'Accuracy: [train: {np.mean(running_acc):.4f}, valid: {val_acc:.4f}] '
            + f'F-score: [train: {np.mean(running_f1):.4f}, valid: {val_f1:.4f}] '
            + update_msg
        )
        print(print_msg)
    log = {
        'loss': {'train': train_losses, 'valid': valid_losses},
        'acc' : {'train': train_acces, 'valid': valid_acces}
    }
    return log, best

def evaluate_model(model, loader, loss_fn, device='cpu', dtype=torch.float):
    model.eval()
    losses, trues, preds = [], [], []
    for i, (x, y) in enumerate(loader):
        x = x.to(device, dtype=dtype)
        y = y.to(device, dtype=torch.uint8)
        with torch.no_grad():
            pred = model(x)
            loss = loss_fn(pred, y)
            pred_y = torch.argmax(pred, dim=1)
        
        losses.append(loss.item())
        trues.append(y.detach().cpu())
        preds.append(pred_y.detach().cpu())
    trues = torch.cat(trues)
    preds = torch.cat(preds)
    return (
        np.mean(np.array(losses)),
        accuracy_score(trues, preds),
        f1_score(trues, preds, average='macro'),
        trues,
        preds,
    )

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
    print('best epoch: {}'.format(best_epoch))
    
    acc = accuracy_score(trues, preds)
    rec = recall_score(trues, preds, average='macro')
    pre = precision_score(trues, preds, average='macro')
    f1s = f1_score(trues, preds, average='macro')
    print('accuracy :', acc)
    print('recall   :', rec)
    print('precision:', pre)
    print('f1       :', f1s)
    
    fig = plt.figure(figsize=(8,8))
    sns.heatmap(confusion_matrix(trues, preds), square=True, cbar=True, annot=True, cmap='Blues')
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
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    cfg.results_dir = os.path.join(
        cfg.results_dir,
        cfg.dataset,
        'transfer' if cfg.transfer else 'finetune',
        cfg.ckpt.split('/')[-3].split('-')[0]+'_ckpt'+cfg.ckpt.split('/')[-1][:-3]+'_'+datetime.now().strftime('%d-%m-%Y_%H:%M:%S'),
    )
    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.results_dir+'/bests/', exist_ok=True)
    os.makedirs(cfg.results_dir+'/logs/', exist_ok=True)
    os.makedirs(cfg.results_dir+'/confusion_matrix/', exist_ok=True)
    sys.stdout = Logger(cfg.results_dir+'/output.log')
    
    ##############
    # setup data #
    ##############
    x_train = np.load(os.path.join(args.data_path, args.dataset, 'x_train.npy'))
    x_valid = np.load(os.path.join(args.data_path, args.dataset, 'x_valid.npy'))
    x_test  = np.load(os.path.join(args.data_path, args.dataset, 'x_test.npy'))
    y_train = np.load(os.path.join(args.data_path, args.dataset, 'y_train.npy'))
    y_valid = np.load(os.path.join(args.data_path, args.dataset, 'y_valid.npy'))
    y_test  = np.load(os.path.join(args.data_path, args.dataset, 'y_test.npy'))
    w_train = get_class_weights(y_train)
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0., std=1., inplace=True)])
    train_dataset = NormalDataset(x_train, y_train, name="train", transform=transform)
    valid_dataset = NormalDataset(x_valid, y_valid, name="val", transform=transform)
    test_dataset  = NormalDataset(x_test, y_test, name="test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    
    ###############
    # setup model #
    ###############
    checkpoint = torch.load(cfg.ckpt, map_location=lambda storage, loc: storage)
    ckpt_args = checkpoint["args"] if "args" in checkpoint else None

    sit_state_dict = {}
    for k, v in checkpoint['ema'].items():
        if 'transformer' in k:
            k_ = k.replace('transformer.', '')
            sit_state_dict[k_] = v

    model = sits.__dict__[ckpt_args.model](
        input_size=ckpt_args.window_size,
        patch_size=ckpt_args.patch_size,
        num_classes=cfg.num_classes,
    ).to(device, cfg.dtype)
    model.load_state_dict(sit_state_dict, strict=False)
    
    if cfg.transfer:
        for n, p in model.named_parameters():
            if not 'head' in n:
                p.requires_grad = False
    
    
    n, accs, f1s, = 5, [], []
    for i in range(n):
        #######
        # run #
        #######
        model_ = copy.deepcopy(model)
        log, best = train_model(model_, train_loader, valid_loader, w_train, cfg, device)
        torch.save(best['model'].state_dict(), cfg.results_dir+'/bests/'+str(i)+'.pt')
        ##########
        # figure #
        ##########
        import matplotlib.pyplot as plt
        plt = plot_fig(log)
        plt.savefig(cfg.results_dir+'/logs/'+str(i)+'.png')
        #############
        # test eval #
        #############
        import matplotlib.pyplot as plt
        loss, acc, _, preds, trues = evaluate_model(best['model'], test_loader, nn.CrossEntropyLoss(), device=device)
        acc, rec, pre, f1, plt = test_evaluation(preds, trues, best['epoch'])
        plt.savefig(cfg.results_dir+'/confusion_matrix/'+str(i)+'.png')
        accs.append(acc)
        f1s.append(f1)
    
    print('**************')
    print('    result    ')
    print('**************')
    print('acc: {:.4f}±{:.4f}'.format(np.mean(accs), np.std(accs)))
    print('f1 : {:.4f}±{:.4f}'.format(np.mean(f1s), np.std(f1s)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--data-path", type=str, default='../signal_data/downstream')
    parser.add_argument("--results-dir", type=str, default='./experiment_log/downstream')
    parser.add_argument("--dataset", type=str, choices=['adl', 'oppo', 'pamap', 'realworld', 'wisdm'])
    parser.add_argument("--window-size", type=int, default=300)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--num-classes", type=int) # adl:5, oppo:4, pamap:8, realworld:8, wisdm:18
    parser.add_argument("--transfer", type=bool_flag, default=False) # True: transfer learning, False: fine-tuning
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dtype", type=torch.dtype, default=torch.float)
    parser.set_defaults(
        dataset='pamap',
        ckpt='./experiment_log/downstream/<dataset>/diffusion/<pre-trained model>',
    )
    args = parser.parse_args()
    args.num_classes = {'adl':5, 'oppo':4, 'pamap':8, 'realworld':8, 'wisdm':18}[args.dataset]
    main(args)
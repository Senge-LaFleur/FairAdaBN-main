"""
Baseline model: ResNet-152 trained with CrossEntropyLoss only (no fairness constraint).
This is the 'b' (baseline) model used in the FATE metric:

    FATE = (ACC_m - ACC_b) / ACC_b  -  lambda * (FC_m - FC_b) / FC_b

Train this first, then train FairAdaBN (main.py) or FairAdaBN+Lmi (main_Lmi.py) as the
mitigation model 'm', then run fate_eval.py to compute FATE.

Usage:
    python main_baseline.py --exp_id 1 --rand_seed 0 --max_epoch 400
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from basemodels import cusResNet152
from Fitz17k import Fitz17k
from model import resnet152
from rich.console import Console
import time
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms
from utils import *


def train(net, criterion, train_loader, valid_loader, optimizer, max_epoch, valid_interval=1):
    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []
    best_acc = 0

    n_epochs   = max_epoch
    n_train    = len(train_loader)
    n_valid    = len(valid_loader)

    for iteration in range(n_epochs):
        t0 = time.time()

        # ── train ─────────────────────────────────────────────────────────────
        epoch_loss = AverageMeter()
        epoch_acc  = AverageMeter()
        net.train()

        for batch_idx, (_, image, label, sensitive_attribute) in enumerate(train_loader):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            image, label = image.to(device), label.to(device)
            label = label.squeeze(dim=1).long()

            output, _ = net(image, task_idx=0)
            loss = criterion(output, label)
            preds = output.argmax(dim=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.update(loss.item())
            epoch_acc.update((sum(preds == label) / label.shape[0]).item())

            # single in-place progress line — overwrites itself each batch
            print(f'\r  epoch progress [{iteration+1}/{n_epochs}]  '
                  f'train [{batch_idx+1}/{n_train}]', end='', flush=True)

        train_losses.append(epoch_loss.avg)
        train_accs.append(epoch_acc.avg)
        scheduler.step()

        # ── validate ──────────────────────────────────────────────────────────
        v_acc_str = v_loss_str = 'N/A'
        if iteration % valid_interval == 0:
            epoch_loss = AverageMeter()
            epoch_acc  = AverageMeter()
            net.eval()

            for batch_idx, (_, image, label, sensitive_attribute) in enumerate(valid_loader):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                image, label = image.to(device), label.to(device)
                label = label.squeeze(dim=1).long()

                print(f'\r  epoch progress [{iteration+1}/{n_epochs}]  '
                      f'valid [{batch_idx+1}/{n_valid}]', end='', flush=True)

                with torch.no_grad():
                    output, _ = net(image, task_idx=0)
                    loss  = criterion(output, label)
                    preds = output.argmax(dim=1)

                epoch_loss.update(loss.item())
                epoch_acc.update((sum(preds == label) / label.shape[0]).item())

            valid_losses.append(epoch_loss.avg)
            valid_accs.append(epoch_acc.avg)
            v_acc_str  = f'{epoch_acc.avg:.4f}'
            v_loss_str = f'{epoch_loss.avg:.4f}'

            if epoch_acc.avg > best_acc:
                best_acc = epoch_acc.avg
                save_best_model(net, args.exp_id, args.rand_seed)

        elapsed = time.time() - t0
        # ── single summary line per epoch ─────────────────────────────────────
        print(f'\repoch [{iteration+1:>4}/{n_epochs}]  '
              f'train acc={train_accs[-1]:.4f}  train loss={train_losses[-1]:.4f}  '
              f'valid acc={v_acc_str}  valid loss={v_loss_str}  '
              f'({elapsed:.0f}s)')

    return train_accs, train_losses, valid_accs, valid_losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline training (no fairness constraint)')
    parser.add_argument('--exp_id',    default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr',        default=1e-4)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--rand_seed', type=str, default='0')

    args = parser.parse_args()

    console = Console()
    console.log('BASELINE model (no fairness) — backbone resnet-152')
    console.log(args)

    # ── data ──────────────────────────────────────────────────────────────────
    ann_train = pd.read_csv(
        '/kaggle/working/FairAdaBN-main/dataset/Fitzpatrick-17k/processed/rand_seed={}/split/train.csv'.format(args.rand_seed),
        index_col=0)
    ann_valid = pd.read_csv(
        '/kaggle/working/FairAdaBN-main/dataset/Fitzpatrick-17k/processed/rand_seed={}/split/val.csv'.format(args.rand_seed),
        index_col=0)
    ann_train.reset_index(inplace=True)
    ann_valid.reset_index(inplace=True)

    train_pkl = r'/kaggle/input/datasets/njihsenge/fitzpatrick17k/fitzpatrick17k/data/finalfitz17k'
    valid_pkl = r'/kaggle/input/datasets/njihsenge/fitzpatrick17k/fitzpatrick17k/data/finalfitz17k'

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(128, scale=(0.4, 1), ratio=(3 / 4, 4 / 3)),
        transforms.ToTensor(),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_set = Fitz17k(dataframe=ann_train, path_to_pickles=train_pkl,
                        sens_name='skintone', sens_classes=2, transform=transform_train)
    weights = train_set.get_weights(resample_which='group')
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    train_loader = data.DataLoader(train_set, batch_size=args.batch_size,
                                   sampler=sampler, num_workers=4,
                                   pin_memory=True, drop_last=True)

    valid_set = Fitz17k(dataframe=ann_valid, path_to_pickles=valid_pkl,
                        sens_name='skintone', sens_classes=2, transform=transform_val)
    valid_loader = data.DataLoader(valid_set, batch_size=args.batch_size,
                                   shuffle=False, num_workers=4,
                                   pin_memory=True, drop_last=True)

    # ── output dirs ───────────────────────────────────────────────────────────
    for d in ['weights', 'logs', 'preds', 'fair_scores']:
        os.makedirs(d, exist_ok=True)

    # ── model ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    net = resnet152(num_classes=9, select_pos=False)
    load_weights(net)
    net = net.to(device)

    optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     T_max=args.max_epoch,
                                                     eta_min=0)

    # ── train ─────────────────────────────────────────────────────────────────
    t_acc, t_loss, v_acc, v_loss = train(net, criterion, train_loader, valid_loader,
                                         optimizer, args.max_epoch, valid_interval=5)
    save_final_model(net, args.exp_id, args.rand_seed)
    save_logs(t_acc, t_loss, v_acc, v_loss, args.exp_id, args.rand_seed)

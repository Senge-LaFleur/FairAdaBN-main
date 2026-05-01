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
from rich.progress import (BarColumn, Progress, TextColumn, TimeElapsedColumn,
                           TimeRemainingColumn)
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms
from utils import *


def train(net, criterion, train_loader, valid_loader, optimizer, max_epoch, valid_interval=10):
    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []
    best_acc = 0

    with Progress(TextColumn("[progress.description]{task.description}"),
                  BarColumn(),
                  TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                  TimeRemainingColumn(),
                  TimeElapsedColumn(), console=console) as progress:
        epoch_tqdm = progress.add_task(description="epoch progress", total=max_epoch)
        train_tqdm = progress.add_task(description="train progress", total=len(train_loader))
        valid_tqdm = progress.add_task(description="valid progress", total=len(valid_loader))

        for iteration in range(max_epoch):
            console.rule('epoch {}'.format(iteration))

            # ── train ─────────────────────────────────────────────────────────
            epoch_loss = AverageMeter()
            epoch_acc  = AverageMeter()
            net.train()

            for _, image, label, sensitive_attribute in train_loader:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                image, label = image.to(device), label.to(device)

                label = label.squeeze(dim=1).long()

                # Single forward pass — no fairness split, no SPD loss
                output, _ = net(image, task_idx=0)
                loss = criterion(output, label)
                preds = output.argmax(dim=1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss.update(loss.item())
                epoch_acc.update((sum(preds == label) / label.shape[0]).item())
                progress.advance(train_tqdm, advance=1)

            console.log('train acc = {:.4f}\ttrain loss = {:.4f}'.format(
                epoch_acc.avg, epoch_loss.avg))
            train_losses.append(epoch_loss.avg)
            train_accs.append(epoch_acc.avg)
            progress.reset(train_tqdm)

            scheduler.step()

            # ── validate ──────────────────────────────────────────────────────
            if iteration % valid_interval == 0:
                epoch_loss = AverageMeter()
                epoch_acc  = AverageMeter()
                net.eval()

                for _, image, label, sensitive_attribute in valid_loader:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    image, label = image.to(device), label.to(device)
                    label = label.squeeze(dim=1).long()

                    with torch.no_grad():
                        output, _ = net(image, task_idx=0)
                        loss  = criterion(output, label)
                        preds = output.argmax(dim=1)

                    epoch_loss.update(loss.item())
                    epoch_acc.update((sum(preds == label) / label.shape[0]).item())
                    progress.advance(valid_tqdm, advance=1)

                console.log('valid acc = {:.4f}\tvalid loss = {:.4f}'.format(
                    epoch_acc.avg, epoch_loss.avg))
                valid_losses.append(epoch_loss.avg)
                valid_accs.append(epoch_acc.avg)

                if epoch_acc.avg > best_acc:
                    best_acc = epoch_acc.avg
                    console.log('save model with acc: {:.4f}'.format(best_acc))
                    save_best_model(net, args.exp_id, args.rand_seed)

                progress.reset(valid_tqdm)

            progress.advance(epoch_tqdm, advance=1)

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
        './dataset/Fitzpatrick-17k/processed/rand_seed={}/split/train.csv'.format(args.rand_seed),
        index_col=0)
    ann_valid = pd.read_csv(
        './dataset/Fitzpatrick-17k/processed/rand_seed={}/split/val.csv'.format(args.rand_seed),
        index_col=0)
    ann_train.reset_index(inplace=True)
    ann_valid.reset_index(inplace=True)

    train_pkl = r'..\datasets\fitzpatrick17k\data\finalfitz17k'
    valid_pkl = r'..\datasets\fitzpatrick17k\data\finalfitz17k'

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

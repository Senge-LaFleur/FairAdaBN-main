import argparse

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


class SPD_Loss(nn.Module):
    def __init__(self) -> None:
        super(SPD_Loss, self).__init__()
    
    def forward(self, preds, attrs):
        
        spd = 0
        predictions = torch.zeros(size=(9,2))
        
        for i in range(len(preds)):
            predictions[preds[i]][attrs[i]] += 1
        
        n_attr_1 = torch.nonzero(attrs).shape[0]
        n_attr_0 = attrs.shape[0] - n_attr_1
        
        for i in range(9):
            spd += torch.pow(input=(predictions[i][0] / n_attr_0 - predictions[i][1] / n_attr_1), exponent=2)
        
        return spd

def mi_loss(z_c: torch.Tensor, z_d: torch.Tensor) -> torch.Tensor:
    cos_sim = torch.nn.functional.cosine_similarity(z_c, z_d, dim=-1)
    return (1.0 - cos_sim).mean()

class Fitz17k_Lmi(Fitz17k):
    def __init__(self, dataframe, path_to_pickles, sens_name, sens_classes, transform_1, transform_2):
        super(Fitz17k_Lmi, self).__init__(dataframe, path_to_pickles, sens_name, sens_classes, transform_1)
        self.transform_2 = transform_2
        
    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        import os
        from PIL import Image
        img_id = str(item['md5hash'])
        img_path = os.path.join(self.path_to_images, img_id + '.jpg')
        if not os.path.exists(img_path):
            img_path = os.path.join(self.path_to_images, img_id)
        img = Image.open(img_path).convert('RGB')
        
        img1 = self.transform(img)
        img2 = self.transform_2(img)

        label = torch.FloatTensor([self.Y[idx]])
        sensitive = self.get_sensitive(self.sens_name, self.sens_classes, item)
                               
        return idx, img1, img2, label, sensitive

def train(net, criterion, train_loader, valid_loader, optimizer, max_epoch, args, valid_interval=1):
    # inits
    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []
    best_acc = 0
    spd = SPD_Loss()
  
    n_epochs = max_epoch
    n_train  = len(train_loader)
    n_valid  = len(valid_loader)

    for iteration in range(n_epochs):
        t0 = time.time()

        # train
        epoch_loss     = AverageMeter()
        epoch_acc      = AverageMeter()
        epoch_loss_ce  = AverageMeter()
        epoch_loss_spd = AverageMeter()
        epoch_loss_mi  = AverageMeter()
        net.train()
        for batch_idx, (_, image, image_aug, label, sensitive_attribute) in enumerate(train_loader):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            image, image_aug, label, sensitive_attribute = image.to(device), image_aug.to(device), label.to(device), sensitive_attribute.to(device)

            task_0_idx = (sensitive_attribute == 0).nonzero(as_tuple=True)
            task_1_idx = (sensitive_attribute == 1).nonzero(as_tuple=True)

            image_0, image_aug_0, label_0, sensitive_attribute_0 = image[task_0_idx], image_aug[task_0_idx], label[task_0_idx], sensitive_attribute[task_0_idx]
            image_1, image_aug_1, label_1, sensitive_attribute_1 = image[task_1_idx], image_aug[task_1_idx], label[task_1_idx], sensitive_attribute[task_1_idx]

            label_0, label_1 = label_0.squeeze(dim=1), label_1.squeeze(dim=1)

            output_0, feature_0 = net(image_0, task_idx=0)
            loss_0 = criterion(output_0, label_0.long())
            preds_0 = output_0.argmax(dim=1)

            if image_0.shape[0] != 0:
                _, feature_aug_0 = net(image_aug_0, task_idx=0)
                loss_mi_0 = mi_loss(feature_0, feature_aug_0)
            else:
                loss_mi_0 = 0

            output_1, feature_1 = net(image_1, task_idx=1)
            loss_1 = criterion(output_1, label_1.long())
            preds_1 = output_1.argmax(dim=1)

            if image_1.shape[0] != 0:
                _, feature_aug_1 = net(image_aug_1, task_idx=1)
                loss_mi_1 = mi_loss(feature_1, feature_aug_1)
            else:
                loss_mi_1 = 0

            loss_spd = spd(torch.cat([preds_0, preds_1]), torch.cat([sensitive_attribute_0, sensitive_attribute_1]))
            loss_mi = loss_mi_0 + loss_mi_1
            loss = loss_0 + loss_1 + loss_spd + args.lambda_mi * loss_mi

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.update(loss.item())
            epoch_loss_ce.update(loss_0.item() + loss_1.item())
            epoch_loss_spd.update(loss_spd.item())
            epoch_loss_mi.update(loss_mi.item() if isinstance(loss_mi, torch.Tensor) else loss_mi)
            if label_0.shape[0] != 0:
                epoch_acc.update((sum(preds_0 == label_0.squeeze()) / label_0.shape[0]).item())
            if label_1.shape[0] != 0:
                epoch_acc.update((sum(preds_1 == label_1.squeeze()) / label_1.shape[0]).item())

            print(f'\r  epoch [{iteration+1}/{n_epochs}]  train [{batch_idx+1}/{n_train}]',
                  end='', flush=True)

        train_losses.append(epoch_loss.avg)
        train_accs.append(epoch_acc.avg)
        scheduler.step()

        # valid
        v_acc_str = v_loss_str = 'N/A'
        if iteration % valid_interval == 0:
            epoch_loss = AverageMeter()
            epoch_acc  = AverageMeter()
            net.eval()
            for batch_idx, (_, image, label, sensitive_attribute) in enumerate(valid_loader):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                image, label, sensitive_attribute = image.to(device), label.to(device), sensitive_attribute.to(device)

                task_0_idx = (sensitive_attribute == 0).nonzero(as_tuple=True)
                task_1_idx = (sensitive_attribute == 1).nonzero(as_tuple=True)

                image_0, label_0 = image[task_0_idx], label[task_0_idx]
                image_1, label_1 = image[task_1_idx], label[task_1_idx]

                label_0, label_1 = label_0.squeeze(dim=1), label_1.squeeze(dim=1)

                print(f'\r  epoch [{iteration+1}/{n_epochs}]  valid [{batch_idx+1}/{n_valid}]',
                      end='', flush=True)

                with torch.no_grad():
                    output, _ = net(image_0, task_idx=0)
                    loss_0 = criterion(output, label_0.long())
                    preds_0 = output.argmax(dim=1)

                    output, _ = net(image_1, task_idx=1)
                    loss_1 = criterion(output, label_1.long())
                    preds_1 = output.argmax(dim=1)

                loss = loss_0 + loss_1
                epoch_loss.update(loss.item())
                epoch_acc.update((sum(preds_0 == label_0.squeeze()) / label_0.shape[0]).item())
                if label_1.shape[0] != 0:
                    epoch_acc.update((sum(preds_1 == label_1.squeeze()) / label_1.shape[0]).item())

            valid_losses.append(epoch_loss.avg)
            valid_accs.append(epoch_acc.avg)
            v_acc_str  = f'{epoch_acc.avg:.4f}'
            v_loss_str = f'{epoch_loss.avg:.4f}'

            if epoch_acc.avg > best_acc:
                best_acc = epoch_acc.avg
                save_best_model(net, args.exp_id, args.rand_seed)

        elapsed = time.time() - t0
        print(f'\repoch [{iteration+1:>4}/{n_epochs}]  '
              f'train acc={train_accs[-1]:.4f}  loss={train_losses[-1]:.4f}  '
              f'mi={epoch_loss_mi.avg:.4f}  '
              f'valid acc={v_acc_str}  valid loss={v_loss_str}  '
              f'({elapsed:.0f}s)')

    return train_accs, train_losses, valid_accs, valid_losses
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train settings')
    parser.add_argument('--exp_id', default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--rand_seed', type=str, default=0)
    parser.add_argument('--lambda_mi', type=float, default=1.0)
        
    args = parser.parse_args()
    
    # logs
    console = Console()
    console.log('baseline model with backbone resnet-152')
    console.log(args)
    
    # paths
    ann_train = pd.read_csv('/kaggle/working/FairAdaBN-main/dataset/Fitzpatrick-17k/processed/rand_seed={}/split/train.csv'.format(args.rand_seed), index_col=0)
    ann_valid = pd.read_csv('/kaggle/working/FairAdaBN-main/dataset/Fitzpatrick-17k/processed/rand_seed={}/split/val.csv'.format(args.rand_seed), index_col=0)
    ann_train.reset_index(inplace=True)
    ann_valid.reset_index(inplace=True)
    
    train_pkl = r'/kaggle/input/datasets/njihsenge/fitzpatrick17k/fitzpatrick17k/data/finalfitz17k'
    valid_pkl = r'/kaggle/input/datasets/njihsenge/fitzpatrick17k/fitzpatrick17k/data/finalfitz17k'


    # standard transform
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # augmentation transform
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(128, scale=(0.4, 1),
                                     ratio=(3 / 4, 4 / 3)),
        transforms.ToTensor(),
    ])

    transform_train_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(128, scale=(0.4, 1),
                                     ratio=(3 / 4, 4 / 3)),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ])
    
    # create train dataset
    train_set = Fitz17k_Lmi(dataframe=ann_train, path_to_pickles=train_pkl, sens_name='skintone', sens_classes=2, transform_1=transform_train, transform_2=transform_train_aug)
    weights = train_set.get_weights(resample_which='group')
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    train_loader = data.DataLoader(train_set,
                                    batch_size=args.batch_size,
                                    sampler=sampler,
                                    num_workers=4,
                                    pin_memory=True,
                                    drop_last=True)
    # create validation dataset
    valid_set = Fitz17k(dataframe=ann_valid, path_to_pickles=valid_pkl, sens_name='skintone', sens_classes=2, transform=transform)
    valid_loader = data.DataLoader(valid_set,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=4,
                                     pin_memory=True,
                                     drop_last=True)

    # Ensure output directories exist
    import os
    for d in ['weights', 'logs', 'preds', 'fair_scores']:
        os.makedirs(d, exist_ok=True)

    # GPU setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    criterion = nn.CrossEntropyLoss()
    net = resnet152(num_classes=9, select_pos=False)
    load_weights(net)
    net = net.to(device)
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9,0.999), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     T_max=args.max_epoch,
                                                     eta_min=0)
    # train
    t_acc, t_loss, v_acc, v_loss = train(net, criterion, train_loader, valid_loader, optimizer, args.max_epoch, args, valid_interval=1)
    save_final_model(net, args.exp_id, args.rand_seed)
    save_logs(t_acc, t_loss, v_acc, v_loss, args.exp_id, args.rand_seed)
    
    

    
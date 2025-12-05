# This is a sample Python script.

import torch
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from dataset import MVTecDataset
import torch.backends.cudnn as cudnn
import argparse
from test import evaluation, visualization, test
from torch.nn import functional as F
import csv

def init_csv_logger(class_name, test_id):
    log_dir = f'./logs/{test_id}'
    os.makedirs(log_dir, exist_ok=True)
    csv_path = f'{log_dir}/{class_name}.csv'
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Loss', 'Pixel_AUROC', 'Sample_AUROC', 'Pixel_AUPRO'])
    return csv_path

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def loss_fucntion(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                        b[item].view(b[item].shape[0], -1)))
    return loss

def loss_concat(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):
        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))
    a_map = torch.cat(a_map, 1)
    b_map = torch.cat(b_map, 1)
    loss += torch.mean(1 - cos_loss(a_map, b_map))
    return loss

def train(_class_, resume=False, test_id=1):
    print(f"\n🔧 Training class: {_class_}")
    epochs = 200
    learning_rate = 0.005
    batch_size = 16
    image_size = 256

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    csv_path = init_csv_logger(_class_, test_id)

    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    train_path = './mvtec/' + _class_ + '/train'
    test_path = './mvtec/' + _class_
    os.makedirs(f'./checkpoints/{test_id}', exist_ok=True)
    # os.makedirs('./checkpoints', exist_ok=True)
    # ckp_path = './checkpoints/' + 'wres50_'+_class_+'.pth'
    ckp_path = f'./checkpoints/{test_id}/wres50_{_class_}.pth'

    train_data = ImageFolder(root=train_path, transform=data_transform)
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=learning_rate, betas=(0.5, 0.999))

    print(f"Trainable parameters in bottleneck module (bn): {count_parameters(bn):,}")
    print(f"Trainable parameters in decoder module: {count_parameters(decoder):,}")

    start_epoch = 0
    if resume and os.path.exists(ckp_path):
        print(f"🔁 Resuming from checkpoint: {ckp_path}")
        checkpoint = torch.load(ckp_path)
        decoder.load_state_dict(checkpoint['decoder'])
        bn.load_state_dict(checkpoint['bn'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    else:
        print("🚀 Starting training from scratch")

    for epoch in range(start_epoch, epochs):
        bn.train()
        decoder.train()
        loss_list = []
        for img, label in train_dataloader:
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            loss = loss_fucntion(inputs, outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))

        if (epoch + 1) % 10 == 0:
            auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder, test_dataloader, device)
            print('Pixel Auroc:{:.3f}, Sample Auroc{:.3f}, Pixel Aupro{:.3}'.format(auroc_px, auroc_sp, aupro_px))
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, round(np.mean(loss_list), 4), auroc_px, auroc_sp, aupro_px])

            torch.save({
                'epoch': epoch + 1,
                'bn': bn.state_dict(),
                'decoder': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, ckp_path)

    return auroc_px, auroc_sp, aupro_px

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--class', dest='target_class', type=str, default=None,
                        help='Specify a single class to train, e.g. --class carpet')
    parser.add_argument('--test', dest='test_id', type=int, default=1, help='Specify test run ID')
    args = parser.parse_args()

    setup_seed(111)

    if args.target_class:
        train(args.target_class, resume=args.resume, test_id=args.test_id)
        visualization(args.target_class, test_id=args.test_id)
    else:
        item_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                     'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
        for i in item_list:
            train(i, resume=args.resume, test_id=args.test_id)
            visualization(i, test_id=args.test_id)
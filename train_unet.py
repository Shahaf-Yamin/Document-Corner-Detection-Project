import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading import DocumentDataset, FullResolutionCroppedDocumentDataset, FullResolutionResizedDocumentDataset, PartialDocumentDatasetMaskSegmentation
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet
from matplotlib import pyplot as plt
dir_img = Path('data/CNN_data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = None


def train_net(net,
              device,
              train_set,
              val_set,
              experiment,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 0.001,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False,
              patience: int = 3):

    # 1. Create data loaders
    # loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    loader_args = dict(batch_size=batch_size, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)


    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))
    n_train = len(train_set)
    n_val = len(val_set)
    logging.info(f'''Starting training:
        Experiment Name: {args.exp_name}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp} 
        Patience: {patience}''')

    # 2. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=patience, factor=0.5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # weights_dict = dataset.imbalanceRatioDict
    # We give more weights to the class which has less possiblity to be selected using the following rule:
    # weights = torch.FloatTensor([1 - weights_dict[class_idx] for class_idx in range(2)]).to(device)
    # criterion = nn.CrossEntropyLoss(weight=weights)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    ce_loss = criterion(masks_pred, true_masks)
                    curr_dice_loss = dice_loss(F.softmax(masks_pred, dim=1).float(),
                                       F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)
                    loss = ce_loss + curr_dice_loss

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({'train loss': loss.item(), 'step': global_step, 'epoch': epoch})
                experiment.log({'Dice loss': curr_dice_loss.item(), 'step': global_step, 'epoch': epoch})
                experiment.log({'Dice Score': 1 - curr_dice_loss.item(), 'step': global_step, 'epoch': epoch})
                experiment.log({'Cross Entropy loss': ce_loss.item(), 'step': global_step, 'epoch': epoch})

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(net, val_loader, device)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--exp_name', type=str, default='', help='Experiment Name of the current run')
    parser.add_argument('--filter_size', type=int, default=5, help='Size of Guassian Filter Kernel')
    parser.add_argument('--image_size', type=int, default=512, help='Size of Resized Image')
    parser.add_argument('--patience', type=int, default=3, help='How many episodes to wait before reducing the learning rate')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    exp_name = args.exp_name + f'patience_{args.patience}_lr_{args.lr}_filter_size_{args.filter_size}'
    experiment = wandb.init(project="dl_course_project", entity="shahaf_yamin", resume='allow', name=exp_name, config=args)
    args = experiment.config

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=3, n_classes=2, bilinear=True)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    # 1. Create dataset
    SIZE = args.image_size
    FILTER_SIZE = args.filter_size
    logging.info(f'Size {SIZE}')
    logging.info(f'Filter Size {FILTER_SIZE}')
    train_set = PartialDocumentDatasetMaskSegmentation(Path='data/Train', Transforms={'GaussianBlur': FILTER_SIZE, 'RandomCropNearCorner': 64, 'Normalize': None}, Size=SIZE)
    val_set = PartialDocumentDatasetMaskSegmentation(Path='data/Validation', Transforms={'GaussianBlur': FILTER_SIZE,  'RandomCropNearCorner': 64, 'Normalize': (train_set.stats['mean'], train_set.stats['std'])}, Size=SIZE)
    # 2. Create Check Points directory
    dir_checkpoint = Path(f'./checkpoints/{args.exp_name}/')
    logging.info('This Version')
    try:
        train_net(net=net,
                  train_set=train_set,
                  val_set = val_set,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp,
                  experiment=experiment,
                  patience=args.patience)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
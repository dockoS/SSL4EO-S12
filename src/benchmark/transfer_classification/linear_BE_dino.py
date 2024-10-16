# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import json
from pathlib import Path
import sys
import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

from models.dino import utils
from models.dino import vision_transformer as vits

# load bigearthnet dataset
from datasets.BigEarthNet.bigearthnet_dataset_seco import Bigearthnet
from datasets.BigEarthNet.bigearthnet_dataset_seco_lmdb_s2_uint8 import LMDBDataset,random_subset

### end of change ###
import pdb
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import average_precision_score
import builtins
torch.device('cpu')
import torch.optim as optim

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import v2
import segmentation_models_pytorch as smp
from torchmetrics.classification import F1Score, MulticlassJaccardIndex
MEANS= {'B1': 2175.7995533938183, 'B2': 2036.2739530256324, 'B3': 2100.8864073786062, 'B4': 2259.670904983591, 'B5': 2401.4535482508154, 'B6': 3239.511063914319, 'B7': 3804.818377507816, 'B8': 3480.9846790712386, 'B8A': 4136.2014152288275, 'B9': 467.6799662569083, 'B10': 16.239072759978043, 'B11': 3768.791296569817, 'B12': 2555.6047187792947}
STD= {'B1': 443.72058924036867, 'B2': 488.43510998754664, 'B3': 486.88830792486743, 'B4': 571.6584223323891, 'B5': 499.5042846572638, 'B6': 459.6175067406037, 'B7': 498.29341083169993, 'B8': 443.65053936363404, 'B8A': 520.1053576865286, 'B9': 79.93659155518765, 'B10': 1.7570587120223873, 'B11': 704.6183236092784, 'B12': 666.5783354478218}

class NumpyDataset(Dataset):
    def __init__(self, input_dir, label_dir, transform=None):
        self.input_files = sorted(os.listdir(input_dir))
        self.label_files = sorted(os.listdir(label_dir))
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.transform = transform
        self.label_mapping = {0: -1, 1: 15, 2: 16, 3: 17, 4: 18, 5: 19, 6: 20, 7: 21, 8: 0, 9: 1, 10: 2,11:3,12:4,13:5,14:6,15:7,16:8,17:9,18:10,19:11,20:12,21:13,22:14}

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        input_data = np.load(input_path).astype(np.float32)
        label_data = np.load(label_path)

        if self.transform:
            input_data = torch.from_numpy(input_data)
      
            input_data = self.transform(input_data)
        label_data = np.vectorize(lambda x: self.label_mapping.get(x, -1))(label_data)
        
        return input_data, label_data[0]
def calculate_num_classes(label_dir):
    all_labels = []
    for file in os.listdir(label_dir):
        label_data = np.load(os.path.join(label_dir, file)).astype(np.float32)
        all_labels.append(label_data)
    
    all_labels = np.concatenate(all_labels, axis=None)  # Concaténer tous les labels
    unique_labels = np.unique(all_labels)
    num_classes = len([label for label in unique_labels if label in full_dataset.label_mapping])  # Ne compter que les classes valides

    return [label for label in unique_labels if label in full_dataset.label_mapping],num_classes

# Calculer la moyenne et l'ecart-type pour chaque canal
def calculate_mean_std(input_dir):
    all_data = []
    for file in os.listdir(input_dir):
        data = np.load(os.path.join(input_dir, file))
        all_data.append(data)
    
    all_data = np.concatenate(all_data, axis=0)  # Concaténer sur la dimension des exemples
    mean = np.mean(all_data, axis=(0, 1, 2))  # Moyenne sur les dimensions spatiales et des exemples
    std = np.std(all_data, axis=(0, 1, 2))    # Ecart-type sur les dimensions spatiales et des exemples
    
    return mean, std

# Dossiers des données
input_dir = '/Users/mac/Desktop/docko/tolbi-next-gen/preprocessing-geospatial-data/data/cartagraphie_des_cultures_dataset_2024-07-01_2024-12-01_datasetS2/S2/chips'
label_dir = '/Users/mac/Desktop/docko/tolbi-next-gen/preprocessing-geospatial-data/data/cartagraphie_des_cultures_dataset_2024-07-01_2024-12-01_datasetS2/labels'

# Calcul de la moyenne et de l'écart-type
# mean, std = calculate_mean_std(input_dir)
# print(f"Mean: {mean}, Std: {std}")

# Transformation pour normaliser les données

transform=v2.Compose(
            [
                v2.Normalize(mean=list(MEANS.values()), std=list(STD.values())),
            ]
            ,)


# Créer le dataset
full_dataset = NumpyDataset(input_dir=input_dir, label_dir=label_dir, transform=transform)

# Diviser en train et validation (80% train, 20% validation)
train_size = int(0.99 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Créer les DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

def eval_linear(args):
    # #utils.init_distributed_mode(args)
    # if args.rank != 0:
    #     def print_pass(*args):
    #         pass
    #     builtins.print = print_pass
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    
    #cudnn.benchmark = True

    # ============ building network ... ============
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
  
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0, in_chans=13)
        embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():

        model = torchvision_models.__dict__[args.arch]()
        embed_dim = model.fc.weight.shape[1]
        model.conv1 = torch.nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Identity()
        model = nn.Sequential(*list(model.children())[:-2]) 
        #model.fc = torch.nn.Linear(2048,19)
    # if the network is a XCiT
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
        embed_dim = model.embed_dim
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)
    #model.cuda()
    model.eval()
    # load weights to evaluate
    utils.load_pretrained_weights(model, args.pretrained, args.checkpoint_key, args.arch, args.patch_size)

    num_classes=len(set(full_dataset.label_mapping.values()))-1
    seg_model=SegmentationClassifier(embed_dim,num_classes)
    # print(seg_model)
    # print(embed_dim)
    # print(f"Model {args.arch} built.")
    data=torch.rand((32,13,64,64))
    x=model(data)
    # mask=seg_model(x)
    print(x.shape)
    exit()


    # linear_classifier = LinearClassifier(embed_dim, num_labels=19)
    # linear_classifier = linear_classifier.cuda()
    # linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

    # ============ preparing data ... ============






    if args.evaluate:
        utils.load_pretrained_linear_weights(seg_model, args.arch, args.patch_size)
        test_stats = validate_network(val_loader, model,seg_model, args.n_last_blocks, args.avgpool_patchtokens)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return    
    
            
    


    # set optimizer
    optimizer = torch.optim.SGD(
        seg_model.parameters(),
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., # linear scaling rule
        momentum=0.9,
        weight_decay=0, # we do not apply weight decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.}
    if args.resume:
        utils.restart_from_checkpoint(
            os.path.join(args.checkpoints_dir, "checkpoint.pth.tar"),
            run_variables=to_restore,
            state_dict=seg_model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]

    
    if args.rank==0 and not os.path.isdir(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir,exist_ok=True)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Utiliser une valeur qui ne correspond à aucune classe pour ignorer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_fn = smp.losses.FocalLoss(mode="multiclass")
    iou = MulticlassJaccardIndex(
            num_classes=num_classes,
            average="weighted",
        )
    f1 = F1Score(
            task="multiclass",
            num_classes=num_classes,
            average="weighted",
        )
    for epoch in range(start_epoch, args.epochs):
        #train_loader.sampler.set_epoch(epoch)

        train_stats = train(model,seg_model, optimizer, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens,criterion,loss_fn,iou,f1)
        scheduler.step()

    #     log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
    #                  'epoch': epoch}
    #     if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
    #         test_stats = validate_network(val_loader, model, seg_model, args.n_last_blocks, args.avgpool_patchtokens)
    #         print(f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    #         best_acc = max(best_acc, test_stats["acc1"])
    #         print(f'Max accuracy so far: {best_acc:.2f}%')
    #         log_stats = {**{k: v for k, v in log_stats.items()},
    #                      **{f'test_{k}': v for k, v in test_stats.items()}}
    #     if utils.is_main_process():
    #         with (Path(args.checkpoints_dir) / "log.txt").open("a") as f:
    #             f.write(json.dumps(log_stats) + "\n")
    #         save_dict = {
    #             "epoch": epoch + 1,
    #             "state_dict": seg_model.state_dict(),
    #             "optimizer": optimizer.state_dict(),
    #             "scheduler": scheduler.state_dict(),
    #             "best_acc": best_acc,
    #         }
    #         torch.save(save_dict, os.path.join(args.checkpoints_dir, "checkpoint.pth.tar"))
    # print("Training of the supervised linear classifier on frozen features completed.\n"
    #             "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))

def train(model, linear_classifier, optimizer, loader, epoch, n, avgpool,criterion,loss_fn,iou,f1):
    linear_classifier.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    running_loss = 0.0
    running_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    for (images, target) in metric_logger.log_every(loader, 20, header):

        b_zeros = torch.zeros((images.shape[0],1,images.shape[2],images.shape[3]),dtype=torch.float32)
        # inp = torch.cat((images[:,:10,:,:],b_zeros,images[:,10:,:,:]),dim=1)
        # print(inp.shape)

        inp=images
        # move to gpu
        
        #inp = inp.cuda(non_blocking=True)
        #target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if avgpool:
                    output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                
                output = model(inp)
        output = linear_classifier(output)
        
        # compute cross entropy loss

        optimizer.zero_grad()
        target=target.long()
        loss = criterion(output, target)
        loss.backward()
  
        optimizer.step()
    
        # _, preds = torch.max(output, dim=1)
        # mask = target != -1  # Ignorer les pixels de fond
        # correct_pixels += torch.sum((preds == target) & mask).item()
        # total_pixels += torch.sum(mask).item()
   



        #print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

        # compute the gradients
        
        # log 
        
        #torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0
    print(f"Epoch [{epoch+1}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, linear_classifier, n, avgpool):
    linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    for images, target in metric_logger.log_every(val_loader, 20, header):

        b_zeros = torch.zeros((images.shape[0],1,images.shape[2],images.shape[3]),dtype=torch.float32)
        inp = torch.cat((images[:,:10,:,:],b_zeros,images[:,10:,:,:]),dim=1)        
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if avgpool:
                    output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(inp)
        output = linear_classifier(output)
        loss = nn.MultiLabelSoftMarginLoss()(output, target.long())

        '''
        if linear_classifier.module.num_labels >= 5:
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        else:
            acc1, = utils.accuracy(output, target, topk=(1,))
        '''
        score = torch.sigmoid(output).detach().cpu()
        acc1 = average_precision_score(target.cpu(), score, average='micro') * 100.0
        acc5 = acc1
        
        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        
        if linear_classifier.module.num_labels >= 5:
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        
    
    if linear_classifier.module.num_labels >= 5:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    else:
        print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)
    
class SegmentationClassifier(nn.Module):
    """Segmentation classifier layer to train on top of frozen features"""
    def __init__(self, dim, num_classes):
        super(SegmentationClassifier, self).__init__()
        self.num_classes = num_classes
        
        # A 1x1 convolutional layer to output the desired number of classes (like in linear classifier)
        self.conv = nn.Conv2d(dim, num_classes, kernel_size=1)
        
        # Initialize weights (similar to the LinearClassifier's initialization)
        self.conv.weight.data.normal_(mean=0.0, std=0.01)
        self.conv.bias.data.zero_()

    def forward(self, x):
        # Apply 1x1 convolution to map the features to the number of classes
        x = self.conv(x)
        
        # Optional: Upsample the output back to the original size
        x = F.interpolate(x, scale_factor=32, mode='bilinear', align_corners=True)
        
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on BigEarthNet.')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained', default='/Users/mac/Desktop/docko/tolbi-next-gen/SSL4EO-S12/data/B13_rn50_dino_0099_ckpt.pth', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=5, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--checkpoints_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    
    parser.add_argument('--lmdb_dir', default='/path/to/imagenet/', type=str, help='Please specify path to the ImageNet folder.')
    parser.add_argument('--bands', type=str, default='all', help="input bands")
    parser.add_argument("--lmdb", action='store_true', help="use lmdb dataset")
    parser.add_argument("--is_slurm_job", action='store_true', help="running in slurm")
    parser.add_argument("--resume", action='store_true', help="resume from checkpoint")
    parser.add_argument("--train_frac", default=1.0, type=float, help="use a subset of labeled data")
    parser.add_argument("--seed",default=42,type=int)
    
    args = parser.parse_args()


    eval_linear(args)

import argparse
import os
import random
import time

import apex
import cv2
import numpy as np
import pandas as pd
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from apex import amp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm import tqdm
from train import get_trans
from util import GradualWarmupSchedulerV2

from dataset import MelanomaDataset, get_df, get_transforms
from models import Effnet_Melanoma, Resnest_Melanoma, Seresnext_Melanoma


device = 'cpu'
args = {}
ModelClass = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel-type', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='/raid/')
    parser.add_argument('--data-folder', type=int, required=True)
    parser.add_argument('--image-size', type=int, required=True)
    parser.add_argument('--enet-type', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=50)
    parser.add_argument('--out-dim', type=int, default=8)
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--use-meta', action='store_true')
    parser.add_argument('--DEBUG', action='store_true')
    parser.add_argument('--model-dir', type=str, default='./test_weights')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--sub-dir', type=str, default='./subs')
    parser.add_argument('--eval', type=str, choices=['best', 'best_20', 'final'], default="best")
    parser.add_argument('--n-test', type=int, default=8)
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    parser.add_argument('--n-meta-dim', type=str, default='512,128')

    args, _ = parser.parse_known_args()
    return args


def main():

    # TODO: ask Sanna what was kernel type, again :c.
    # Because if I refactor the code, it's not needed anymore
    _, df_test, mel_idx = get_df(
        args.out_dim,
        args.data_dir,
        args.data_folder,
        args.use_meta
    )

    _, transforms_val = get_transforms(args.image_size)

    if args.DEBUG:
        df_test = df_test.sample(args.batch_size * 3)
    dataset_test = MelanomaDataset(df_test, 'test', transform=transforms_val)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.num_workers)

    # Load the models
    models = []
    for fold in range(1):
        if args.eval == 'best':
            model_file = os.path.join(args.model_dir, f'{args.kernel_type}_best_fold{fold}.pth')
        elif args.eval == 'best_20':
            model_file = os.path.join(args.model_dir, f'{args.kernel_type}_best_20_fold{fold}.pth')
        if args.eval == 'final':
            model_file = os.path.join(args.model_dir, f'{args.kernel_type}_final_fold{fold}.pth')

        # Creates and instance of the available model
        model = ModelClass(
            args.enet_type, # Ask Sanna if it's really needed in the case you use trainned models from pytorch
            out_dim=args.out_dim
        )
        model = model.to(device)

        try:  # single GPU model_file
            model.load_state_dict(torch.load(model_file), strict=True)
        except:  # multi GPU model_file
            state_dict = torch.load(model_file)
            state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
            model.load_state_dict(state_dict, strict=True)

        # Split the data into different GPU's if there is more than one available
        if len(os.environ['CUDA_VISIBLE_DEVICES']) > 1:
            model = torch.nn.DataParallel(model)

        model.eval()
        models.append(model)

    # predict
    PROBS = []
    with torch.no_grad(): # TODO: Might change to `torch.inference_mode()` decorator
        for (data) in tqdm(test_loader):
            # If use metadata is set to be used, destructure the data into image features and metadata
            if args.use_meta:
                data, meta = data
                data, meta = data.to(device), meta.to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                for model in models:
                    for I in range(args.n_test):
                        l = model(get_trans(data, I), meta)
                        probs += l.softmax(1)
            else:
                data = data.to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                for model in models:
                    for I in range(args.n_test):
                        l = model(get_trans(data, I))
                        probs += l.softmax(1)

            probs /= args.n_test
            probs /= len(models)

            PROBS.append(probs.detach().cpu())

    PROBS = torch.cat(PROBS).numpy()

    # save cvs
    #this is where you can get the labels
    df_test['target'] = PROBS[:, mel_idx]
    print(PROBS)
    PROBS=pd.DataFrame(PROBS)
    PROBS['image_name'] = df_test['image_name']
    # Export the probablities per each image to be malignat or the granulars benign
    PROBS.to_csv(os.path.join(args.sub_dir, f'probs_{args.kernel_type}_{args.eval}.csv'), index=False)
    # Export if a sample is cancer or not.
    df_test[['image_name', 'target']].to_csv(os.path.join(args.sub_dir, f'sub_{args.kernel_type}_{args.eval}.csv'), index=False)


def run(**kargs):
    global device
    global args
    global ModelClass

    args = kargs
    print(args)
    os.makedirs(args.sub_dir, exist_ok=True)

    if args.enet_type == 'resnest101':
        ModelClass = Resnest_Melanoma
    elif args.enet_type == 'seresnext101':
        ModelClass = Seresnext_Melanoma
    elif 'efficientnet' in args.enet_type:
        ModelClass = Effnet_Melanoma
    else:
        raise NotImplementedError()

    device = 'cuda' if torch.cuda_is_available() else torch.device('cuda')
    print(f'Single fold running on: {device}')





# if __name__ == '__main__':

#     args = parse_args()
#     os.makedirs(args.sub_dir, exist_ok=True)
#     os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES
#     if args.enet_type == 'resnest101':
#         ModelClass = Resnest_Melanoma
#     elif args.enet_type == 'seresnext101':
#         ModelClass = Seresnext_Melanoma
#     elif 'efficientnet' in args.enet_type:
#         ModelClass = Effnet_Melanoma
#     else:
#         raise NotImplementedError()

#     DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

#     device = torch.device('cuda')

#     main()

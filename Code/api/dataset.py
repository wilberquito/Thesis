from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# TODO: this repeated in NN and API module (take care of it!!)

class MelanomaDataset(Dataset):

    def __init__(self, csv, mode, transform=None):
        self.csv = csv.reset_index(drop=True)
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):

        # Get image path
        sample = self.csv.iloc[index]

        # Read image from path, transforming to tree channels, rgb
        image = cv2.imread(sample.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Transform the images using `albumentation`
        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        # Make color channel first
        image = image.transpose(2, 0, 1)
        data = torch.tensor(image).float()

        # If this is just for a test porpouse you can forget the label
        if self.mode == 'test':
            return data
        else:
            return data, torch.tensor(sample['target']).long()


# Get the dataframe to work with
def get_df(out_dim: int, data_dir: Path, data_folder: str):

    """Create a train an test dataframe that handles the train and test images"""

    # Supported output dimentions, meaning the classes to predict :)
    assert out_dim in [4, 8]

    # 2020 data
    train_path = data_dir / Path(f'jpeg-melanoma-{data_folder}x{data_folder}/train.csv')
    df_train = pd.read_csv(train_path)

    # Drops samples where tfrecord is -1
    df_train = df_train[df_train['tfrecord'] >= 0].reset_index(drop=True)
    df_train['fold'] = df_train['tfrecord']
    df_train['filepath'] = df_train['image_name'].apply(
        lambda img: train_path.parents[0] / Path(f'train/{img}.jpg'))
    df_train['is_ext'] = 0

    # 2018 and 2019 data
    train_path = data_dir / Path(f'jpeg-isic2019-{data_folder}x{data_folder}/train.csv')
    df_train2 = pd.read_csv(train_path)
    df_train2 = df_train2[df_train2['tfrecord'] >= 0].reset_index(drop=True)
    df_train2['filepath'] = df_train2['image_name'].apply(
        lambda img: train_path.parents[0] / Path(f'train/{img}.jpg'))
    df_train2['is_ext'] = 1

    # Preprocess Target
    df_train['diagnosis'] = df_train['diagnosis'].apply(
        lambda x: x.replace('seborrheic keratosis', 'BKL'))
    df_train['diagnosis'] = df_train['diagnosis'].apply(
        lambda x: x.replace('lichenoid keratosis', 'BKL'))
    df_train['diagnosis'] = df_train['diagnosis'].apply(
        lambda x: x.replace('solar lentigo', 'BKL'))
    df_train['diagnosis'] = df_train['diagnosis'].apply(
        lambda x: x.replace('lentigo NOS', 'BKL'))

    # Join categorical variables to adapt to the output of the nn
    if out_dim == 8:
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(
            lambda x: x.replace('NV', 'nevus'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(
            lambda x: x.replace('MEL', 'melanoma'))
    elif out_dim == 4:
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(
            lambda x: x.replace('NV', 'nevus'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(
            lambda x: x.replace('MEL', 'melanoma'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(
            lambda x: x.replace('DF', 'unknown'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(
            lambda x: x.replace('AK', 'unknown'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(
            lambda x: x.replace('SCC', 'unknown'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(
            lambda x: x.replace('VASC', 'unknown'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(
            lambda x: x.replace('BCC', 'unknown'))
    else:
        raise NotImplementedError()

    # Definition of the hole train dataframe
    df_train = pd.concat([df_train, df_train2]).reset_index(drop=True)

    # Create target column that represents the diagnostic using a number
    # Notice that target is malignant only and only if diagnosis is malignant,
    # Otherwise, is benign, in this case, diagnosis is more granular.
    diagnosis2idx = {d: idx for idx, d in enumerate(
        sorted(df_train.diagnosis.unique()))}
    df_train['target'] = df_train['diagnosis'].map(diagnosis2idx)

    # Save which idx represent the melanoma
    mel_idx = diagnosis2idx['melanoma']

    # Test dataframe
    test_path = data_dir / Path(f'jpeg-melanoma-{data_folder}x{data_folder}/test.csv')
    df_test = pd.read_csv(test_path)
    df_test['filepath'] = df_test['image_name'].apply(
        lambda img: test_path.parents[0] / Path(f'test/{img}.jpg'))

    return df_train, df_test, mel_idx


def get_transforms(image_size):
    """
    Returns a pair of transformers.
    The first transformer applies image augmentation of different kinds and resize the image.
    The second transformer, is thought to be used in test data, resizes and normalize the image.
    """

    transforms_train = A.Compose([
        A.Transpose(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightness(limit=0.2, p=0.75),
        A.RandomContrast(limit=0.2, p=0.75),
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=5),
            A.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7),

        A.OneOf([
            A.OpticalDistortion(distort_limit=1.0),
            A.GridDistortion(num_steps=5, distort_limit=1.),
            A.ElasticTransform(alpha=3),
        ], p=0.7),

        A.CLAHE(clip_limit=4.0, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        A.Resize(image_size, image_size),
        A.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7),
        A.Normalize()
    ])

    transforms_val = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize()
    ])

    return transforms_train, transforms_val
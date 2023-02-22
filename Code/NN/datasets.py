from pathlib import Path
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset
import cv2
import torch

class MelanomaDataset(Dataset):

    def __init__(self, csv, mode, meta_features, transform=None):
        self.csv = csv.reset_index(drop=True)
        self.mode = mode
        self.use_meta = meta_features is not None
        self.meta_features = meta_features
        self.transform = transform

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):

        sample = self.csv.iloc[index]

        image = cv2.imread(sample.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        # This makes color channel first
        image = image.transpose(2, 0, 1)

        # Optionally you can ask to get also metadata in conjuntion to the image
        if self.use_meta:
            data = (torch.tensor(image).float(), torch.tensor(sample[self.meta_features]).float())
        else:
            data = torch.tensor(image).float()

        # If this is just for a test porpouse you can forget to the label
        if self.mode == 'test':
            return data
        else:
            return data, torch.tensor(sample.target).long()



def get_meta_data(df_train, df_test):

    # One-hot encoding of anatom_site_general_challenge feature
    concat = pd.concat([df_train['anatom_site_general_challenge'], df_test['anatom_site_general_challenge']], ignore_index=True)
    dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix='site')
    df_train = pd.concat([df_train, dummies.iloc[:df_train.shape[0]]], axis=1)
    df_test = pd.concat([df_test, dummies.iloc[df_train.shape[0]:].reset_index(drop=True)], axis=1)

    # Sex features
    df_train['sex'] = df_train['sex'].map({'male': 1, 'female': 0})
    df_test['sex'] = df_test['sex'].map({'male': 1, 'female': 0})
    df_train['sex'] = df_train['sex'].fillna(-1)
    df_test['sex'] = df_test['sex'].fillna(-1)

    # Age features
    df_train['age_approx'] /= 90
    df_test['age_approx'] /= 90
    df_train['age_approx'] = df_train['age_approx'].fillna(0)
    df_test['age_approx'] = df_test['age_approx'].fillna(0)
    df_train['patient_id'] = df_train['patient_id'].fillna(0)

    # Compute the number of images that each user has
    df_train['n_images'] = df_train['patient_id'].map(df_train.groupby(['patient_id']).image_name.count())
    df_test['n_images'] = df_test['patient_id'].map(df_test.groupby(['patient_id']).image_name.count())
    df_train.loc[df_train['patient_id'] == -1, 'n_images'] = 1
    df_train['n_images'] = np.log1p(df_train['n_images'].values)
    df_test['n_images'] = np.log1p(df_test['n_images'].values)

    # Compute train image size
    train_images = df_train['filepath'].values
    train_sizes = np.zeros(train_images.shape[0])
    for i, img_path in enumerate(tqdm(train_images)):
      if (not Path(img_path).exists()):
        print(f"Img {img_path} doen't exist")
      else:
        train_sizes[i] = os.path.getsize(img_path)
    df_train['image_size'] = np.log(train_sizes)

    # Compute test image size
    test_images = df_test['filepath'].values
    test_sizes = np.zeros(test_images.shape[0])
    for i, img_path in enumerate(tqdm(test_images)):
      if (not Path(img_path).exists()):
        print(f"Img {img_path} doen't exist")
      else:
        test_sizes[i] = os.path.getsize(img_path)
    df_test['image_size'] = np.log(test_sizes)

    meta_features = ['sex', 'age_approx', 'n_images', 'image_size'] + [col for col in df_train.columns if col.startswith('site_')]
    n_meta_features = len(meta_features)

    return df_train, df_test, meta_features, n_meta_features


def get_df(out_dim: int, data_dir: str, data_folder: str, use_meta:bool = False):

  # 2020 data
  train_path = Path(data_dir) / Path(f'jpeg-melanoma-{data_folder}x{data_folder}/train.csv')
  df_train = pd.read_csv(train_path)

  # Drops samples where tfrecord is -1
  df_train = df_train[df_train['tfrecord'] >= 0].reset_index(drop=True)
  df_train['fold'] = df_train['tfrecord']
  df_train['filepath'] = df_train['image_name'].apply(lambda img : train_path.parents[0] / Path(f'train/{img}.jpg'))
  df_train['is_ext'] = 0

  # 2018 and 2019 data
  train_path = Path(data_dir) / Path(f'jpeg-isic2019-{data_folder}x{data_folder}/train.csv')
  df_train2 = pd.read_csv(train_path)
  df_train2 = df_train2[df_train2['tfrecord'] >= 0].reset_index(drop=True)
  df_train2['filepath'] = df_train2['image_name'].apply(lambda img: train_path.parents[0] / Path(f'train/{img}.jpg'))
  df_train2['is_ext'] = 1

  # Preprocess Target
  df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('seborrheic keratosis', 'BKL'))
  df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('lichenoid keratosis', 'BKL'))
  df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('solar lentigo', 'BKL'))
  df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('lentigo NOS', 'BKL'))

  # Join categorical variables to adapt to the output of the nn
  if out_dim == 8:
    df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('NV', 'nevus'))
    df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('MEL', 'melanoma'))
  elif out_dim == 4:
    df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('NV', 'nevus'))
    df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('MEL', 'melanoma'))
    df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('DF', 'unknown'))
    df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('AK', 'unknown'))
    df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('SCC', 'unknown'))
    df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('VASC', 'unknown'))
    df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('BCC', 'unknown'))
  else:
    raise NotImplementedError()

  # Definition of the hole train dataframe
  df_train = pd.concat([df_train, df_train2]).reset_index(drop=True)

  # Create target column that represents the diagnostic using a number
  diagnosis2idx = {d: idx for idx, d in enumerate(sorted(df_train.diagnosis.unique()))}
  df_train['target'] = df_train['diagnosis'].map(diagnosis2idx)

  # Save which idx represent the melanoma
  mel_idx = diagnosis2idx['melanoma']

  # Test dataframe
  test_path = Path(data_dir) / Path(f'jpeg-melanoma-{data_folder}x{data_folder}/test.csv')
  df_test = pd.read_csv(test_path)
  df_test['filepath'] = df_test['image_name'].apply(lambda img : test_path.parents[0] / Path(f'test/{img}.jpg'))

  if use_meta:
    df_train, df_test, meta_features, n_meta_features = get_meta_data(df_train, df_test)
  else:
    meta_features = None
    n_meta_features = 0

  return df_train, df_test, meta_features, n_meta_features, mel_idx
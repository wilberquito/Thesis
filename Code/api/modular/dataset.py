from pathlib import Path

import numpy as np
import pandas as pd
import PIL as pil
import torch

from modular.utility import find_files


class TaskDataset(torch.utils.data.Dataset):
    def __init__(self,
                 csv: pd.DataFrame,
                 transform=None):

        self.csv = csv.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):

        sample = self.csv.iloc[index]
        img_path: Path = sample['path']
        img = pil.Image(img_path)

        if self.transform is not None:
            res = self.transform(image=img)
            img = res['image'].astype(np.float32)
        else:
            img = img.astype(np.float32)

        print(img.shape)

        # Channel first
        img = img.transpose(2, 0, 1)

        # Convert PIL img to Pytorch tensor
        tensor =  torch.tensor(img).float()

        print(img.shape)

        return tensor


def get_csv(parent_dir: Path,
            extensions=('png')):

    # Loads the images
    images_path = find_files(parent_dir=parent_dir,
                             extensions=('.png'))
    names = [img.name for img in images_path]

    # Built the csv
    csv = pd.DataFrame({
        'name': names,
        'path': images_path
    })

    return csv
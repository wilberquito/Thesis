from pathlib import Path

import numpy as np
import pandas as pd
import cv2
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
        filepath: Path = sample.filepath
        image = cv2.imread(str(filepath))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        # Channel first
        image = image.transpose(2, 0, 1)

        # Convert PIL image to Pytorch tensor
        tensor =  torch.tensor(image).float()

        return tensor


def get_csv(parent_dir: Path):

    # Loads the images
    images_path = find_files(parent_dir=parent_dir,
                             extensions=('.png'))
    names = [image.name for image in images_path]

    # Built the csv
    csv = pd.DataFrame({
        'name': names,
        'filepath': images_path
    })

    return csv
from pathlib import Path
import pandas as pd
import os
import shutil


def mapping(csv: pd.DataFrame) -> pd.DataFrame:
    """
    This function maps some classes to more generic classes
    """
    csv['diagnosis'] = csv['diagnosis'] \
        .apply(lambda x: x.replace('seborrheic keratosis', 'BKL'))
    csv['diagnosis'] = csv['diagnosis'] \
        .apply(lambda x: x.replace('lichenoid keratosis', 'BKL'))
    csv['diagnosis'] = csv['diagnosis'] \
        .apply(lambda x: x.replace('solar lentigo', 'BKL'))
    csv['diagnosis'] = csv['diagnosis'] \
        .apply(lambda x: x.replace('lentigo NOS', 'BKL'))
    csv['diagnosis'] = csv['diagnosis'] \
        .apply(lambda x: x.replace('NV', 'nevus'))
    csv['diagnosis'] = csv['diagnosis'] \
        .apply(lambda x: x.replace('MEL', 'melanoma'))
    csv['diagnosis'] = csv['diagnosis'] \
        .apply(lambda x: x.replace('cafe-au-lait macule', 'unknown'))
    csv['diagnosis'] = csv['diagnosis'] \
        .apply(lambda x: x.replace('atypical melanocytic proliferation', 'unknown'))

    return csv


def train_etl(from_path: str, to_path: str, folder: str):
    train_csv = Path(f'{from_path}/{folder}/train.csv')
    csv = pd.read_csv(train_csv)
    # Maps targets
    csv = mapping(csv)
    # Drop entries with no tfrecord
    csv = csv[csv['tfrecord'] != -1].reset_index(drop=True)
    # Drop unknown diagnosis
    csv = csv[csv['diagnosis'] != 'unknown'].reset_index(drop=True)
    # Set the filepath
    csv['filepath'] = csv['image_name'] \
        .apply(lambda x: os.path.join(to_path, f'{folder}/train/{x}.jpg'))
    # Set the filepath (origin)
    csv['filepath.origin'] = csv['image_name'] \
        .apply(lambda x: os.path.join(from_path, f'{folder}/train/{x}.jpg'))

    tuples = zip(csv['filepath.origin'].values, csv['filepath'].values)
    for src_file, dst_file in tuples:
        origin, target = Path(src_file), Path(dst_file)
        target.parents[0].mkdir(parents=True, exist_ok=True)
        shutil.copy(origin, target)

    csv = csv.drop('filepath.origin', axis=1)
    # Saves the file into the to origin
    csv.to_csv(f'{to_path}/{folder}/train.csv')


def test_etl(from_path: str, to_path: str, folder: str):
    test_csv = Path(f'{from_path}/{folder}/test.csv')
    csv = pd.read_csv(test_csv)
    # Set the filepath
    csv['filepath'] = csv['image_name'] \
        .apply(lambda x: os.path.join(to_path, f'{folder}/test/{x}.jpg'))
    # Copy the hole dataset from origin
    origin = Path(f'{from_path}/{folder}/test')
    target = Path(f'{to_path}/{folder}/test')
    shutil.copytree(origin, target)
    # Saves the file into the to origin
    csv.to_csv(f'{to_path}/{folder}/test.csv')


def optimizer(from_path: str, to_path: str, folder: str):
    """
    Function that tries map the origin folder to
    the destinity without some data unnneed
    """

    print('Optimizing: ' + folder)

    if 'melanoma' in folder:
        train_etl(from_path, to_path, folder)
        test_etl(from_path, to_path, folder)
    else:
        train_etl(from_path, to_path, folder)


if __name__ == '__main__':
    # Make the parent folder if it does not exist
    to_path = './data.etl'
    from_path = './data'

    xss = [
        'jpeg-melanoma-1024x1024',
        'jpeg-melanoma-512x512',
        'jpeg-melanoma-768x768',
        'jpeg-isic2019-1024x1024',
        'jpeg-isic2019-768x768',
        'jpeg-isic2019-512x512'
    ]

    for origin in xss:
        folder = Path(f'{to_path}/{origin}')
        Path(folder).mkdir(exist_ok=True, parents=True)
        optimizer(from_path, to_path, origin)

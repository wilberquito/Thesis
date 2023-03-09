"""
Contains various utility functions for PyTorch model training and saving.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def transform(img: torch.tensor, transform_type: int):
    # Reverse the image applying a transpose
    if transform_type >= 4:
        img = img.transpose(2, 3)
    # Do nothing
    if transform_type % 4 == 0:
        return img
    # Flip img in axis 0
    elif transform_type % 4 == 1:
        return img.flip(2)
    # Flip img in axis 1
    elif transform_type % 4 == 2:
        return img.flip(3)
    # Flip img in axis 0 and 1
    elif transform_type % 4 == 3:
        return img.flip(2).flip(3)


def model_input_size(model_name):
    """Mapping from model name
    to the image size that a model support as image size value"""
    model_name = model_name.lower()
    mapping = {
        'effnet': 600,
        'resnet': 232,
        'convnext': 232
    }
    return mapping[model_name]


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)


def plot_images(dataset: torch.utils.data.Dataset, rows=3, cols=3):
    """Plots nrows*ncols random images"""
    torch.manual_seed(42)
    fig = plt.figure(figsize=(9, 9))
    class_names = dataset.classes
    for i in range(1, rows * cols + 1):
        random_idx = torch.randint(0, len(dataset), size=[1]).item()
        img, label = dataset[random_idx]
        fig.add_subplot(rows, cols, i)
        plt.imshow(img.squeeze(), cmap="gray")
        plt.title(class_names[label])
        plt.axis(False);
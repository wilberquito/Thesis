"""
Contains various utility functions for PyTorch model training and saving.
"""

import random
from pathlib import Path

import numpy as np
import torch
import pandas as pd


import matplotlib.pyplot as plt
from typing import List, Dict
import torchvision
from torchmetrics import ConfusionMatrix
import mlxtend.plotting as plotting

import modular.checkpoint as checkpoint


# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def print_train_time(start, end, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format).
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time


@torch.inference_mode()
def plot_confusion_matrix(model: torch.nn.Module,
                          val_dataloader: torch.utils.data.DataLoader,
                          class_names: list,
                          device: torch.device):

    y_preds = []
    y_labels = []
    model.eval()

    for inputs, labels in val_dataloader:
        # Send data and targets to target device
        inputs, labels = inputs.to(device), labels.to(device)
        # Do the forward pass
        y_logit = model(inputs)
        # Turn predictions from logits to labels
        y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
        # Put predictions on CPU for evaluation
        y_preds.append(y_pred.cpu())
        # Put the labels on CPU for evaluation
        y_labels.append(labels.cpu())

    # Concatenate list of predictions into a tensor
    y_pred_tensor = torch.cat(y_preds)
    y_labels_tensor = torch.cat(y_labels)

    confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
    confmat_tensor = confmat(preds=y_pred_tensor,
                             target=y_labels_tensor)

    fig, ax = plotting.plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),
        class_names=class_names,
        figsize=(10, 7))


# Plot loss curves of a model
def plot_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "train_auc": [...],
             "val_loss": [...],
             "val_auc": [...],
             "val_acc": [...]}
    """

    auc = results["train_auc"]
    val_auc = results["val_auc"]

    loss = results["train_loss"]
    val_loss = results["val_loss"]

    accuracy = results["train_acc"]
    val_accuracy = results["val_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot auc
    plt.subplot(1, 3, 1)
    plt.plot(range(1, epochs), auc, label="train_auc")
    plt.plot(range(1, epochs), val_auc, label="val_auc")
    plt.title("Area Under the Curve")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot loss
    plt.subplot(1, 3, 2)
    plt.plot(range(1, epochs), loss, label="train_loss")
    plt.plot(range(1, epochs), val_loss, label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 3, 3)
    plt.plot(range(1, epochs), accuracy, label="train_accuracy")
    plt.plot(range(1, epochs), val_accuracy, label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


def pred_and_plot_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str] = None,
    transform=None,
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Makes a prediction on a target image with a trained model and plots the image.

    Args:
        model (torch.nn.Module): trained PyTorch image classification model.
        image_path (str): filepath to target image.
        class_names (List[str], optional): different class names for target image. Defaults to None.
        transform (_type_, optional): transform of target image. Defaults to None.
        device (torch.device, optional): target device to compute on. Defaults to "cuda" if torch.cuda.is_available() else "cpu".

    Returns:
        Matplotlib plot of target image and model prediction as title.

    Example usage:
        pred_and_plot_image(model=model,
                            image="some_image.jpeg",
                            class_names=["class_1", "class_2", "class_3"],
                            transform=torchvision.transforms.ToTensor(),
                            device=device)
    """

    # 1. Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255.0

    # 3. Transform if necessary
    if transform:
        target_image = transform(target_image)

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))

    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 8. Plot the image alongside the prediction and prediction probability
    plt.imshow(
        target_image.squeeze().permute(1, 2, 0)
    )  # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)


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


def set_seed(seed=42):
    """Force determinism in different libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def display_random_images(dataset: torch.utils.data.dataset.Dataset,
                          n: int = 10,
                          display_label: bool = True,
                          display_shape: bool = True,
                          seed: int = None):
    """Display random images from a dataset collection"""

    # 2. Adjust display if n too high
    if n > 10:
        display_shape = False
        print("""For display purposes, n shouldn't
        be larger than 10, setting to 10 and removing shape display.""")

    # 3. Set random seed
    if seed:
        random.seed(seed)

    # 4. Get random sample indexes
    random_samples_idx = random.sample(range(len(dataset)), k=n)

    # 5. Setup plot
    plt.figure(figsize=(16, 8))

    # 6. Loop through samples and display random samples
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

        # 7. Adjust image tensor shape for plotting:
        # [color_channels, height, width] -> [height, width, color_channels]
        targ_image = targ_image.cpu().numpy()
        targ_image_adjust = targ_image.transpose(1, 2, 0)
        targ_image_adjust = targ_image_adjust / 255

        # Plot adjusted samples
        plt.subplot(n // 10 + 1, n if n <= 10 else n // 2, i+1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        if display_label:
            title = dataset.idx_to_class[targ_label]
        if display_shape:
            title = title + f"\nshape: {targ_image_adjust.shape}"
        plt.title(title)


def model_writter(model_name: str):
    """Saves the pythorch trainned model and generates
    a log file in csv format"""

    def writter(point: Dict):
        # Save checkpoint
        checkpoint.save_checkpoint(point,
                                   model_name + 'pth.tar')

        # Logging the trainning
        log_filename = model_name + '.csv'
        stats = point['stats']
        pd.DataFrame(stats).to_csv(log_filename)

    return writter

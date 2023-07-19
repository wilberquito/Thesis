"""
Module to train neural network. It supports

 - Early stop
 - Scheduler
 - Number of epochs
 - Test-Time augmentation
 - Writter
"""

from typing import Dict, Callable, NewType

import torch
import time
import torch.nn as nn
import copy
from sklearn.metrics import roc_auc_score
import numpy as np
import modular.test as test

Writter = NewType("Writter", Callable[[Dict], None])


def train_model(model: nn.Module,
                mel_idx: int,
                about_data: Dict,
                device: torch.device,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler.LRScheduler = None,
                num_epochs: int = 25,
                patience: int = 5,
                writter: Writter = None,
                val_times: int = 1):

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_optimizer_wts = copy.deepcopy(optimizer.state_dict())
    best_scheduler_wts = \
        copy.deepcopy(scheduler.state_dict()) if scheduler else None
    best_ovr = 0
    best_epoch = 0

    stats = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "train_ovr": [],
        "val_ovr": []
    }

    # Writter mechanism agnositc to train method
    is_save_required = writter is not None
    # Do I need to apply just in time test?
    val_augmentation_required = val_times > 1
    # Patience counter early stop
    early_stop_count = 0

    dataloaders, datasets = about_data['dataloaders'], about_data['datasets']

    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            EPOCH_LABELS = []
            EPOCH_PROBS = []
            # Iterate over data (batches).
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'val' and val_augmentation_required:
                        outputs = test.test_time_agumentation(model,
                                                              inputs,
                                                              val_times)
                    else:
                        outputs = model(inputs)

                    probs = torch.softmax(outputs, 1)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    EPOCH_LABELS.append(labels.detach().cpu())
                    EPOCH_PROBS.append(probs.detach().cpu())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / datasets['size'][phase]
            epoch_acc = running_corrects / datasets['size'][phase]
            epoch_ovr = compute_ovr(mel_idx, EPOCH_LABELS, EPOCH_PROBS)

            cphase = phase.capitalize()
            print(f'{cphase} OvR: {epoch_ovr:.4f} \t|\t' +
                  f'{cphase} Loss: {epoch_loss:.4f} \t|\t' +
                  f'{cphase} Acc: {epoch_acc:.4f}')

            # Update results dictionary
            stats[f"{phase}_ovr"].append(round(epoch_ovr, 4))
            stats[f"{phase}_acc"].append(round(epoch_acc, 4))
            stats[f"{phase}_loss"].append(round(epoch_loss, 4))

            # Scheduler step
            if phase == 'train' and scheduler:
                scheduler.step()

            # Valid phase
            if phase == 'val':
                # Check if network has learned
                network_learned = epoch_ovr > best_ovr

        # Updates metadata of the trainning
        if network_learned:
            # Reset early stopping
            early_stop_count = 0
            # Updates metadata
            best_ovr = epoch_ovr
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            best_optimizer_wts = copy.deepcopy(optimizer.state_dict())
            best_scheduler_wts = \
                copy.deepcopy(scheduler.state_dict()) if scheduler else None
        else:
            early_stop_count += 1

        # Save model state after every epoch if writter is defined
        if is_save_required:
            save_point = {
                'best_epoch': best_epoch,
                'optimizer_state_dict': best_optimizer_wts,
                'scheduler_state_dict': best_scheduler_wts,
                'model_state_dict': best_model_wts,
                'stats': stats,
                'epochs': epoch
            }
            writter(save_point)

        # Stop trainning
        if early_stop_count >= patience:
            print(f'\nEarly stopping after {epoch} epochs')
            break

    time_elapsed = time.time() - since
    formated_time = f'{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s'
    print(f'\nTraining completed after: {formated_time}')
    print(f'Best epoch: {best_epoch}')
    print(f'Best val OvR: {best_ovr:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, stats


def compute_ovr(target: int, y_true: list, y_probs: list):
    """This functions expectes the one element to
    be compared with the others and two list of tensors.
    The first list is the true value and the other represent
    the prediction made by the model.
    """

    # To numpy array
    y_true = torch.cat(y_true).numpy()
    y_probs = torch.cat(y_probs).numpy()

    # Adjust the shape of y_true if necessary
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)

    # Adjust the shape of y_probs if necessary
    if len(y_probs.shape) > 1:
        y_probs = y_probs[:, target]

    matches = (y_true == target).astype(int)
    ovr = roc_auc_score(matches, y_probs, multi_class="ovr")
    return ovr

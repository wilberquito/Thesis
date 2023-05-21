"""
Module to train neural network. It supports

 - Early stop
 - Scheduler
 - Number of epochs
 - Test-time augmentation
 - Writter
"""

from typing import Dict, Callable, NewType

import torch
import time
import torch.nn as nn
import copy

StopEvaluator = NewType("StopEvaluator",
                        Callable[[torch.Tensor, torch.Tensor], torch.Tensor])

Writter = NewType("Writter", Callable[[Dict], None])


def train_model(model: nn.Module,
                dataloaders: Dict,
                datasets_size: Dict,
                device: torch.device,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler.LRScheduler = None,
                num_epochs: int = 25,
                patience: int = 5,
                early_stop_evaluator: StopEvaluator = None,
                writter: Writter = None,
                val_times: int = 1):

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_estimator = 0.0

    stats = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    is_save_required = writter is not None
    val_agumentation_required = val_times > 1
    early_stop_count = 0

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

            # Iterate over data (batches).
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'val' and val_agumentation_required:
                        outputs = val_augmentation(model, inputs, val_times)
                    else:
                        outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} ' +
                  f'{phase.capitalize()} Acc: {epoch_acc:.4f}')

            # Update results dictionary
            stats[f"{phase}_loss"].append(round(epoch_loss, 4))
            stats[f"{phase}_acc"].append(round(epoch_acc, 4))

            # Scheduler step
            if phase == 'train' and scheduler:
                scheduler.step()

            # Valid phase
            if phase == 'val':
                # Check if network has learned
                if early_stop_evaluator:
                    epoch_estimator = early_stop_evaluator(preds, labels)
                    network_learned = epoch_estimator > best_estimator
                else:
                    network_learned = epoch_acc > best_acc

        # Save model if required after every epoch
        if network_learned:
            early_stop_count = 0
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

            if is_save_required:
                optimizer_wts = optimizer.state_dict()
                scheduler_wts = \
                    scheduler.state_dict() if scheduler else None
                save_point = {
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer_wts,
                    'scheduler_state_dict': scheduler_wts,
                    'model_state_dict': best_model_wts,
                    'stats': stats,
                }
                if (writter):
                    writter(save_point)
        else:
            # Early stop the training
            early_stop_count += 1
            if early_stop_count >= patience:
                print(f'\nEarly stopping after {epoch} epochs')
                break

    time_elapsed = time.time() - since
    print(f'\nTraining complete in \
        {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, stats


@torch.inference_mode()
def tta_validation(model: torch.nn, inputs: torch.Tensor, val_times: int):
    predictions = []
    for n in range(val_times):
        augmented_img = tta_transform(inputs, n)
        augmented_img = torch.unsqueeze(augmented_img, 0)
        outputs = model(tta_transform(inputs, n))
        probabilities = torch.softmax(outputs, dim=1)
        predictions.append(probabilities)

    averaged_predictions = torch.mean(predictions, dim=0)
    return averaged_predictions


def tta_transform(img: torch.Tensor, n: int):
    if n >= 4:
        img = img.transpose(2, 3)
    if n % 4 == 0:
        return img
    elif n % 4 == 1:
        return img.flip(2)
    elif n % 4 == 2:
        return img.flip(3)
    elif n % 4 == 3:
        return img.flip(2).flip(3)

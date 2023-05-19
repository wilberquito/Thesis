"""
Module to train neural network. It supports

 - Early stop
 - Scheduler
 - Number of epochs
 - Writter
"""

from typing import Dict, Callable

import torch
import time
import torch.nn as nn
import copy


def train_model(model: nn.Module,
                dataloaders: Dict,
                dataset_sizes: Dict,
                device: torch.device,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler.LRScheduler = None,
                num_epochs: int = 25,
                patience: int = 3,
                writter: Callable[[Dict], None] = None):

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    stats = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    is_save_required = writter is not None

    early_stop_count = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
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
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} ' +
                  f'{phase.capitalize()} Acc: {epoch_acc:.4f}')

            # Update results dictionary
            stats[f"{phase}_loss"].append(epoch_loss)
            stats[f"{phase}_acc"].append(epoch_acc)

            # Scheduler step
            if phase == 'train' and scheduler:
                scheduler.step()

            # Valid phase
            if phase == 'val':
                # Check if network has learned
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
                checkpoint = {
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer_wts,
                    'scheduler_state_dict': scheduler_wts,
                    'model_state_dict': best_model_wts,
                    'stats': stats,
                }
                if (writter):
                    writter(checkpoint)
        else:
            # Early stop the training
            early_stop_count += 1
            if early_stop_count >= patience:
                print(f'\nEarly stopping after {epoch} epochs')
                break

    time_elapsed = time.time() - since
    print(f'\nTraining complete in \
        {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, stats

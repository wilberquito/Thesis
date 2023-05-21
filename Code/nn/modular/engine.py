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

StopEvaluator = NewType("StopEvaluator",
                        Callable[[torch.Tensor, torch.Tensor], torch.Tensor])

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
                patience_metric: str = "auc",
                writter: Writter = None,
                val_times: int = 1):

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc = 0

    stats = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "train_auc": [],
        "val_auc": []
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

            # Iterate over data (batches).
            for inputs, labels in dataloaders[phase]:
                PROBS, LABELS = [], []
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'val' and val_augmentation_required:
                        outputs = tta_validation(model, inputs, val_times)
                    else:
                        outputs = model(inputs)

                    probs = torch.softmax(outputs, 1)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    PROBS.append(probs.detach().cpu())
                    LABELS.append(labels.detach().cpu())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / datasets['size'][phase]
            epoch_acc = running_corrects / datasets['size'][phase]
            PROBS, LABELS = torch.cat(PROBS).numpy(), torch.cat(LABELS).numpy()
            epoch_auc = roc_auc_score((LABELS == mel_idx).astype(float),
                                      PROBS[:, mel_idx])

            cphase = phase.capitalize()
            print(f'{cphase} Auc: {epoch_auc:.4f} \t|\t' +
                  f'{cphase} Loss: {epoch_loss:.4f} \t|\t' +
                  f'{cphase} Acc: {epoch_acc:.4f}')

            # Update results dictionary
            stats[f"{phase}_loss"].append(round(epoch_loss, 4))
            stats[f"{phase}_acc"].append(round(epoch_acc, 4))
            stats[f"{phase}_auc"].append(round(epoch_auc, 4))

            # Scheduler step
            if phase == 'train' and scheduler:
                scheduler.step()

            # Valid phase
            if phase == 'val':
                # Check if network has learned
                network_learned = epoch_auc > best_auc

        # Save model if required after every epoch
        if network_learned:
            early_stop_count = 0
            best_auc = epoch_auc
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
    print(f'Best Val AUC: {best_auc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, stats


@torch.inference_mode()
def tta_validation(model: torch.nn, inputs: torch.Tensor, val_times: int):
    """Applies time test transformation to a set of tensor images
    an returns the logits"""

    logits = []
    for n in range(val_times):
        augmented_img = tta_transform(inputs, n)
        augmented_img = torch.unsqueeze(augmented_img, 0)
        outputs = model(tta_transform(inputs, n))
        logits.append(outputs)

    stacked_logits = torch.stack(logits)
    stacked_logits = torch.mean(stacked_logits, dim=0)
    return stacked_logits


def tta_transform(img: torch.Tensor, n: int):
    """Given a tensor it applies dummy transformation
    on de n value"""

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

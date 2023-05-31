"""
Utility functions to make predictions.
"""

import torch


@torch.inference_mode()
def tta(model: torch.nn, inputs: torch.Tensor, val_times: int):
    """Applies time test time agumentation to a set of tensor images
    an returns the logits"""

    model.eval()

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

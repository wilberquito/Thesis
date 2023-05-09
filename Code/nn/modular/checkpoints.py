import os
import torch
from typing import Optional, Dict


# If exists a best model, load its weights!
def load_checkpoint(resume_weights: str,
                    cuda: bool = False) -> Optional[Dict]:
    if os.path.isfile(resume_weights):
        print("=> loading checkpoint '{}' ...".format(resume_weights))
        if cuda:
            checkpoint = torch.load(resume_weights)
        else:
            # Load GPU model on CPU
            checkpoint = torch.load(resume_weights,
                                    map_location=lambda storage,
                                    loc: storage)
        print("=> loaded checkpoint '{}' (trained for {} epochs)".
              format(resume_weights, checkpoint['epoch']))

    return checkpoint


def save_checkpoint(state: Dict, is_best, history: Optional[Dict] filename='/output/checkpoint.pth.tar'):
    """Save checkpoint if a new best is achieved"""

    if is_best:
        print("=> Saving a new best")
        torch.save(state, filename)  # save checkpoint
    else:
        print("=> Validation Accuracy did not improve")

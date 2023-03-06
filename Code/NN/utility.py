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
    model_name = model_name.lower()
    mapping = {
        'effnet': 600
    }
    return mapping[model_name]

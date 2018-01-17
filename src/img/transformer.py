import torch
import numpy as np


def image_to_tensor(image):
    """
    Transforms an image to a tensor
    Args:
        image (np.ndarray): A RGB array image
        mean: The mean of the image values
        std: The standard deviation of the image values

    Returns:
        tensor: A Pytorch tensor
    """
    if len(image.shape) < 3:
        image = image.reshape(image.shape[0], image.shape[1], 1)

    image = image.astype(np.float32)
    image = image.transpose((2, 0, 1))
    tensor = torch.from_numpy(image)
    return tensor

def mask_to_tensor(mask, threshold, invert_mask = False):
    """
    Transforms a mask to a tensor
    Args:
        mask (np.ndarray): A greyscale mask array
        threshold: The threshold used to consider the mask present or not

    Returns:
        tensor: A Pytorch tensor
    """
    mask = mask
    mask = mask > threshold
    if invert_mask:
        mask = np.invert(mask)

    mask = mask.astype(np.float32)

    tensor = torch.from_numpy(mask).type(torch.FloatTensor)
    return tensor


def label_to_tensor(label):
    tensor = torch.from_numpy(label)
    return tensor

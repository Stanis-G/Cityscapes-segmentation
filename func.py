import os
from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import torch
from torch.nn.functional import one_hot
from torch.utils.data import Dataset


# Colormap is taken from here https://github.com/tensorflow/models/blob/17e923da9e8caba5dfbd58846ce75962206ffa64/research/deeplab/utils/get_dataset_colormap.py#L207
COLORMAP = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32],
])


class CustomData(Dataset):
    
    def __init__(self, fold, transform=None, return_onehot=False):
        self.path = os.path.join('data', fold)
        self.files = sorted(os.listdir(self.path))
        self.transform = transform
        self.return_onehot = return_onehot

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """Read file and split it to image and mask"""
        filename = self.files[idx] # Find filename by index
        filepath = os.path.join(self.path, filename) # Create path to file
        img = cv.cvtColor(cv.imread(filepath), cv.COLOR_BGR2RGB) # Read file and convert to RGB
        
        img_in = img[:,:256,:] # Extract input image
        img_in = torch.from_numpy(img_in).float() / 255 # Convert np.array to torch.Tensor and notmalize to range (0,1)
        img_in = img_in.permute(2, 0, 1) # Put channels first, cause Conv2D works that way
        
        img_out = img[:,256:,:] # Extract output image
        img_out = rgb2class(img_out, return_onehot=self.return_onehot) # Reassign channels meaning from RGB values to class labels
        img_out = torch.from_numpy(img_out).float()
        if self.return_onehot:
            img_out = img_out.permute(2, 0, 1) # Put channels first, because model prediction will have this format

        if self.transform:
            img_in = self.transform(img_in)
            img_out = self.transform(img_out)

        return img_in, img_out
    

def rgb2class(img_out, return_onehot=False):
    """Transform image output mask from shape (width, height, num_channels) to shape (width, height, num_classes).
    Utilize Euclidian distance to find nearest point from colormap to current pixel. It is necessary due to corruption of pixels in datasets
    """
    num_classes = len(COLORMAP)
    
    dst_v = np.expand_dims(img_out, axis=0) - COLORMAP.reshape(num_classes, 1, 1, 3) # Distance vectors from each pixel to each color from colormap
    dst = np.linalg.norm(dst_v, axis=3) # Norm of distance vectors (means Euclidian distance from each pixel to each color from colormap)
    indeces = np.argmin(dst, axis=0) # Indeces of minimal Euclidian distance (index corresponds to color from colormap)
    if return_onehot:
        onehot_image = np.eye(num_classes)[indeces] # Apply onehot encoding
        return onehot_image
    return indeces


def mask2image(mask, return_bgr=False):
    """Restore image using its class mask and colormap"""
    img_out = COLORMAP[mask]
    if return_bgr:
        img_out = img_out.permute(2, 0, 1)
    return img_out


def plot_masks(img_in, img_target, img_predict):
    """Plot original image, ground truth output and predicted image"""

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    # Display each image in a different subplot
    axes[0].imshow(img_in)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(img_target)
    axes[1].set_title('Ground truth')
    axes[1].axis('off')

    axes[2].imshow(img_predict)
    axes[2].set_title('Predicted')
    axes[2].axis('off')

    plt.tight_layout()
    return fig


def pixel_accuracy(predict, target, channel_dim=1):
    """Compute pixel-wise accuracy"""
    accuracy = (predict.argmax(dim=channel_dim) == target).sum() / target.numel()
    return accuracy.item()


def dsc_score(predict, target, channel_dim=1):
    """Compute Dice Similarity Coefficient"""
    predict_mask = predict.argmax(dim=channel_dim)
    num_classes = predict.shape[channel_dim]

    # Apply onehot encoding to masks
    onehot_predict = one_hot(predict_mask, num_classes=num_classes)
    onehot_target = one_hot(target, num_classes=num_classes)

    # Calc untersections and unions for each class
    intersection = (onehot_predict * onehot_target).sum(dim=(1, 2))
    union = onehot_predict.sum(dim=(1, 2)) + onehot_target.sum(dim=(1, 2)) - intersection

    # Calc mean dsc by classes
    dsc = (2 * intersection / union).mean()

    return dsc.item()


def plot_learning_curve(y_dict, loc='center right'):
    fig, ax = plt.subplots()
    x = [i for i in range(len(list(y_dict.values())[0]))]
    for y_label, y in y_dict.items():
        ax.plot(x, y, label=y_label)
    ax.set_xlabel('Epoch')
    ax.legend(loc=loc)
    return fig

"""Pixel-identical mapping to the GAP8 C downsampler for grayscale cameras."""
import numpy as np
from PIL import Image
import torch


class CameraTensor:
    def __call__(self, image):
        gray = np.asarray(image.convert('L'))
        height, width = gray.shape
        y = ((2 * np.arange(64) + 1) * height) // 128
        x = ((2 * np.arange(64) + 1) * width) // 128
        resized = gray[y[:, None], x[None, :]].copy()
        return torch.from_numpy(resized).float().unsqueeze(0) / 255

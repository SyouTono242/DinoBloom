# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import random
from typing import Sequence

import torch
from torchvision import transforms


class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)


class MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)


class EnsureThreeChannels(object):
    """
    Ensure tensor images use three channels by repeating grayscale channels.
    """

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 3:
            raise ValueError(f"Expected CHW tensor, got shape {tuple(tensor.shape)}")
        if tensor.shape[0] == 3:
            return tensor
        if tensor.shape[0] == 1:
            return tensor.repeat(3, 1, 1)
        raise ValueError(f"Unsupported channel count {tensor.shape[0]}; expected 1 or 3 channels")


class GaussianNoise(object):
    """
    Add light Gaussian noise to tensor images.
    """

    def __init__(self, *, p: float = 0.5, sigma_min: float = 0.005, sigma_max: float = 0.03):
        self.p = p
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return tensor
        sigma = random.uniform(self.sigma_min, self.sigma_max)
        noise = torch.randn_like(tensor) * sigma
        return (tensor + noise).clamp(0.0, 1.0)


# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def _normalize_stats_to_rgb(stats: Sequence[float]) -> Sequence[float]:
    if len(stats) == 1:
        return (stats[0], stats[0], stats[0])
    return stats


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    mean = _normalize_stats_to_rgb(mean)
    std = _normalize_stats_to_rgb(std)
    return transforms.Normalize(mean=mean, std=std)


# This roughly matches torchvision's preset for classification training:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L6-L44
def make_classification_train_transform(
    *,
    crop_size: int = 224,
    interpolation=transforms.InterpolationMode.BICUBIC,
    hflip_prob: float = 0.5,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
):
    transforms_list = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
    if hflip_prob > 0.0:
        transforms_list.append(transforms.RandomHorizontalFlip(hflip_prob))
    transforms_list.extend(
        [
            MaybeToTensor(),
            make_normalize_transform(mean=mean, std=std),
        ]
    )
    return transforms.Compose(transforms_list)


# This matches (roughly) torchvision's preset for classification evaluation:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69
def make_classification_eval_transform(
    *,
    resize_size: int = 256,
    interpolation=transforms.InterpolationMode.BICUBIC,
    crop_size: int = 224,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    transforms_list = [
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        MaybeToTensor(),
        make_normalize_transform(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)

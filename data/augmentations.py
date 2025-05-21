from typing import Callable, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms.v2 as v2
from torchvision import transforms
from torch import nn
from torchvision.transforms.v2 import Transform
from torchvision.transforms.v2 import functional as F
from PIL import ImageFilter, ImageOps, Image, ImageDraw
import random
from torchvision.transforms.v2 import InterpolationMode


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class RemoveChannel(nn.Module):
    def __init__(self, p: float = 0.2) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[0]
        if torch.rand(1) < self.p and c > 2:
            channel_to_blacken = torch.randint(0, c - 1, (1,))
            x[channel_to_blacken] = 0
        return x


class RescaleProtein(nn.Module):
    """
    attentuates or amplifies the last channel (4th channel)
    """
    def __init__(self, p: float = 0.2) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.p and x.max() > 0:
            random_factor = (np.random.rand() * 2) / (x.max() + 1e-6)
            x[-1] = x[-1] * random_factor
        return x


class PerChannelColorJitter(nn.Module):
    def __init__(self, brightness=0.5, contrast=0.5, saturation=None, hue=None, p=0.5):
        super().__init__()
        self.transform = v2.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(x.shape[0]):
            if torch.rand(1) < self.p:
                x[i] = self.transform(x[i][None, ...])
        return x


class PerChannelGaussianBlur(nn.Module):
    def __init__(self, kernel_size=7, sigma=(0.1, 2.0), p=0.5):
        super().__init__()
        self.transform = v2.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(x.shape[0]):
            if torch.rand(1) < self.p:
                x[i] = self.transform(x[i][None, ...])
        return x


class PerChannelAdjustSharpness(nn.Module):
    def __init__(self, sharpness_factor=2, p=0.5):
        super().__init__()
        self.transform = v2.RandomAdjustSharpness(sharpness_factor=sharpness_factor)
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(x.shape[0]):
            if torch.rand(1) < self.p:
                x[i] = self.transform(x[i][None, ...])
        return x


class GaussianNoise(nn.Module):
    def __init__(self, sigma_range: Tuple[float, float], p: float = 0.5) -> None:
        super().__init__()
        self.p = p
        self.sigma_range = sigma_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        c = x.shape[0]
        if torch.rand(1) < self.p:
            sigma = (
                torch.rand((c, 1, 1), device=device)
                * (self.sigma_range[1] - self.sigma_range[0])
                + self.sigma_range[0]
            )
            return x + (torch.randn_like(x) * sigma)
        else:
            return x


class PerChannelRandomErasing(nn.Module):
    def __init__(self, scale=(0.02, 0.1), ratio=(0.3, 3.3), p=0.5):
        super().__init__()
        self.transform = v2.RandomErasing(scale=scale, ratio=ratio)
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(x.shape[0]):
            if torch.rand(1) < self.p:
                x[i] = self.transform(x[i][None, ...])
        return x


class PerBatchCompose(Transform):
    def __init__(self, transforms: Sequence[Callable]) -> None:
        super().__init__()
        self.transforms = transforms

    def get_masked_transforms(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_t = [[] for _ in range(x.shape[0])]
        mask_t = [[] for _ in range(x.shape[0])]
        for i in range(x.shape[0]):
            x_i = x[i]
            mask_i = mask[i]
            for transform in self.transforms:
                x_i, mask_i = transform(x_i, mask_i)
            x_t[i] = x_i
            mask_t[i] = mask_i
        x_t = torch.stack(x_t)
        mask_t = torch.stack(mask_t)
        return x_t, mask_t

    def get_transforms(self, x):
        x_t = [[] for _ in range(x.shape[0])]
        for i in range(x.shape[0]):
            x_i = x[i]
            for transform in self.transforms:
                x_i = transform(x_i)
            x_t[i] = x_i
        x_t = torch.stack(x_t)
        return x_t

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is not None:
            return self.get_masked_transforms(x, mask)
        else:
            return self.get_transforms(x)

    def __repr__(self) -> str:
        format_string = []
        for t in self.transforms:
            format_string.append(f"    {t}")
        return self.__class__.__name__ + "(\n" + "\n".join(format_string) + "\n)"


class PerChannelCompose(Transform):
    def __init__(self, transforms: Sequence[Callable]) -> None:
        super().__init__()
        self.transforms = transforms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        reshape_x = x.view(b * c, 1, h, w)
        for i in range(b * c):
            for transform in self.transforms:
                reshape_x[i] = transform(reshape_x[i])
        trans_x = reshape_x.view(b, c, h, w)
        return trans_x

    def __repr__(self) -> str:
        format_string = []
        for t in self.transforms:
            format_string.append(f"    {t}")
        return self.__class__.__name__ + "(\n" + "\n".join(format_string) + "\n)"


class DataAugmentationFinetune_HPA(object):
    def __init__(self, split='train'):

        self.crop_flip =  v2.Compose([
            v2.RandomResizedCrop(
                size=(224, 224),
                scale=(0.75, 1),
                interpolation=InterpolationMode.BILINEAR,),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                ])

        if split=='train':
            transform = v2.Compose([
                v2.RandomResizedCrop(
                    size=(224, 224),
                    scale=(0.75, 1),
                    interpolation=InterpolationMode.BILINEAR,),
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.RandomVerticalFlip(p=0.5),
                RescaleProtein(p=0.25),
                PerChannelColorJitter(brightness=0.5, contrast=0.5, p=1.0),
                v2.RandomChoice(
                    [
                        PerChannelGaussianBlur(kernel_size=7, sigma=(0.1, 2.0), p=0.5),
                        PerChannelAdjustSharpness(sharpness_factor=2, p=0.5),
                    ]),
                GaussianNoise(sigma_range=(0.01, 0.05), p=0.5),
            ])

        elif split=='valid':
            transform = v2.Resize(size=(224, 224), interpolation=InterpolationMode.BILINEAR,)

        self.transform = transform

    def __call__(self, image):
        
        return self.transform(image)



class DataAugmentationiBOT_HPA_CVIT(object):
    def __init__(self, global_crops_scale, local_crops_scale, global_crops_number, local_crops_number, global_crops_size=224,
                 local_crops_size=96,):

        # TODO: WE CAN CHANGE THIS LATER
        self.channel_budget = 2
        self.total_channels = 4

        self.crop_flip =  v2.Compose([
            v2.RandomResizedCrop(
                size=(global_crops_size, global_crops_size),
                scale=global_crops_scale,
                interpolation=InterpolationMode.BILINEAR,),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                ])

        self.global_transfo1 = v2.Compose(
            [
                self.crop_flip,
                v2.RandomChoice(
                [
                    v2.RandomAffine(
                        degrees=90,
                        translate=(0.2, 0.2),
                        scale=(0.8, 1.2),
                        interpolation=InterpolationMode.BILINEAR,
                        fill=0,
                    ),
                    v2.RandomPerspective(
                        distortion_scale=0.25,
                        interpolation=InterpolationMode.BILINEAR,
                        fill=0,
                    ),
                ]),
            ])
        
        self.global_transfo2 = v2.Compose(
            [
                self.crop_flip,
                RescaleProtein(p=0.25),
                PerChannelColorJitter(brightness=0.5, contrast=0.5, p=1.0),
                v2.RandomChoice(
                    [
                        PerChannelGaussianBlur(kernel_size=7, sigma=(0.1, 2.0), p=0.5),
                        PerChannelAdjustSharpness(sharpness_factor=2, p=0.5),
                    ]),
                GaussianNoise(sigma_range=(0.01, 0.05), p=0.5),
            ])

        self.global_crops_number = global_crops_number
        self.local_crops_number = local_crops_number

        # transformation for the local crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            v2.RandomResizedCrop(local_crops_size, scale=local_crops_scale, interpolation=Image.BICUBIC),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            PerChannelColorJitter(brightness=0.5, contrast=0.5, p=1.0),
            v2.RandomChoice(
                [
                    PerChannelGaussianBlur(kernel_size=7, sigma=(0.1, 2.0), p=0.5),
                    PerChannelAdjustSharpness(sharpness_factor=2, p=0.5),
                ]),
        ])

    def __call__(self, image):
        crops = []
        chans = []

        chan = torch.randperm(self.total_channels)[:self.channel_budget]

        crops.append(self.global_transfo1(image)[chan])
        chans.append(chan)

        for _ in range(self.global_crops_number - 1):
            chan = torch.randperm(self.total_channels)[:self.channel_budget]
            chans.append(chan)

            crops.append(self.global_transfo2(image)[chan])

        for _ in range(self.local_crops_number):
            chan = torch.randperm(self.total_channels)[:self.channel_budget]
            chans.append(chan)

            crops.append(self.local_transfo(image)[chan])

        return crops, chans


class DataAugmentationiBOT_HPA(object):
    def __init__(self, global_crops_scale, local_crops_scale, global_crops_number, local_crops_number, global_crops_size=224,
                 local_crops_size=96,):

        self.crop_flip =  v2.Compose([
            v2.RandomResizedCrop(
                size=(global_crops_size, global_crops_size),
                scale=global_crops_scale,
                interpolation=InterpolationMode.BILINEAR,),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                ])

        self.global_transfo1 = v2.Compose(
            [
                self.crop_flip,
                v2.RandomChoice(
                [
                    v2.RandomAffine(
                        degrees=90,
                        translate=(0.2, 0.2),
                        scale=(0.8, 1.2),
                        interpolation=InterpolationMode.BILINEAR,
                        fill=0,
                    ),
                    v2.RandomPerspective(
                        distortion_scale=0.25,
                        interpolation=InterpolationMode.BILINEAR,
                        fill=0,
                    ),
                ]),
            ])
        
        self.global_transfo2 = v2.Compose(
            [
                self.crop_flip,
                RescaleProtein(p=0.25),
                PerChannelColorJitter(brightness=0.5, contrast=0.5, p=1.0),
                v2.RandomChoice(
                    [
                        PerChannelGaussianBlur(kernel_size=7, sigma=(0.1, 2.0), p=0.5),
                        PerChannelAdjustSharpness(sharpness_factor=2, p=0.5),
                    ]),
                GaussianNoise(sigma_range=(0.01, 0.05), p=0.5),
            ])

        self.global_crops_number = global_crops_number
        self.local_crops_number = local_crops_number

        # transformation for the local crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            v2.RandomResizedCrop(local_crops_size, scale=local_crops_scale, interpolation=Image.BICUBIC),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            PerChannelColorJitter(brightness=0.5, contrast=0.5, p=1.0),
            v2.RandomChoice(
                [
                    PerChannelGaussianBlur(kernel_size=7, sigma=(0.1, 2.0), p=0.5),
                    PerChannelAdjustSharpness(sharpness_factor=2, p=0.5),
                ]),
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        for _ in range(self.global_crops_number - 1):
            crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))

        return crops

class DataAugmentationiBOT(object):
    def __init__(self, global_crops_scale, local_crops_scale, global_crops_number, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406, 0.406), (0.229, 0.224, 0.225, 0.225)),
        ])

        self.normalize=normalize

        self.global_crops_number = global_crops_number
        # transformation for the first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            normalize,
        ])
        # transformation for the rest of global crops
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize,
        ])
        # transformation for the local crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        for _ in range(self.global_crops_number - 1):
            crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops
    

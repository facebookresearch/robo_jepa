# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Video transforms for robot video training.

Provides :func:`make_transforms`, the main entry point that returns a
:class:`VideoTransform` callable.  The transform pipeline is:

1. **Random resized crop** — Inception-style area-based random crop followed by
   bilinear resize to ``(crop_height, crop_width)``.
2. **(optional) Resize** — additional bilinear resize of the normalised
   ``buffer`` only (not ``prenorm_buffer``), controlled by the ``resize``
   parameter.
3. **ImageNet normalisation** — per-channel mean/std normalisation applied to
   ``buffer``.  ``prenorm_buffer`` retains raw 0–255 float pixel values.

The callable accepts a numpy/tensor array of shape ``[T, H, W, C]`` and returns
a tuple ``(buffer, prenorm_buffer)`` both in ``[C, T, H, W]`` layout.
``buffer`` is normalised; ``prenorm_buffer`` is not.
"""

import torch

import src.datasets.utils.video.transforms as video_transforms
from src.datasets.utils.video.randerase import RandomErasing


def make_transforms(
    random_horizontal_flip=False,
    random_resize_aspect_ratio=(3 / 4, 4 / 3),
    random_resize_scale=(0.3, 1.0),
    reprob=0.0,
    auto_augment=False,
    motion_shift=False,
    random_crop=False,
    crop_height=256,
    crop_width=256,
    resize=None,
    normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
):
    """Build a :class:`VideoTransform` with the given augmentation settings.

    Args:
        random_horizontal_flip (bool): Enable random horizontal flips.
        random_resize_aspect_ratio (tuple): ``(min, max)`` aspect-ratio range
            for the Inception-style random resized crop.
        random_resize_scale (tuple): ``(min, max)`` area-fraction range for
            the random resized crop.
        crop_height (int): Target height after random resized crop.
        crop_width (int): Target width after random resized crop.
        resize (int or tuple, optional): If set, ``buffer`` (but not
            ``prenorm_buffer``) is further resized to this spatial size after
            normalisation. Useful when the encoder expects a different
            resolution than the diffusion decoder.
        normalize (tuple): ``((mean_r, mean_g, mean_b), (std_r, std_g, std_b))``
            for ImageNet-style normalisation.

    Returns:
        VideoTransform: A callable that maps ``[T, H, W, C]`` uint8/float
        arrays to ``(buffer, prenorm_buffer)`` both ``[C, T, H, W]`` float32.
    """

    _frames_augmentation = VideoTransform(
        random_horizontal_flip=random_horizontal_flip,
        random_resize_aspect_ratio=random_resize_aspect_ratio,
        random_resize_scale=random_resize_scale,
        reprob=reprob,
        resize=resize,
        auto_augment=auto_augment,
        motion_shift=motion_shift,
        crop_height=crop_height,
        crop_width=crop_width,
        normalize=normalize,
        random_crop=random_crop,
    )
    return _frames_augmentation


class VideoTransform(object):

    def __init__(
        self,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(3 / 4, 4 / 3),
        random_resize_scale=(0.3, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_height=224,
        crop_width=224,
        resize=None,
        normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        random_crop=False,
    ):

        self.random_horizontal_flip = random_horizontal_flip
        self.random_resize_aspect_ratio = random_resize_aspect_ratio
        self.random_resize_scale = random_resize_scale
        self.auto_augment = auto_augment
        self.motion_shift = motion_shift
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.resize = resize
        self.mean = torch.tensor(normalize[0], dtype=torch.float32)
        self.std = torch.tensor(normalize[1], dtype=torch.float32)
        if not self.auto_augment:
            # Without auto-augment, PIL and tensor conversions simply scale uint8 space by 255.
            self.mean *= 255.0
            self.std *= 255.0

        self.autoaug_transform = video_transforms.create_random_augment(
            input_size=(crop_height, crop_width),
            # auto_augment="rand-m4-n4-w1-mstd0.5-inc1",
            auto_augment="rand-m7-n4-mstd0.5-inc1",
            interpolation="bicubic",
        )

        self.spatial_transform = (
            video_transforms.random_resized_crop_with_shift if motion_shift else video_transforms.random_resized_crop
        )

        self.reprob = reprob
        self.erase_transform = RandomErasing(
            reprob,
            mode="pixel",
            max_count=1,
            num_splits=1,
            device="cpu",
        )

    def __call__(self, buffer):

        # if self.auto_augment:
        #     buffer = [transforms.ToPILImage()(frame) for frame in buffer]
        #     buffer = self.autoaug_transform(buffer)
        #     buffer = [transforms.ToTensor()(img) for img in buffer]
        #     buffer = torch.stack(buffer)  # T C H W
        #     buffer = buffer.permute(0, 2, 3, 1)  # T H W C
        if torch.is_tensor(buffer):
            # TODO: ensure input is always a tensor?
            buffer = buffer.to(torch.float32)
        else:
            buffer = torch.tensor(buffer, dtype=torch.float32)

        buffer = buffer.permute(3, 0, 1, 2)  # T H W C -> C T H W

        buffer = self.spatial_transform(
            images=buffer,
            target_height=self.crop_height,
            target_width=self.crop_width,
            scale=self.random_resize_scale,
            ratio=self.random_resize_aspect_ratio,
        )
        # random_crop = transforms.RandomResizedCrop(
        #     size=(self.crop_size, self.crop_size),
        #     scale=self.random_resize_scale,
        #     ratio=self.random_resize_aspect_ratio,
        # )
        # buffer = random_crop(buffer)

        # buffer = torch.nn.functional.interpolate(
        #     buffer,
        #     size=(256, 256),
        #     mode="bilinear",
        #     align_corners=False,
        # )

        # if self.random_horizontal_flip:
        #     buffer, _ = video_transforms.horizontal_flip(0.5, buffer)

        prenorm_buffer = buffer.clone()
        if self.resize is not None:
            buffer = torch.nn.functional.interpolate(
                buffer,
                size=(self.resize, self.resize) if not isinstance(self.resize, (list, tuple)) else self.resize,
                mode="bilinear",
                align_corners=False,
            )
        buffer = _tensor_normalize_inplace(buffer, self.mean, self.std)
        if self.reprob > 0:
            buffer = buffer.permute(1, 0, 2, 3)
            # buffer = self.erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer, prenorm_buffer


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:  # noqa: E721
        mean = torch.tensor(mean)
    if type(std) == list:  # noqa: E721
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


def _tensor_normalize_inplace(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize (with dimensions C, T, H, W).
        mean (tensor): mean value to subtract (in 0 to 255 floats).
        std (tensor): std to divide (in 0 to 255 floats).
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()

    C, T, H, W = tensor.shape
    tensor = tensor.view(C, -1).permute(1, 0)  # Make C the last dimension
    tensor.sub_(mean).div_(std)
    tensor = tensor.permute(1, 0).view(C, T, H, W)  # Put C back in front
    return tensor

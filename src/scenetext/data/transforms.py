"""Image transforms for STR training and inference.

Training uses aggressive augmentation to improve generalisation on the small
ICDAR 2003 dataset.  The augmentation pipeline is ordered so that spatial
transforms come *before* conversion to tensor (they operate on PIL images),
while pixel-level erasing operates on the tensor.
"""

from __future__ import annotations

from PIL import Image
from torchvision import transforms as T


class ResizeWithPad:
    """Resize image preserving aspect ratio to target height, then right-pad to target width.

    This avoids stretching short words or compressing long words, preserving character
    proportions for the CNN feature extractor.
    """

    def __init__(self, target_size: tuple[int, int] = (100, 420), fill: int = 128):
        self.target_h, self.target_w = target_size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w_orig, h_orig = img.size
        if h_orig <= 0 or w_orig <= 0:
            return Image.new(img.mode, (self.target_w, self.target_h), color=self.fill)

        scale = min(self.target_h / h_orig, self.target_w / w_orig)
        new_w = min(self.target_w, max(1, int(round(w_orig * scale))))
        new_h = min(self.target_h, max(1, int(round(h_orig * scale))))
        resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        fill = (self.fill,) * len(img.getbands()) if len(img.getbands()) > 1 else self.fill
        canvas = Image.new(img.mode, (self.target_w, self.target_h), color=fill)
        canvas.paste(resized, (0, (self.target_h - new_h) // 2))
        return canvas


def build_transforms(
    img_size: tuple[int, int] = (100, 420),
    train: bool = False,
    keep_aspect_ratio: bool = True,
) -> T.Compose:
    """Build a transform pipeline for word-crop images.

    Args:
        img_size: ``(height, width)`` to resize every crop to.
        train: when *True* add data augmentation; *False* gives deterministic
               pre-processing suitable for validation / inference.
        keep_aspect_ratio: when *True*, scale preserving aspect ratio and pad width.
    """
    h, w = img_size
    resize_op = ResizeWithPad((h, w)) if keep_aspect_ratio else T.Resize((h, w))
    if not train:
        return T.Compose(
            [
                resize_op,
                T.Grayscale(num_output_channels=1),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,)),
            ]
        )
    return T.Compose(
        [
            # --- spatial transforms (PIL) ---
            resize_op,
            T.RandomAffine(degrees=3, translate=(0.03, 0.03), shear=5),
            T.RandomPerspective(distortion_scale=0.3, p=0.4),
            T.RandomRotation(degrees=5),
            # --- colour / intensity transforms (PIL, before grayscale) ---
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
            T.RandomAutocontrast(p=0.3),
            T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            T.RandomInvert(p=0.05),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            # --- convert to single-channel tensor ---
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
            # --- tensor-level augmentation ---
            T.RandomErasing(p=0.25, scale=(0.02, 0.08), ratio=(0.3, 3.3)),
            T.Normalize((0.5,), (0.5,)),
        ]
    )

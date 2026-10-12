from __future__ import annotations

from torchvision import transforms as T


def build_transforms(img_size: tuple[int, int] = (100, 420), train: bool = False) -> T.Compose:
    h, w = img_size
    if not train:
        return T.Compose(
            [
                T.Resize((h, w)),
                T.Grayscale(num_output_channels=1),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,)),
            ]
        )
    return T.Compose(
        [
            T.Resize((h, w)),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            T.Grayscale(num_output_channels=1),
            T.GaussianBlur(3),
            T.RandomAffine(degrees=1, shear=1),
            T.RandomPerspective(distortion_scale=0.3, p=0.5),
            T.RandomRotation(degrees=2),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
        ]
    )

"""
Preprocessing Transforms for Javanese OCR.
"""

import torchvision.transforms as T
from PIL import Image


class ResizeByHeight:
    """
    Resize image to target height while preserving aspect ratio.
    """

    def __init__(self, height: int):
        """
        Args:
            height: Target height in pixels
        """
        if height <= 0:
            raise ValueError(f"height must be positive, got {height}")
        self.height = height

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if h == 0:
            raise ValueError("Image height is 0")

        # Calculate new width maintaining aspect ratio
        new_w = int(w * self.height / h)
        return img.resize((new_w, self.height), Image.BILINEAR)


def get_preprocessing_pipeline(img_height: int) -> T.Compose:
    """
    Create preprocessing pipeline for inference/training.

    Pipeline:
        1. Convert to grayscale (OCR works on single channel)
        2. (Optional) CLAHE contrast enhancement
        3. Resize to target height (preserve aspect ratio)
        4. Convert to tensor [0, 1] normalized

    Args:
        img_height: Target image height (e.g., 48)
        enhance: Apply CLAHE for low-quality images

    Returns:
        Composed torchvision transforms
    """
    if img_height <= 0:
        raise ValueError(f"img_height must be positive, got {img_height}")

    transforms = [
        T.Grayscale(num_output_channels=1),  # RGB/RGBA → Grayscale
    ]

    transforms.extend(
        [
            ResizeByHeight(img_height),  # Resize to target height
            T.ToTensor(),  # Convert to [C, H, W] tensor, normalize to [0, 1]
        ]
    )

    return T.Compose(transforms)

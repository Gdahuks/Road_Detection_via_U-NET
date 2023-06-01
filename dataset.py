import os

import numpy as np
from PIL import Image
from albumentations import Compose
from torch.utils.data import Dataset


class RoadDetectionDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, transform: Compose = None):
        """
        Initialize the RoadDetectionDataset class.

        Args:
            image_dir (str): Directory path containing the input images.
            mask_dir (str): Directory path containing the corresponding masks.
            transform (callable, optional): Transformations to apply to the images and masks. Default is None.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.images)

    def __getitem__(self, index: int) -> (np.ndarray, np.ndarray):
        """
        Get an item from the dataset at the given index.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            (np.ndarray, np.ndarray): A tuple containing the image and mask.
        """
        img_path = os.path.join(self.image_dir, self.images[index])  # img file format "image\d\d\d.bmp"
        mask_path = os.path.join(self.mask_dir, self.images[index].replace("image", ""))    # mask file format "\d\d\d.bmp"
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return image, mask

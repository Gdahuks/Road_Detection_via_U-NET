import sys
import time
from typing import Callable, Optional

import torch
import torchvision
from torch.utils.data import DataLoader

from dataset import RoadDetectionDataset


def save_checkpoint(state: dict, filename: str = "checkpoint.pth.tar") -> None:
    """
    Save a checkpoint to a file.

    Args:
        state (dict): The state dictionary to save.
        filename (str, optional): The filename to save the checkpoint to. Default is "checkpoint.pth.tar".
    """
    print("=> Saving checkpoint")
    start = time.perf_counter()
    torch.save(state, filename)
    end = time.perf_counter()
    print(f"=> Checkpoint saved ({end-start:0.4f}s)")


def load_checkpoint(checkpoint: dict, model: torch.nn.Module) -> None:
    """
    Load a checkpoint into a model.

    Args:
        checkpoint (dict): The checkpoint dictionary to load.
        model (torch.nn.Module): The model to load the checkpoint into.
    """
    print("=> Loading checkpoint")
    start = time.perf_counter()
    model.load_state_dict(checkpoint["state_dict"])
    end = time.perf_counter()
    print(f"=> Checkpoint loaded ({end-start:0.4f}s)")


def get_loaders(
        train_dir: str,
        train_masks_dir: str,
        test_dir: str,
        test_masks_dir: str,
        batch_size: int,
        val_dir: Optional[str] = None,
        val_masks_dir: Optional[str] = None,
        train_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        num_workers: int = 4,
        pin_memory: bool = True) -> (DataLoader, DataLoader, Optional[DataLoader]):
    """
    Create data loaders for training, testing, and validation datasets.

    Args:
        train_dir (str): The directory path of the training images.
        train_masks_dir (str): The directory path of the training masks.
        test_dir (str): The directory path of the testing images.
        test_masks_dir (str): The directory path of the testing masks.
        batch_size (int): The batch size for the data loaders.
        val_dir (str, optional): The directory path of the validation images. Default is None.
        val_masks_dir (str, optional): The directory path of the validation masks. Default is None.
        train_transform (Callable, optional): The transformation function to apply to training data. Default is None.
        test_transform (Callable, optional): The transformation function to apply to testing data. Default is None.
        val_transform (Callable, optional): The transformation function to apply to validation data. Default is None.
        num_workers (int, optional): The number of workers for data loading. Default is 4.
        pin_memory (bool, optional): Whether to pin memory for faster GPU transfer. Default is True.

    Returns:
        (DataLoader, DataLoader, Optional[DataLoader]): The data loaders for training, testing, and validation datasets.
    """
    train_ds = RoadDetectionDataset(train_dir, train_masks_dir, train_transform)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )

    test_ds = RoadDetectionDataset(test_dir, test_masks_dir, test_transform)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )

    val_loader = None
    if val_dir is not None and val_masks_dir is not None:
        val_ds = RoadDetectionDataset(val_dir, val_masks_dir, val_transform)
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory
        )

    return train_loader, test_loader, val_loader


def check_binary_accuracy(loader: DataLoader, model: torch.nn.Module, device: str = "mps"):
    """
    Calculates the binary accuracy and dice score for a given loader and model.

    Args:
        loader (DataLoader): The data loader containing the input samples and labels.
        model (torch.nn.Module): The model used for prediction.
        device (str, optional): The device to run the computation on (default is "mps").
    """
    print("=> Calculating metrics")
    start = time.perf_counter()
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)  # Add 1e-8 to avoid division by zero
    model.train()

    accuracy = num_correct / num_pixels * 100
    dice_score /= len(loader)
    end = time.perf_counter()

    print(f"  Acc        : {accuracy:.2f}%")
    print(f"  Dice score : {dice_score:.4f}")
    print(f"=> End of metrics calculation ({end-start:0.4f}s)")


def save_predictions_as_imgs(loader: DataLoader, model: torch.nn.Module,
                             folder: str = "saved_images/", device: str = "mps"):
    """
    Saves the predicted and ground truth images as PNG files.

    Args:
        loader (DataLoader): The data loader containing the input samples and labels.
        model (torch.nn.Module): The model used for prediction.
        folder (str, optional): The folder path to save the images (default is "saved_images/").
        device (str, optional): The device to run the computation on (default is "mps").
    """
    print("=> Saving image")
    start = time.perf_counter()
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        y = y.to(device=device)

        images = torch.cat((preds, y.unsqueeze(1)), dim=3)
        images = torchvision.utils.make_grid(images, nrow=1)

        torchvision.utils.save_image(images, f"{folder}/image_{idx}.png")
    model.train()
    sys.stdout.flush()
    end = time.perf_counter()
    print(f"=> Image Saved ({end-start:0.4f}s)")

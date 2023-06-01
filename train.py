import albumentations as a
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from model import UNET
from utils import *

# Hyperparameters and/or constants
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 15
NUM_WORKERS = 4
IMAGE_HEIGHT = 112
IMAGE_WIDTH = 160

PIN_MEMORY = True
LOAD_MODEL = False

# Every how many epochs the operation is to be performed, -1 to turn off.
SAVE_IMG_COUNTER = 3
SAVE_MODEL_COUNTER = 3
ACCURACY_COUNTER = 3

TRAIN_IMG_DIR = "data/Train/image/"
TRAIN_MASK_DIR = "data/Train/label/"
TEST_IMG_DIR = "data/Test/image/"
TEST_MASK_DIR = "data/Test/label/"
VAL_IMG_DIR = "data/Validation/image/"
VAL_MASK_DIR = "data/Validation/label/"


def train(loader: DataLoader, model: nn.Module,
          optimizer: torch.optim.Optimizer,
          loss_fn: callable, scaler: GradScaler = None) -> None:
    """
    Trains the model using the provided data loader, optimizer and loss function.

    Args:
        loader (torch.utils.data.DataLoader): Data loader providing the training data.
        model (torch.nn.Module): Model to be trained.
        optimizer (torch.optim.optimizer.Optimizer): Optimization algorithm.
        loss_fn (callable): Loss function to calculate the training loss.
        scaler (torch.cuda.amp.GradScaler, optional): GradScaler for automatic mixed precision training.
    """
    start = time.perf_counter()
    loop = tqdm(loader, mininterval=0.01, leave=False)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # Forward pass
        if scaler is None:
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        else:
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = loss_fn(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    end = time.perf_counter()
    print(f"End of training epoch. ({end-start:0.2f}s, loss={loss.item():4f})")
    sys.stdout.flush()


def main():
    print("--- Initializing ---")
    train_transform = a.Compose([
        a.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        a.Rotate(limit=35, p=1.0),
        a.HorizontalFlip(p=0.5),
        a.VerticalFlip(p=0.1),
        a.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2()])

    test_transform = a.Compose([
        a.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        a.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2()])

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, test_loader, _ = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        batch_size=BATCH_SIZE,
        train_transform=train_transform,
        test_transform=test_transform,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("checkpoint.pth.tar"), model)

    if ACCURACY_COUNTER != -1:
        check_binary_accuracy(test_loader, model, device=DEVICE)

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Starting epoch: {epoch+1} ----")
        train(train_loader, model, optimizer, loss_fn)
        if epoch % SAVE_MODEL_COUNTER == 0 and SAVE_MODEL_COUNTER != -1:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)

        if epoch % ACCURACY_COUNTER == 0 and ACCURACY_COUNTER != -1:
            check_binary_accuracy(test_loader, model, device=DEVICE)
        if epoch % SAVE_IMG_COUNTER == 0 and SAVE_IMG_COUNTER != -1:
            save_predictions_as_imgs(test_loader, model, folder="saved_images/", device=DEVICE)

    if NUM_EPOCHS - 1 % SAVE_MODEL_COUNTER == 0 and SAVE_MODEL_COUNTER != -1:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

    if NUM_EPOCHS - 1 % ACCURACY_COUNTER == 0 and ACCURACY_COUNTER != -1:
        check_binary_accuracy(test_loader, model, device=DEVICE)
    if NUM_EPOCHS - 1 % SAVE_IMG_COUNTER == 0 and SAVE_IMG_COUNTER != -1:
        save_predictions_as_imgs(test_loader, model, folder="saved_images/", device=DEVICE)


if __name__ == "__main__":
    main()

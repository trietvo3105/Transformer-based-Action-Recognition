import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score

from tqdm import tqdm

from src.utils import increment_path

# Script full hand-written but optimized and bug fixed with Cursor


def train_sl_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    """
    Train the model for one epoch in supervised learning mode.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): The dataloader for training.
        optimizer (optim.Optimizer): The optimizer to use.
        criterion (nn.Module): The loss function.
        device (str): The device to train on ('cpu' or 'cuda').

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    total_loss = 0
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    total_loss /= len(dataloader)
    print(f">>> Training Loss: {total_loss:.4f}")
    return total_loss


def validate_sl_one_epoch(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: str
) -> tuple[float, float]:
    """
    Validate the model in supervised learning mode.

    Args:
        model (nn.Module): The model to validate.
        dataloader (DataLoader): The dataloader for validation.
        criterion (nn.Module): The loss function.
        device (str): The device to validate on ('cpu' or 'cuda').

    Returns:
        tuple[float, float]: The average validation loss and accuracy.
    """
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predicted = torch.argmax(outputs, 1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    total_loss /= len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    print(f">>> Validation Loss: {total_loss:.4f}")
    print(f">>> Validation Accuracy: {accuracy:.4f}")
    return total_loss, accuracy


def train_ssl_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    """
    Train the model for one epoch in self-supervised learning mode.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): The dataloader for training.
        optimizer (optim.Optimizer): The optimizer to use.
        criterion (nn.Module): The loss function.
        device (str): The device to train on ('cpu' or 'cuda').

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    total_loss = 0
    for inputs1, inputs2 in tqdm(dataloader, desc="Training"):
        inputs1, inputs2 = inputs1.to(device), inputs2.to(device)
        optimizer.zero_grad()
        p1, p2, z1, z2 = model(inputs1, inputs2)
        loss = criterion(p1, p2, z1, z2)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    total_loss /= len(dataloader)
    print(f">>> Training Loss: {total_loss:.4f}")
    return total_loss


def validate_ssl_one_epoch(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: str
) -> tuple[float, None]:
    """
    Validate the model in self-supervised learning mode.

    Args:
        model (nn.Module): The model to validate.
        dataloader (DataLoader): The dataloader for validation.
        criterion (nn.Module): The loss function.
        device (str): The device to validate on ('cpu' or 'cuda').

    Returns:
        tuple[float, None]: The average validation loss and None for accuracy.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs1, inputs2 in tqdm(dataloader, desc="Validating"):
            inputs1, inputs2 = inputs1.to(device), inputs2.to(device)
            p1, p2, z1, z2 = model(inputs1, inputs2)
            loss = criterion(p1, p2, z1, z2)
            total_loss += loss.item()
    total_loss /= len(dataloader)
    print(f">>> Validation Loss: {total_loss:.4f}")
    return total_loss, None


TRAIN_FNS = {
    "sl": train_sl_one_epoch,
    "ssl": train_ssl_one_epoch,
}

VALIDATE_FNS = {
    "sl": validate_sl_one_epoch,
    "ssl": validate_ssl_one_epoch,
}


def train(
    mode: str,
    model: nn.Module,
    model_and_data_config: dict,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    num_epochs: int,
    device: str,
    save_model_path: str,
    log_dir: str = "runs/exp",
) -> None:
    """
    Train a model and validate it.

    Args:
        mode (str): The mode to train in ("sl" or "ssl")
        model (nn.Module): The model to train.
        model_and_data_config (dict): The model's configuration arguments.
        train_dataloader (DataLoader): The dataloader for training.
        val_dataloader (DataLoader): The dataloader for validation.
        optimizer (optim.Optimizer): The optimizer to use.
        criterion (nn.Module): The loss function.
        num_epochs (int): The number of epochs to train for.
        device (str): The device to train on ('cpu' or 'cuda').
        save_model_path (str): The path to save the best model to.
        log_dir (str): The directory to save tensorboard logs to.

    Returns:
        None
    """
    best_val_loss = float("inf")

    log_dir = increment_path(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Tensorboard logs will be saved to {log_dir}")

    final_lr_factor = 0.01
    lr_lambda = lambda epoch: 1 - (1 - final_lr_factor) * epoch / num_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    model.to(device)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs-1}")
        train_loss = TRAIN_FNS[mode](
            model, train_dataloader, optimizer, criterion, device
        )
        validation_results = VALIDATE_FNS[mode](
            model, val_dataloader, criterion, device
        )
        scheduler.step()

        val_loss, val_accuracy = validation_results

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        if val_accuracy is not None:
            writer.add_scalar("Metrics/accuracy", val_accuracy, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "config": model_and_data_config,
                    "state_dict": model.state_dict(),
                },
                save_model_path,
            )
            print(f"New best model saved to {save_model_path}")
        print("--------------------------------\n")

    writer.close()

"""Training and evaluation loops, separated from experiment configuration."""
import torch
from torch import nn


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = correct = total = 0
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        size = targets.size(0)
        total_loss += loss.item() * size
        correct += (logits.argmax(1) == targets).sum().item()
        total += size
    return {'loss': total_loss / total, 'accuracy': 100 * correct / total}


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = correct = total = 0
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        logits = model(images)
        size = targets.size(0)
        total_loss += criterion(logits, targets).item() * size
        correct += (logits.argmax(1) == targets).sum().item()
        total += size
    return {'loss': total_loss / total, 'accuracy': 100 * correct / total}

"""Evaluate a selected checkpoint once on the held-out CIFAR-10 test set."""
import argparse
import json
import torch
from src.data import make_loaders
from src.engine import evaluate
from src.model import VGG6
from train import seed_everything


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    args = p.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    seed_everything(config['seed'])
    _, _, test = make_loaders(config['batch_size'], config['seed'],
                              workers=config['workers'], val_size=config['val_size'],
                              use_cutout=not config['no_cutout'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VGG6(activation=config['activation'], batch_norm=not config['no_batch_norm'],
                 channels=tuple(config['channels'])).to(device)
    model.load_state_dict(checkpoint['model_state'])
    metrics = evaluate(model, test, device)
    print(json.dumps({'test_top1_accuracy_percent': metrics['accuracy'],
                      'test_cross_entropy_loss': metrics['loss'],
                      'selected_validation_accuracy_percent': checkpoint['best_val_acc'],
                      'selected_epoch': checkpoint['best_epoch']}, indent=2))


if __name__ == '__main__':
    main()

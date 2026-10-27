"""Train one experiment; select checkpoint by validation top-1 accuracy only."""
import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import optim

from src.data import make_loaders
from src.engine import evaluate, train_one_epoch
from src.model import VGG6


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def make_optimizer(name, parameters, lr, weight_decay, momentum):
    if name == 'sgd':
        return optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == 'nesterov':
        if momentum <= 0:
            raise ValueError('Nesterov requires momentum > 0')
        return optim.SGD(parameters, lr=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    if name == 'adam':
        return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name == 'adagrad':
        return optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    if name == 'rmsprop':
        return optim.RMSprop(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == 'nadam':
        return optim.NAdam(parameters, lr=lr, weight_decay=weight_decay)
    raise ValueError(f'Unknown optimizer {name}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=None, help='Optional JSON config with matching keys')
    p.add_argument('--activation', choices=['relu', 'sigmoid', 'tanh', 'silu', 'gelu'], default='relu')
    p.add_argument('--optimizer', choices=['sgd', 'nesterov', 'adam', 'adagrad', 'rmsprop', 'nadam'], default='adam')
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight-decay', type=float, default=0.0005)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val-size', type=int, default=5000)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--no-cutout', action='store_true')
    p.add_argument('--no-batch-norm', action='store_true')
    p.add_argument('--channels', nargs=2, type=int, default=[64, 128])
    p.add_argument('--run-name', default='baseline')
    p.add_argument('--out-dir', default='runs')
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--wandb-project', default='vgg6-cifar10-assignment')
    args = p.parse_args()
    if args.config:
        config = json.loads(Path(args.config).read_text())
        for key, value in config.items():
            if not hasattr(args, key):
                raise ValueError(f'Unknown config key: {key}')
            setattr(args, key, value)
    if args.epochs < 1 or args.lr <= 0 or args.batch_size < 1:
        raise ValueError('epochs, lr and batch_size must be positive')
    return args


def main():
    args = parse_args()
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.run_name,
                               config=vars(args))
        # W&B sweep overrides: use wandb.config as the effective configuration.
        for key in ('activation', 'optimizer', 'lr', 'batch_size', 'epochs',
                    'momentum', 'weight_decay', 'seed'):
            setattr(args, key, wandb.config[key])
        wandb.define_metric('epoch')
        for key in ('train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr_current'):
            wandb.define_metric(key, step_metric='epoch')
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_dir = Path(args.out_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.json').write_text(json.dumps(vars(args), indent=2))
    train_loader, val_loader, _ = make_loaders(args.batch_size, args.seed,
                                               workers=args.workers, val_size=args.val_size,
                                               use_cutout=not args.no_cutout)
    model = VGG6(activation=args.activation, batch_norm=not args.no_batch_norm,
                 channels=tuple(args.channels)).to(device)
    optimizer = make_optimizer(args.optimizer, model.parameters(), args.lr,
                               args.weight_decay, args.momentum)
    # Constant LR makes LR comparisons straightforward; change explicitly in experiments.
    best_acc, best_epoch = -1.0, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        record = {'epoch': epoch, 'train_loss': train_metrics['loss'],
                  'train_acc': train_metrics['accuracy'], 'val_loss': val_metrics['loss'],
                  'val_acc': val_metrics['accuracy'], 'lr_current': optimizer.param_groups[0]['lr']}
        history.append(record)
        if record['val_acc'] > best_acc:
            best_acc, best_epoch = record['val_acc'], epoch
            torch.save({'model_state': model.state_dict(), 'config': vars(args),
                        'best_val_acc': best_acc, 'best_epoch': best_epoch}, run_dir / 'best.pt')
        if wandb_run:
            import wandb
            wandb.log(record)
        print(f"epoch {epoch:03d}/{args.epochs} train_loss={record['train_loss']:.4f} "
              f"train_acc={record['train_acc']:.2f}% val_loss={record['val_loss']:.4f} "
              f"val_acc={record['val_acc']:.2f}%")
    with (run_dir / 'history.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    summary = {'run_name': args.run_name, 'best_val_acc': best_acc, 'best_epoch': best_epoch,
               'last_val_acc': history[-1]['val_acc'], 'wall_seconds': time.perf_counter() - started}
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if wandb_run:
        import wandb
        wandb.summary.update(summary)
        wandb.finish()


if __name__ == '__main__':
    main()

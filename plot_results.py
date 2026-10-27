"""Make local plots from run histories; W&B produces required online sweep plots."""
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    args = p.parse_args()
    folder = Path(args.run_dir)
    with (folder / 'history.csv').open() as file:
        history = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(file)]
    for metric, ylabel in [('train_loss', 'Training loss'), ('train_acc', 'Training accuracy (%)'),
                           ('val_loss', 'Validation loss'), ('val_acc', 'Validation accuracy (%)')]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([row['epoch'] for row in history], [row[metric] for row in history])
        ax.set(xlabel='Epoch', ylabel=ylabel, title=ylabel)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(folder / (metric + '.png'), dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter([row['epoch'] for row in history], [row['val_acc'] for row in history])
    ax.set(xlabel='Epoch (evaluation step)', ylabel='Validation accuracy (%)',
           title='Validation accuracy vs evaluation step')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(folder / 'val_acc_scatter.png', dpi=160)
    plt.close(fig)
    print('Saved five local plots in', folder)


if __name__ == '__main__':
    main()

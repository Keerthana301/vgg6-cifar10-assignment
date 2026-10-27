# CS6886W Assignment 1 — VGG6 on CIFAR-10

## 1. Project Overview

This project implements a modular VGG6 convolutional neural network (CNN) using PyTorch for image classification on the CIFAR-10 dataset.

The assignment explores model training, activation functions, optimizers, learning rates, batch sizes, epoch budgets, validation-based model selection, and final test evaluation.

**GitHub Repository:**  
https://github.com/Keerthana301/vgg6-cifar10-assignment

## 2. Project Objectives

- Implement and train a VGG6 CNN for CIFAR-10 image classification.
- Evaluate training and validation performance over multiple epochs.
- Compare activation functions, optimizers, and hyperparameters through controlled experiments.
- Visualize training and validation loss and accuracy.
- Select a model using validation accuracy and evaluate the selected checkpoint on the test dataset.
- Maintain reproducible code, configurations, and experiment results.

## 3. Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- CIFAR-10 dataset
- Google Colab
- Weights & Biases (W&B), for experiment tracking when enabled

## 4. Repository Structure

```text
vgg6-cifar10-assignment/
├── configs/
│   ├── baseline.json
│   └── sweep.yaml
├── notebooks/
│   └── VGG6_Assignment1_Training.ipynb
├── results/
│   ├── config.json
│   ├── history.csv
│   ├── summary.json
│   ├── train_acc.png
│   ├── train_loss.png
│   ├── val_acc.png
│   ├── val_acc_scatter.png
│   └── val_loss.png
├── src/
├── .gitignore
├── plot_results.py
├── README.md
├── requirements.txt
├── test_best.py
└── train.py
```

The `runs/` directory is generated during training and stores experiment outputs and checkpoints. It may not be present in the repository until an experiment is run or its selected artifacts are added.

## 5. Dataset and Preprocessing

The project uses CIFAR-10, which contains 60,000 color images across 10 classes.

The dataset is divided into:

- Training: 45,000 images
- Validation: 5,000 images
- Test: 10,000 images

The training pipeline uses random cropping with padding, random horizontal flipping, tensor conversion, channel normalization, and cutout augmentation. Validation and test images use tensor conversion and normalization without random training augmentation.

## 6. Model Architecture

The VGG6 implementation uses the following feature stages:

1. Two convolutional layers with 64 channels, batch normalization, and the selected activation function.
2. Max pooling.
3. Two convolutional layers with 128 channels, batch normalization, and the selected activation function.
4. Max pooling.
5. Global average pooling.
6. A fully connected layer producing 10 class scores.

The implementation supports configurable activation functions and optimizers. Refer to the source code and configuration files for the exact implementation and available settings.

## 7. Setup and Installation

### Requirements

Python 3.10 or 3.11 is the intended environment. A compatible GPU is recommended, but CPU execution is also supported.

Clone the repository:

```bash
git clone https://github.com/Keerthana301/vgg6-cifar10-assignment.git
cd vgg6-cifar10-assignment
```

Create and activate a virtual environment.

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Verify the PyTorch installation:

```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

For GPU execution, install a PyTorch build compatible with your system and CUDA environment if required. Refer to the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).

## 8. Q1 — Baseline Experiment

Run the baseline configuration:

```bash
python train.py --config configs/baseline.json
```

Generate the training and validation plots:

```bash
python plot_results.py --run-dir runs/baseline
```

The baseline configuration uses the following parameters:

| Parameter | Value |
|---|---|
| Model | VGG6 |
| Activation | ReLU |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 128 |
| Epochs | 30 |
| Random seed | 42 |

### Baseline Results

The completed baseline experiment produced the following validation results:

| Metric | Result |
|---|---:|
| Best validation accuracy | 79.58% |
| Best epoch | 27 |
| Final validation accuracy | 77.18% |
| Training epochs | 30 |

The `results/` directory contains the saved configuration, training history, summary, and plots from this baseline run.

These are validation results, not test results. Test accuracy and test loss must be reported separately after evaluating the selected checkpoint.

## 9. Q2 / Q3 — Hyperparameter Experiments

The project supports experiments involving activation functions, optimizers, learning rates, batch sizes, and epoch budgets.

### Weights & Biases Setup

Install W&B if it is not already included in the project dependencies:

```bash
python -m pip install wandb
```

Log in:

```bash
wandb login
```

Initialize the sweep:

```bash
wandb sweep configs/sweep.yaml
```

Copy the sweep ID returned by W&B and start an agent:

```bash
!wandb agent keerthana_cs25m512-indian-institute-of-technology-madras/vgg6-cifar10-assignment/v382ji5p --count 20
```

Replace the placeholders with the actual entity, project, and sweep ID.

A 20-run Bayesian sweep does not guarantee that every activation function and optimizer combination will be evaluated. For controlled comparisons, keep other parameters fixed and change one factor at a time.

Example experiments:

```bash
python train.py --activation gelu --optimizer adam --lr 0.001 --batch-size 128 --epochs 30 --run-name gelu_adam --wandb
```

```bash
python train.py --activation relu --optimizer nesterov --lr 0.01 --batch-size 128 --epochs 30 --run-name relu_nesterov --wandb
```

```bash
python train.py --activation relu --optimizer adam --lr 0.001 --batch-size 64 --epochs 30 --run-name batch64 --wandb
```

Run these commands only if the corresponding options are supported by the implementation. Record the exact parameters and actual outcomes of every experiment performed.

W&B can be used to visualize loss, accuracy, and hyperparameter comparisons. Include exported charts in the assignment report if required.

**Experiment status:** The baseline results are documented above. Additional comparisons and W&B sweep results should be documented only after the corresponding experiments have actually been completed.

## 10. Q4 — Final Model Selection and Test Evaluation

Select the model configuration using the best validation accuracy. Record the chosen parameters, random seed, and selected epoch.

Evaluate the selected checkpoint on the test dataset:

```bash
python test_best.py --checkpoint runs/baseline/best.pt
```

This example assumes that the selected checkpoint exists at the specified path. If a different experiment is selected, replace the path with that experiment's checkpoint.

Report the following in the final assignment:

- Selected model configuration
- Best validation accuracy
- Selected epoch
- Test top-1 accuracy
- Test cross-entropy loss
- Checkpoint path and experiment identifier

Use the test dataset for final evaluation only. Do not select hyperparameters based on test accuracy.

## 11. Q5 — Reproducibility

The project uses a fixed random seed of 42 by default, and the dataset split uses a NumPy random generator seeded with 42.

For reproducibility, record:

- Python, PyTorch, and Torchvision versions
- Operating system and GPU model
- CUDA version, where applicable
- Configuration files and exact training commands
- Random seeds and experiment identifiers
- W&B sweep ID and URL, if W&B is used
- Validation and test metrics from actual runs

Training results may vary slightly across hardware, software versions, and nondeterministic operations.

The selected model checkpoint should be retained and made accessible to the evaluator if required by the assignment. If a checkpoint exceeds GitHub's normal file-size limits, use Git LFS or an accessible release artifact.

## 12. Results and Visualizations

The `results/` directory contains the baseline experiment artifacts:

- `history.csv` — training and validation metrics by epoch
- `summary.json` — experiment summary
- `config.json` — recorded configuration
- `train_loss.png` — training loss curve
- `train_acc.png` — training accuracy curve
- `val_loss.png` — validation loss curve
- `val_acc.png` — validation accuracy curve
- `val_acc_scatter.png` — validation accuracy scatter plot

These artifacts support the analysis of learning progress and validation performance.

## 13. Conclusion

This project provides a modular implementation of VGG6 for CIFAR-10 image classification. The baseline model was trained for 30 epochs and achieved a best validation accuracy of 79.58% at epoch 27.

The training history and visualization plots provide evidence of the model's learning behavior. Additional hyperparameter comparisons and final test metrics should be included only after their corresponding experiments have been completed.

## 14. Assignment Submission

The assignment submission consists of:

1. The GitHub repository containing the source code, configurations, notebook, and available experiment artifacts.
2. A separate PDF report containing the question-wise discussion, actual experimental results, relevant plots, and the GitHub repository URL.

**GitHub Repository:**  
https://github.com/Keerthana301/vgg6-cifar10-assignment

All reported metrics must be based on actual experimental outputs. No unmeasured or fabricated results are included.

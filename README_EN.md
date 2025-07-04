# SimCLR Cell Image Self-Supervised Learning Project

A self-supervised learning system based on SimCLR for cell image feature learning.

## Features

- **Self-Supervised Pre-training**
  - Feature learning using SimCLR framework
  - Automatic Mixed Precision (AMP) training
  - Support for multiple data augmentation methods
  - Configurable learning rate scheduling
  - Automatic checkpoint saving every 10 epochs
  - Best model auto-saving

- **Downstream Tasks**
  - K-fold cross-validation support
  - Two-stage training strategy
  - Configurable data augmentation
  - Detailed training metrics logging
  - Automatic confusion matrix and training curve generation

## Installation Requirements

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

## Directory Structure

```
SimCLR/
├── config.yaml                 # Configuration file
├── requirements.txt           # Dependency list
├── downstream/               # Downstream task dataset
├── pretrain/                # Pre-training dataset
├── experiments_results_[timestamp]/  # Experiment results directory
│   ├── batch_8/            # Results for batch_size=8
│   │   ├── best_model/     # Best model checkpoint
│   │   │   └── best_model.ckpt
│   │   └── downstream_results/  # Downstream task results for this batch
│   │       ├── logs/       # Training logs
│   │       ├── plots/      # Training curves and confusion matrices
│   │       └── training_results.txt  # Training results summary
│   ├── batch_16/           # Results for batch_size=16
│   │   ├── best_model/
│   │   └── downstream_results/
│   └── ...                 # Results for other batch sizes
└── src/                    # Source code directory
```

## Usage

### 1. Data Preparation

First, run split.py to divide the dataset into pre-training and downstream task sets:

```bash
python split.py
```

### 2. Pre-training Phase

Run SimCLR pre-training with specified batch sizes:

```bash
# Using single batch size
python simclr_schedule.py --batch_size 32

# Using multiple batch sizes sequentially
python simclr_schedule.py --batch_size 16 32 64
```

### 3. Downstream Task Training

Run downstream task training with K-fold cross-validation and data augmentation support:

```bash
# Basic training
python downstream_integrated.py

# Using 5-fold cross-validation and 3x data augmentation
python downstream_integrated.py --k-fold 5 --augment 3
```

## Configuration

Configure the following parameters in `config.yaml`:

```yaml
base:
  seed: 42              # Random seed

data:
  batch_size: 88        # Batch size
  num_workers: 4        # Number of data loading workers
  valid_size: 0.2       # Validation set ratio
  input_shape: [224, 224]  # Input image dimensions
  strength: 0.5         # Data augmentation strength

training:
  epochs: 200             # Number of training epochs
  learning_rate: 0.001  # Learning rate
  weight_decay: 1e-4    # Weight decay
  temperature: 0.4     # Contrastive learning temperature
  early_stopping_patience: 10  # Early stopping patience
```

## License

This project is licensed under the MIT License. 
# Heuristic-Driven Hyperparameter Optimization Algorithms

![CI](https://github.com/outsidermm/Heuristic-driven-Hyperparameter-Optimization-Algorithms/actions/workflows/ci.yml/badge.svg)

Welcome to the GitHub repository for the paper "Heuristic-driven Hyperparameter Optimization Algorithms." This project explores heuristic-based hyperparameter optimization (HPO) algorithms, offering an efficient alternative to traditional methods like Grid Search and Bayesian Optimization. The proposed algorithms are designed to balance computational efficiency and model performance, making them accessible to both novice and experienced machine learning practitioners.

## Table of Contents

1. [Abstract](#abstract)
2. [Introduction](#introduction)
3. [Heuristic HPO Algorithms](#heuristic-hpo-algorithms)
4. [Experiments](#experiments)
5. [Results](#results)
6. [Future Work](#future-work)
7. [Conclusion](#conclusion)
8. [Acknowledgements](#acknowledgements)
9. [Installation](#installation)

## Abstract

This project presents heuristic-driven algorithms for hyperparameter optimization, designed to reduce computational effort while maintaining near-optimal model performance. The algorithms are evaluated using image classification tasks on the CIFAR-100 dataset and compared against the traditional Grid Search technique.

## Introduction

Machine learning models' performance heavily relies on the optimization of hyperparameters (Hyper_Params). Traditional methods like Grid Search and Bayesian Optimization are computationally expensive. This project introduces heuristic-based algorithms that incorporate the systematic nature of Grid Search but use heuristics to focus on areas of higher importance, thus reducing computational time.

## Heuristic HPO Algorithms

### Algorithm Overview

The heuristic algorithms proposed include:

1. **Epoch Tuning Algorithm**: Uses a binary search-inspired approach to find the optimal number of epochs.
2. **Batch Size Tuning Algorithm**: Adjusts the batch size to find the maximum size that avoids significant performance drops.
3. **Learning Rate Tuning Algorithm**: Utilizes a binary search framework to identify the optimal learning rate.

### Key Features

- **Model Overhead Metric**: Combines training time and accuracy into a single metric, allowing for configurable weights based on user preferences.
- **Exploration Factor**: Defines the granularity of the search, balancing between detailed exploration and computational efficiency.

## Experiments

The experiments were conducted using the CIFAR-100 dataset with a modified VGG16 architecture. The baseline performance was established using Grid Search, and the heuristic algorithms were evaluated for:

- Training time
- Model accuracy
- Number of iterations

### Experimental Setup

- **Dataset**: CIFAR-100
- **Model Architecture**: Modified VGG16
- **Hardware**: Dual 12-core Intel Xeon E5-2650 v5 processors, 4 NVIDIA GV100GL GPUs

## Results

### Epoch Tuning

| HPO Algorithm    | Optimal Epoch | Model Overhead | Model Accuracy (%) | Training Time (seconds) |
|------------------|---------------|----------------|---------------------|-------------------------|
| Grid Search      | 70            | 0.3693         | 26.84               | 691                     |
| Heuristic Tuning | 40            | 0.3608         | 18.16               | 419                     |

### Batch Size Tuning

| HPO Algorithm    | Optimal Batch Size | Model Accuracy (%) | Training Time (seconds) |
|------------------|---------------------|---------------------|-------------------------|
| Grid Search      | 1024                | 31.0                | 1437                    |
| Heuristic Tuning | 2048                | 20.8                | 1416                    |

### Learning Rate Tuning

| HPO Algorithm    | Optimal Learning Rate |
|------------------|------------------------|
| Grid Search      | 0.01                   |
| Heuristic Tuning | 0.01                   |

### Performance Comparison

| Hyper_Param  | Metric              | Grid Search | Heuristic Tuning |
|--------------|---------------------|-------------|------------------|
| Epoch        | Number of Iterations| 15          | 8                |
|              | Total Time Cost     | 11847       | 4349             |
| Batch Size   | Number of Iterations| 5           | 5                |
|              | Total Time Cost     | 22483       | 18439            |
| Learning Rate| Number of Iterations| 7           | 5                |
|              | Total Time Cost     | 16782       | 11790            |

## Future Work

Future studies should evaluate the robustness of these heuristic HPO algorithms under various conditions, including:

- Different dataset sizes
- Different dataset natures
- Different model architectures

Further research could also explore the impact of ML optimization techniques such as pruning, quantization, and parallelism on the reliability and accuracy of these heuristic-driven tuning algorithms.

## Conclusion

This project presents three heuristic-driven algorithms for hyperparameter optimization, compared against Grid Search on the CIFAR-100 dataset. The results show that heuristic algorithms can significantly reduce the time needed for hyperparameter search while maintaining competitive model performance, making them suitable for entry-level optimization in ML communities. The flexibility in configuring the overhead weighting according to personal preferences also highlights the practical applicability of these algorithms.

## Acknowledgements

We would like to thank Professor Seda Memik from the University of Northwestern for providing mentoring and feedback. We also thank Professor Manyi Wang and his research team from the Nanjing University of Information Science and Technology for their technical support.

## Installation

**With uv (recommended):**
```bash
uv add heuristic-hpo
# TensorFlow / Keras are optional — install when you need to run training:
uv add "heuristic-hpo[tf]"
```

**With pip:**
```bash
pip install heuristic-hpo
pip install "heuristic-hpo[tf]"  # includes TensorFlow + Keras
```

**From source:**
```bash
git clone https://github.com/outsidermm/Heuristic-driven-Hyperparameter-Optimization-Algorithms.git
cd Heuristic-driven-Hyperparameter-Optimization-Algorithms
uv sync --extra tf
```

## Quick Start

```python
from utility.dataloader import DataLoader
from algorithm import EpochTuner, BatchSizeTuner, LrTuner

# Load your dataset (expects ./dataset/cifar100/{X,y}_{train,test}.npy)
loader = DataLoader("cifar100")
train_ds, val_ds, test_ds = loader.load_dataset()

# Find the most cost-efficient epoch count between 10 and 250
tuner = EpochTuner("cifar100", left_bound=10, right_bound=250, exploration_factor=5)
best_epoch, accuracy, time_taken = tuner.binary_search_efficient_epoch()
print(f"Best epoch: {best_epoch}, accuracy: {accuracy:.4f}, time: {time_taken:.1f}s")

# Find the largest safe batch size (log2 scale: 4=16, 12=4096)
bs_tuner = BatchSizeTuner("cifar100", left_bound=4, right_bound=12, acceptable_range=0.30)
best_bs, accuracy, time_taken = bs_tuner.search()
print(f"Best batch size: {best_bs}")

# Find the optimal learning rate (searches 10^-1 to 10^-7)
lr_tuner = LrTuner("cifar100", left_bound=1, right_bound=7, local_extrema_allowance=0.05)
lr_exp, accuracy, time_taken = lr_tuner.search()
print(f"Best learning rate: 1e-{lr_exp}, accuracy: {accuracy:.4f}")
```

## Development

```bash
uv sync --group dev       # install dev tools (pytest, ruff)
uv run pytest tests/ -v   # run tests (TF-independent tests run without --extra tf)
uv run ruff check .       # lint
uv build                  # build wheel
```

# TRM - Tiny Recursive Model

This repository is an unofficial implementation of the paper [**"Less is More: Recursive Reasoning with Tiny Networks"**](https://arxiv.org/abs/2510.04871).

### Project Status
- **WIP:** This project is currently a work in progress. There are a lot of hardcoded params still. In general, the structure, configuration, and available tasks are subject to change.
- **Implemented Tasks:** "Naive" Sudoku

### Prerequisites
- Python 3.11+

### Installation

It is highly recommended to create and activate a virtual environment first:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
````

Install the project and its dependencies in editable mode:

```bash
pip install -e .
```

### Usage

#### 1\. Training

To start training the model on the Sudoku task, run the main script:

```bash
python -m trm.main
```

You should see training logs printed to the console, and wandb logs (you need to configure wandb first). It should converge after around 6-7 epochs. The ACT (adaptive computation time) mechanism starts to kick in shortly after convergence (before that, the Q-values of the ACT will stay very low).

#### 2\. Configuration

Training parameters (e.g., learning rate, batch size, number of recursive steps) can be modified in the `trm.config.py` file (subject to change).


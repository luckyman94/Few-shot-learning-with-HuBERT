# Evaluation

This folder contains utility functions used to evaluate models and visualize results across supervised and few-shot learning experiments.

The modules are designed to be lightweight and reusable from the experimental notebooks.

---

## Contents

- `evaluate.py`  
  High-level evaluation routines used to compute performance metrics on a dataset or dataloader.

- `metrics.py`  
  Common evaluation metrics such as accuracy, F1 score and related helpers.

- `confusion.py`  
  Utilities to compute and display confusion matrices.

- `tsne.py`  
  Visualization tools for embedding analysis using t-SNE projections.

---

## Usage

The evaluation utilities are typically imported and used inside the notebooks after training or few-shot evaluation steps.




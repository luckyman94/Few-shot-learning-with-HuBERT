# Methods

This folder contains the learning methods implemented in the project, including few-shot learning approaches and supervised fine-tuning baselines.

The code is organized to clearly separate **few-shot episodic methods** from
**standard supervised training**.

---

## Structure

### `fewshot/`

Implements Prototypical Networks and utilities for episodic few-shot learning and for the few shot for inference.

- `train.py`  
  Episodic prototypical training with optional batching and embedding caching.

- `benchmark.py`  
  Few-shot evaluation routines using episodic sampling.

- `prototypical.py`  
  Core prototypical operations (prototype computation and classification).

- `sampling.py`  
  Utilities to sample few-shot episodes and tasks.

---

### `finetuning/`

Implements supervised training and inference baselines for comparison.

- `training.py`  
  Standard supervised training loops.

- `inference.py`  
  Model inference and evaluation for fine-tuned models.


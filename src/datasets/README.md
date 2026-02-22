# Datasets

This folder contains all dataset definitions used in the project.

Each dataset is implemented as a PyTorch-compatible class returning
`(waveform, label)` pairs and can be used interchangeably in the experimental
notebooks.

The datasets are designed to support both few-shot learning and
supervised fine-tuning experiments.

---

## Dataset implementations

- `bird_dog_cat.py`  
  Animals audio dataset (birds, dogs, cats).

- `urban_sound_8k.py`  
  UrbanSound8K environmental sound dataset.

- `speech_commands.py`  
  Google Speech Commands dataset.

- `timit.py`  
  TIMIT speech dataset.

- `crema.py`  
  CREMA-D emotion recognition dataset.

- `snoring_dataset.py`  
  Snoring / non-snoring audio dataset.

- `noise_dataset.py`  
  Synthetic noisy audio dataset.

- `harmonics_dataset.py`  
  Synthetic harmonic audio dataset.


---

## Dataset infrastructure

- `registry.py`  
  Central registry listing all available datasets and their configuration.

- `factory.py`  
  Factory functions used to instantiate datasets from the registry.

- `split.py`  
  Utilities to split datasets by classes for few-shot experiments.

- `utils.py`  
  Shared helper functions used across dataset implementations.

---

## Design principles

- All datasets return `(waveform, label)` pairs.
- Audio is resampled, normalized and optionally truncated.
- Labels are represented as integer class indices.
- Datasets can be used for:
  - episodic few-shot training
  - few-shot evaluation on unseen classes
  - supervised fine-tuning baselines

---

## Few-shot setting

For few-shot experiments, datasets are split at the class level:
- training and test sets contain disjoint classes
- episodes are sampled dynamically during training and evaluation

This follows the canonical few-shot learning protocol.

---

## ⚠️ Notes

- Some datasets rely on external sources (HuggingFace, Kaggle).
- Dataset download and setup are handled inside the notebooks.
# Prototypical Few-Shot Learning with HuBERT

## Overview

This project was developed as part of the Deep Learning course of the MVA program
at ENS Paris-Saclay and focuses on few-shot audio classification using
[HuBERT](https://arxiv.org/abs/2106.07447) and Prototypical Networks.

It reimplements the Prototypical Networks approach proposed by
[Snell et al.](https://arxiv.org/abs/1703.05175) and applies it to audio classification,
comparing few-shot episodic learning with standard supervised fine-tuning.

It reimplements the Prototypical Networks approach proposed by Snell et al. and applies it to audio classification, comparing few-shot episodic learning with standard supervised fine-tuning.

---

## Method overview

We follow the **Prototypical Networks** approach for few-shot learning. Audio signals are first encoded using a pretrained **HuBERT** model, which is kept frozen throughout all experiments.

In each few-shot episode, a small support set is sampled from a subset of classes. A prototype is computed for each class as the average of the support embeddings. Query samples are then classified by assigning them to the nearest prototype using Euclidean distance.

Training is performed episodically over many such episodes.
A  projection head is learned on top of the HuBERT embeddings to shape
the metric space used for prototype-based classification.

To speed up training and evaluation, HuBERT embeddings are precomputed and cached. All episodic learning is then carried out directly in the embedding space.

## Project structure

The project is organized into modular components to separate datasets, methods, evaluation utilities and experimental notebooks. A README.md is available in each folder to describe it.

## Datasets

The project includes a variety of real-world and synthetic audio datasets to evaluate few-shot learning across different audio domains.

Real-world datasets include environmental sounds, speech commands, speaker and emotion recognition tasks. Synthetic datasets are used to analyze robustness and frequency-based patterns under controlled conditions.

The main datasets used in this project are:
- Animals (dogs, cats, birds)
- UrbanSound8K
- Speech Commands
- TIMIT
- CREMA-D
- Snoring detection
- Synthetic noise and harmonic datasets

All datasets are processed into waveform–label pairs and split at the **class level**
for few-shot experiments, ensuring that training and test sets contain disjoint classes.

## Experiments

Three types of experiments are conducted in this project.

**Few-shot inference**  
In this setting, HuBERT is used as a frozen encoder and class prototypes are computed directly from support examples without any additional training. Query samples are classified by nearest-prototype matching.

**Prototypical training**  
Here, a projection head is trained episodically using the Prototypical
Networks framework. HuBERT remains frozen and training is performed by minimizing a distance-based loss over multiple few-shot episodes.

**Supervised fine-tuning**  
As a baseline, HuBERT is fine-tuned using standard supervised learning on the available training data. This setting serves as a comparison point against few-shot approaches under limited data regimes.


## Results

Few-shot inference using frozen HuBERT embeddings already provides strong performance on several datasets, especially for speech and synthetic audio tasks and improves steadily from 1-shot to 10-shot settings.

Episodic prototypical training with a learned projection head further improves performance on most real-world datasets, indicating that metric learning helps structure the embedding space beyond raw HuBERT representations.

Supervised fine-tuning achieves the highest accuracy when sufficient labeled data is available, but often generalizes poorly to unseen classes. In contrast, prototypical learning remains more robust in low-data regimes and better suited for class-level generalization.



## References

- Snell, J., Swersky, K., & Zemel, R. (2017).  
  *Prototypical Networks for Few-shot Learning*.  
  Advances in Neural Information Processing Systems (NeurIPS).

- Hsu, W.-N., Bolte, B., Tsai, Y.-H. H., Lakhotia, K., Salakhutdinov, R., & Mohamed, A. (2021).  
  *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units*.  
  IEEE/ACM Transactions on Audio, Speech, and Language Processing.
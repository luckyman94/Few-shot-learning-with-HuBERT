# Experiments (Notebooks)

All experiments in this project are implemented as Jupyter notebooks and can be found in the `notebooks/` directory.

Each notebook corresponds to a specific dataset or experimental setting (few-shot learning, fine-tuning, prototypical training, etc.).

---

## Available notebooks

- `run_prototypical_with_training.ipynb`  
  Prototypical Networks with episodic training (few-shot setting).

- `run_dog_bird_cat.ipynb`  
  Few-shot experiments on the Animals (dog/bird/cat) dataset.

- `run_urban.ipynb`  
  Few-shot experiments on UrbanSound8K.

- `run_snoring.ipynb`, `run_speech_commands.ipynb`, `run_timit.ipynb`, `run_crema.ipynb`  
  Few-shot experiments on additional audio datasets.

- `run_finetuning_*.ipynb`  
  Supervised fine-tuning baselines for comparison with few-shot learning.

- `run_harmonics.ipynb`, `run_noisy.ipynb`  
  Experiments on synthetic audio datasets.

---

## Notes on execution (important)

Some notebooks are designed to run on **Kaggle** or **Google Colab**.

Depending on the environment, you may need to:
- ncomment specific cells (e.g. `git clone`, `pip install`, Kaggle credentials)
- adjust dataset paths
- enable GPU execution

Relevant cells are clearly marked inside the notebooks.

---

## Remarks

- HuBERT embeddings are cached when required to speed up training.
- Few-shot experiments follow a class-disjoint split (train/test classes are different).
- Results may slightly vary due to episodic sampling.

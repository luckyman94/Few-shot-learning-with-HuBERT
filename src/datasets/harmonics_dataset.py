import torch
import numpy as np
from torch.utils.data import Dataset


class SyntheticAudioHarmonicsDataset(Dataset):
    """
    A synthetic dataset that generates audio samples with varying numbers of harmonics.
    The number of harmonics can be controlled via the max_harmonics param.
    """
    def __init__(
        self,
        n_classes=8,
        n_samples=800,
        sample_rate=16000,
        max_len=16000,
        max_harmonics=8,      
        harmonic_decay=0.5,   
        seed=42,
    ):

        assert n_samples % n_classes == 0, "n_samples must be divisible by n_classes"

        rng = np.random.RandomState(seed)

        self.sample_rate = sample_rate
        self.max_len = max_len
        self.n_classes = n_classes

        self.classes = [f"harmonics_{i+1}" for i in range(n_classes)]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        samples_per_class = n_samples // n_classes
        self.data = []

        duration = max_len / sample_rate
        t = np.linspace(0, duration, max_len, endpoint=False)

        base_freq = 220.0  

        for class_id in range(n_classes):
            n_harmonics = min(class_id + 1, max_harmonics)

            for _ in range(samples_per_class):
                signal = np.zeros_like(t)

                
                signal += np.sin(2 * np.pi * base_freq * t)

                
                for h in range(2, 2 + n_harmonics):
                    amplitude = harmonic_decay ** (h - 1)
                    signal += amplitude * np.sin(2 * np.pi * base_freq * h * t)

                self.data.append((signal.astype(np.float32), class_id))

        print(
            f"[INFO] SyntheticAudioHarmonicsDataset | "
            f"{n_classes} classes | "
            f"{n_samples} samples | "
            f"Max harmonics={max_harmonics}"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio, label = self.data[idx]

        waveform = torch.tensor(audio)

        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-9)

        return waveform, label

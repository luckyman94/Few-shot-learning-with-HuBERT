import torch
import numpy as np
from torch.utils.data import Dataset


class SyntheticAudioNoiseDataset(Dataset):
    def __init__(
        self,
        n_classes=8,
        n_samples=200,
        sample_rate=16000,
        max_len=16000,
        snr_db=20,      
        seed=42,
    ):  
        assert n_samples % n_classes == 0, "n_samples must be divisible by n_classes"

        rng = np.random.RandomState(seed)

        self.sample_rate = sample_rate
        self.max_len = max_len
        self.n_classes = n_classes

        self.classes = [f"class_{i}" for i in range(n_classes)]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        samples_per_class = n_samples // n_classes
        self.data = []

        duration = max_len / sample_rate
        t = np.linspace(0, duration, max_len, endpoint=False)

        for class_id in range(n_classes):
            base_freq = 200 + class_id * 40

            for _ in range(samples_per_class):
                signal = np.sin(2 * np.pi * base_freq * t)

                noise = rng.randn(len(signal))
                signal_power = np.mean(signal ** 2)
                noise_power = signal_power / (10 ** (snr_db / 10))
                noise = np.sqrt(noise_power) * noise

                audio = signal + noise
                self.data.append((audio.astype(np.float32), class_id))

        print(
            f"[INFO] SyntheticAudioNoiseDataset | "
            f"{n_classes} classes | "
            f"{n_samples} samples | "
            f"SNR = {snr_db} dB"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio, label = self.data[idx]

        waveform = torch.tensor(audio)

        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-9)

        return waveform, label

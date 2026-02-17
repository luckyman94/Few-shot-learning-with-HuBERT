import os
import torch
import torchaudio
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(
        self,
        root_dir,
        sample_rate=16000,
        max_len=16000,
        normalize=True,
        random_crop=False,
    ):
        self.sample_rate = sample_rate
        self.max_len = max_len
        self.normalize = normalize
        self.random_crop = random_crop

        self.data = []

        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
            and not d.startswith("_")  # <-- IMPORTANT
        ])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            label = self.class_to_idx[cls]

            for file in os.listdir(cls_path):
                if file.endswith(".wav"):
                    self.data.append((os.path.join(cls_path, file), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]

        waveform, sr = torchaudio.load(path)
        waveform = waveform.mean(dim=0)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, self.sample_rate
            )

        if waveform.shape[0] > self.max_len:
            if self.random_crop:
                start = torch.randint(
                    0, waveform.shape[0] - self.max_len + 1, (1,)
                ).item()
                waveform = waveform[start:start + self.max_len]
            else:
                waveform = waveform[:self.max_len]
        else:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.max_len - waveform.shape[0])
            )

        if self.normalize:
            waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-9)

        return waveform, label

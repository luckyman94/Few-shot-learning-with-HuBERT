import os
import torch
import torchaudio
from torch.utils.data import Dataset
import kagglehub


class CremaDDataset(Dataset):
    def __init__(self, sample_rate=16000, max_len=24000):
        """
        CREMA-D emotion classification dataset

        Returns:
            waveform: Tensor [T]
            label: int
        """
        base_path = kagglehub.dataset_download("ejlok1/cremad")
        self.root_dir = os.path.join(base_path, "AudioWAV")

        self.sample_rate = sample_rate
        self.max_len = max_len
        self.data = []

        # Emotion mapping (official CREMA-D)
        self.emotion_map = {
            "ANG": "angry",
            "DIS": "disgust",
            "FEA": "fearful",
            "HAP": "happy",
            "NEU": "neutral",
            "SAD": "sad",
        }

        self.classes = sorted(self.emotion_map.values())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for file in os.listdir(self.root_dir):
            if not file.endswith(".wav"):
                continue

            parts = file.split("_")
            if len(parts) < 3:
                continue

            emotion_code = parts[2]
            if emotion_code not in self.emotion_map:
                continue

            emotion = self.emotion_map[emotion_code]
            label = self.class_to_idx[emotion]

            self.data.append(
                (os.path.join(self.root_dir, file), label)
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]

        waveform, sr = torchaudio.load(path)
        waveform = waveform.mean(dim=0)  # mono

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, self.sample_rate
            )

        if waveform.shape[0] < self.max_len:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.max_len - waveform.shape[0])
            )
        else:
            waveform = waveform[:self.max_len]

        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-9)

        return waveform, label

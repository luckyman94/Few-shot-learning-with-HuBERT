import os
import torch
import torchaudio
from torch.utils.data import Dataset
import kagglehub


class TimitDataset(Dataset):
    def __init__(
        self,
        root_dir="",
        sample_rate=16000,
        max_len=16000,
        max_files=None,
    ):
        """
        DARPA TIMIT dataset (Kaggle)

        Folder structure (typical):
        TIMIT/
          ├── TRAIN/
          │   ├── DR1/
          │   │   ├── MABC0/
          │   │   │   ├── SA1.wav
          │   │   │   ├── SX1.wav
          │   │   │   └── ...
          ├── TEST/
          │   └── ...

        Here:
        - class = speaker ID
        - task = few-shot speaker classification
        """

        base_path = kagglehub.dataset_download(
            "mfekadu/darpa-timit-acousticphonetic-continuous-speech"
        )

        self.root_dir = os.path.join(base_path, root_dir)
        self.sample_rate = sample_rate
        self.max_len = max_len
        self.data = []

        # ─────────────────────────────────────────────
        # Collect speakers
        # ─────────────────────────────────────────────
        speakers = set()

        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(".wav"):
                    speaker_id = os.path.basename(root)
                    speakers.add(speaker_id)

        self.classes = sorted(list(speakers))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # ─────────────────────────────────────────────
        # Collect audio files
        # ─────────────────────────────────────────────
        for root, _, files in os.walk(self.root_dir):
            speaker_id = os.path.basename(root)
            if speaker_id not in self.class_to_idx:
                continue

            label = self.class_to_idx[speaker_id]

            for file in files:
                if not file.endswith(".wav"):
                    continue

                self.data.append((os.path.join(root, file), label))

                if max_files is not None and len(self.data) >= max_files:
                    break

            if max_files is not None and len(self.data) >= max_files:
                break

        print(
            f"[INFO] TIMIT Dataset: "
            f"{len(self.data)} files | "
            f"{len(self.classes)} speakers"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]

        waveform, sr = torchaudio.load(path)
        waveform = waveform.mean(dim=0)
        sr = int(sr)

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

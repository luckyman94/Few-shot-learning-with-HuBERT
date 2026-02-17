import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset
import kagglehub


class TimitDataset(Dataset):
    def __init__(
        self,
        root_dir="data/TRAIN",
        sample_rate=16000,
        max_len=16000,
        n_speakers=10,
        max_files=None,
        seed=42,
    ):

        random.seed(seed)

        base_path = kagglehub.dataset_download(
            "mfekadu/darpa-timit-acousticphonetic-continuous-speech"
        )

        self.root_dir = os.path.join(base_path, root_dir)
        self.sample_rate = sample_rate
        self.max_len = max_len
        self.data = []

        speakers = set()

        for dr in os.listdir(self.root_dir):
            dr_path = os.path.join(self.root_dir, dr)
            if not os.path.isdir(dr_path):
                continue

            for speaker in os.listdir(dr_path):
                speaker_path = os.path.join(dr_path, speaker)
                if os.path.isdir(speaker_path):
                    speakers.add(speaker)

        speakers = sorted(list(speakers))
        print(f"[INFO] Total speakers in TIMIT: {len(speakers)}")

        if n_speakers is not None:
            speakers = random.sample(
                speakers, min(n_speakers, len(speakers))
            )

        self.classes = speakers
        self.class_to_idx = {s: i for i, s in enumerate(self.classes)}

        
        for dr in os.listdir(self.root_dir):
            dr_path = os.path.join(self.root_dir, dr)
            if not os.path.isdir(dr_path):
                continue

            for speaker in os.listdir(dr_path):
                if speaker not in self.class_to_idx:
                    continue

                speaker_path = os.path.join(dr_path, speaker)
                label = self.class_to_idx[speaker]

                for file in os.listdir(speaker_path):
                    if not file.lower().endswith(".wav"):
                        continue

                    self.data.append(
                        (os.path.join(speaker_path, file), label)
                    )

                    if max_files and len(self.data) >= max_files:
                        break

            if max_files and len(self.data) >= max_files:
                break

        print(
            f"[INFO] TIMIT Dataset loaded: "
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

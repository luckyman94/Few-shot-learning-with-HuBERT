import os
import torch
import torchaudio
from torch.utils.data import Dataset
import kagglehub
import random


class TimitDataset(Dataset):
    def __init__(
        self,
        root_dir="TRAIN",
        sample_rate=16000,
        max_len=16000,
        max_files=None,
        n_speakers=10,
        seed=42,
    ):
        """
        DARPA TIMIT dataset (few-shot speaker classification)

        - class = speaker ID
        - reduced to n_speakers speakers
        """

        base_path = kagglehub.dataset_download(
            "mfekadu/darpa-timit-acousticphonetic-continuous-speech"
        )

        self.root_dir = os.path.join(base_path, root_dir)
        self.sample_rate = sample_rate
        self.max_len = max_len
        self.data = []

        # ─────────────────────────────────────────────
        # Collect all speakers
        # ─────────────────────────────────────────────
        all_speakers = set()

        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(".wav"):
                    speaker_id = os.path.basename(root)
                    all_speakers.add(speaker_id)

        all_speakers = sorted(list(all_speakers))

        print(f"[INFO] Total speakers in TIMIT: {len(all_speakers)}")

        # ─────────────────────────────────────────────
        # Select subset of speakers
        # ─────────────────────────────────────────────
        random.seed(seed)
        selected_speakers = random.sample(all_speakers, n_speakers)

        self.classes = sorted(selected_speakers)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # ─────────────────────────────────────────────
        # Collect audio files for selected speakers
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
            f"[INFO] TIMIT subset: "
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

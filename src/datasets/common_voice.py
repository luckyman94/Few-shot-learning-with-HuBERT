import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path


class CommonVoiceDataset(Dataset):
    def __init__(
        self,
        root_dir,
        tsv="train.tsv",
        sample_rate=16000,
        max_len=32000,
        max_samples=None,
        debug=False,
    ):
        
        print(f"Loading Common Voice from {root_dir} ({tsv})")

        self.sample_rate = sample_rate
        self.max_len = max_len
        self.debug = debug

        self.dataset = torchaudio.datasets.COMMONVOICE(
            root=Path(root_dir),
            tsv=tsv,
        )

        self.data = []
        self.speaker_to_idx = {}

        next_label = 0

        for i in range(len(self.dataset)):
            if max_samples is not None and i >= max_samples:
                break

            waveform, sr, meta = self.dataset[i]

            speaker_id = meta.get("client_id")
            if speaker_id is None:
                continue

            if speaker_id not in self.speaker_to_idx:
                self.speaker_to_idx[speaker_id] = next_label
                next_label += 1

            label = self.speaker_to_idx[speaker_id]
            self.data.append((i, label))

        self.classes = list(range(next_label))

        if self.debug:
            print(f"[DEBUG] Loaded {len(self.data)} samples")
            print(f"[DEBUG] Number of speakers: {len(self.classes)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        original_idx, label = self.data[idx]

        waveform, sr, _ = self.dataset[original_idx]
        sr = int(sr)

        waveform = waveform.mean(dim=0)

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

        if self.debug and idx < 3:
            print(
                f"[DEBUG] idx={idx} | shape={waveform.shape} | label={label}"
            )

        return waveform, label

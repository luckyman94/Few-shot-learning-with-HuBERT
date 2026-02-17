import torch
import torchaudio
from torch.utils.data import Dataset
from datasets import load_dataset


class SpeechCommandsDataset(Dataset):
    def __init__(
        self,
        split="train",
        sample_rate=16000,
        max_len=16000,
        commands=None,
        max_samples_per_class=None,
        debug=False,
    ):
        """
        Google Speech Commands dataset (real audio)

        Returns:
            waveform: Tensor [T]
            label: int
        """
        print(f"Loading Speech Commands ({split})")

        self.sample_rate = sample_rate
        self.max_len = max_len
        self.debug = debug
        self.data = []

        # Default 10 commands (standard benchmark)
        if commands is None:
            commands = [
                "yes", "no", "up", "down", "left",
                "right", "on", "off", "stop", "go"
            ]

        self.classes = sorted(commands)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        dataset = load_dataset(
            "speech_commands",
            "v0.02",
            split=split,
        )

        per_class_count = {c: 0 for c in self.classes}

        for ex in dataset:
            label_id = ex["label"]
            command = dataset.features["label"].int2str(label_id)

            if command not in self.class_to_idx:
                continue

            if (
                max_samples_per_class is not None
                and per_class_count[command] >= max_samples_per_class
            ):
                continue

            audio = ex["audio"]["array"]
            sr = ex["audio"]["sampling_rate"]

            label = self.class_to_idx[command]
            self.data.append((audio, sr, label))
            per_class_count[command] += 1

        if self.debug:
            print(f"[DEBUG] Classes: {self.classes}")
            print(f"[DEBUG] Total samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio, sr, label = self.data[idx]

        waveform = torch.tensor(audio)
        sr = int(sr)

        if waveform.ndim > 1:
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

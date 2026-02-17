import os
import torch
import torchaudio
from torch.utils.data import Dataset
import kagglehub


class SpeechCommandsDataset(Dataset):
    def __init__(
        self,
        root_dir="speech_commands",
        sample_rate=16000,
        max_len=16000,
        max_files=1000,
    ):
        """
        Speech Commands dataset loaded from Kaggle.

        Folder structure:
        speech_commands/
          ├── down/
          ├── left/
          ├── off/
          ├── on/
          ├── right/
          ├── stop/
          ├── up/

        Returns:
            waveform: Tensor [T]
            label: int
        """

        base_path = kagglehub.dataset_download(
            "nikhilkushwaha2529/speech-commands"
        )
        self.root_dir = os.path.join(base_path, root_dir)

        self.sample_rate = sample_rate
        self.max_len = max_len
        self.data = []

        # Classes = folder names
        self.classes = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Collect files (limit to max_files total)
        for cls in self.classes:
            cls_path = os.path.join(self.root_dir, cls)
            label = self.class_to_idx[cls]

            for file in os.listdir(cls_path):
                if not file.endswith(".wav"):
                    continue

                if len(self.data) >= max_files:
                    break

                self.data.append(
                    (os.path.join(cls_path, file), label)
                )

            if len(self.data) >= max_files:
                break

        print(
            f"[INFO] Speech Commands (Kaggle): "
            f"{len(self.data)} files, {len(self.classes)} classes"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]

        waveform, sr = torchaudio.load(path)
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

        return waveform, label

import torch
import torchaudio
from torch.utils.data import Dataset
from datasets import load_dataset


class CommonVoiceDataset(Dataset):
    def __init__(
        self,
        language="en",
        split="train",
        max_samples=2000,
        sample_rate=16000,
        max_len=32000,
        debug=False,
    ):
        """
        Common Voice speaker identification dataset 

        label = speaker_id 

        Parameters
        ----------
        language : str
            Language code (e.g. "en", "fr")
        split : str
            Dataset split ("train", "validation", "test")
        max_samples : int
            Max number of utterances loaded (for speed)
        sample_rate : int
            Target sampling rate
        max_len : int
            Fixed waveform length (padding / truncation)
        debug : bool
            Print debug info
        """
        print(f"Loading Common Voice ({language}, split={split})")

        self.sample_rate = sample_rate
        self.max_len = max_len
        self.debug = debug
        self.data = []

        dataset = load_dataset(
            "mozilla-foundation/common_voice_16_1",
            language,
            split=split,
        )

        speaker_to_idx = {}
        next_label = 0

        for i, example in enumerate(dataset):
            if i >= max_samples:
                break

            speaker_id = example["client_id"]
            if speaker_id is None:
                continue

            if speaker_id not in speaker_to_idx:
                speaker_to_idx[speaker_id] = next_label
                next_label += 1

            label = speaker_to_idx[speaker_id]

            audio = example["audio"]["array"]
            sr = example["audio"]["sampling_rate"]

            self.data.append((audio, sr, label))

        self.classes = list(range(next_label))
        self.class_to_idx = speaker_to_idx

        if self.debug:
            print(f"[DEBUG] Loaded {len(self.data)} samples")
            print(f"[DEBUG] Number of speakers: {len(self.classes)}")

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

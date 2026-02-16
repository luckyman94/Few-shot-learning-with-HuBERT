import torch
import torchaudio
import librosa
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm



def load_speech_commands(cfg):
    root = Path.home() / ".torchaudio"
    root.mkdir(exist_ok=True)

    dataset = torchaudio.datasets.SPEECHCOMMANDS(
        root=root,
        download=True,
    )

    sr_target = cfg["audio"]["sample_rate"]
    max_len = int(cfg["audio"]["max_duration"] * sr_target)
    classes = set(cfg["labels"]["classes"])

    data = defaultdict(list)

    for waveform, sr, label, speaker_id, utt_id in tqdm(dataset, desc="Loading Speech Commands"):

        if label not in classes:
            continue

        audio = waveform.squeeze(0).numpy()

        if sr != sr_target:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sr_target)

        audio = audio[:max_len]

        if len(audio) < int(0.2 * sr_target):
            continue

        data[label].append(audio)

    return dict(data)

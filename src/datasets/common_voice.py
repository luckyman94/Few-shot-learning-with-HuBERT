from datasets import load_dataset
import librosa
from collections import defaultdict
import os


def load_common_voice(cfg):
    token = os.environ.get("HF_TOKEN")
    dataset = load_dataset(
        cfg["hf_dataset"],
        cfg["language"],
        split=cfg["splits"]["split"],
        token=token, 
    )

    sr_target = cfg["audio"]["sample_rate"]
    max_len = int(cfg["audio"]["max_duration"] * sr_target)
    min_utts = cfg["labels"]["min_utterances_per_class"]
    max_speakers = cfg["labels"]["max_speakers"]

    speaker_dict = defaultdict(list)

    for ex in dataset:
        speaker = ex["client_id"]
        audio = ex["audio"]["array"]
        sr = ex["audio"]["sampling_rate"]

        if sr != sr_target:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sr_target)

        audio = audio[:max_len]

        if len(audio) < int(0.2 * sr_target):
            continue

        speaker_dict[speaker].append(audio)

    speaker_dict = {
        spk: utts
        for spk, utts in speaker_dict.items()
        if len(utts) >= min_utts
    }

    speakers = list(speaker_dict.keys())[:max_speakers]

    return {spk: speaker_dict[spk] for spk in speakers}

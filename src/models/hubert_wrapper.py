import torch
import torch.nn as nn
from transformers import (
    HubertModel,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer
)
import json
import os

class HubertProtoNet(nn.Module):
    def __init__(
        self,
        hubert_model: str,
        embedding_dim: int = 256,
        freeze_hubert: bool = True,
    ):
        super().__init__()

        self.processor = self._load_processor(hubert_model)
        self.hubert = HubertModel.from_pretrained(hubert_model, use_safetensors=True)

        if freeze_hubert:
            for p in self.hubert.parameters():
                p.requires_grad = False

        hubert_dim = self.hubert.config.hidden_size

        self.projection = nn.Sequential(
            nn.Linear(hubert_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim),
        )

    def _load_processor(self, hubert_model):
        try:
            return Wav2Vec2Processor.from_pretrained(hubert_model)
        except Exception:
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_model)
            vocab_path = "dummy_vocab.json"
            if not os.path.exists(vocab_path):
                with open(vocab_path, "w") as f:
                    json.dump({"<pad>": 0, "|": 1}, f)

            tokenizer = Wav2Vec2CTCTokenizer(
                vocab_file=vocab_path,
                pad_token="<pad>",
                word_delimiter_token="|",
            )
            return Wav2Vec2Processor(
                feature_extractor=feature_extractor,
                tokenizer=tokenizer,
            )

    def extract_features(self, waveforms, sample_rate=16000):
        inputs = self.processor(
            waveforms,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        outputs = self.hubert(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def forward(self, waveforms):
        feats = self.extract_features(waveforms)
        return self.projection(feats)

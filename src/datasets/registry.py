from .common_voice import load_common_voice
from .speech_commands_dataset import load_speech_commands


DATASET_REGISTRY = {
    "common_voice": load_common_voice,
    "speech_commands": load_speech_commands,
}





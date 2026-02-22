# Dataset registry mapping dataset names to their classes and parameters.

DATASET_REGISTRY = {
    "animals": {
        "class": "AnimalAudioDataset",
        "params": {
            "sample_rate": 16000,
            "max_len": 16000,
        },
        "n_way": 3,
    },

    "speech_commands": {
        "class": "SpeechCommandsDataset",
        "params": {}
    },

    "synthetic_noise_low": {
        "class": "SyntheticAudioNoiseDataset",
        "params": {
            "n_classes": 8,
            "n_samples": 200,
            "snr_db": 30,
        },
        "n_way": 8,
    },

    "synthetic_noise_medium": {
        "class": "SyntheticAudioNoiseDataset",
        "params": {
            "n_classes": 8,
            "n_samples": 200,
            "snr_db": 10,
        },
        "n_way": 8,
    },

    "synthetic_noise_high": {
        "class": "SyntheticAudioNoiseDataset",
        "params": {
            "n_classes": 8,
            "n_samples": 200,
            "snr_db": 0,
        },
        "n_way": 8,
    },

    "synthetic_harmonics_low": {
        "class": "SyntheticAudioHarmonicsDataset",
        "params": {
            "n_classes": 8,
            "n_samples": 200,
            "max_harmonics": 2,
        },
        "n_way": 8,
    },

      "synthetic_harmonics_medium": {
        "class": "SyntheticAudioHarmonicsDataset",
        "params": {
            "n_classes": 8,
            "n_samples": 200,
            "max_harmonics": 4,
        },
        "n_way": 8,
    },

      "synthetic_harmonics_high": {
        "class": "SyntheticAudioHarmonicsDataset",
        "params": {
            "n_classes": 8,
            "n_samples": 200,
            "max_harmonics": 8,
        },
        "n_way": 8,
    },

    "urban": {
        "class": "UrbanDataset",
        "params": {},
        "n_way": 10,
    },

    "crema": {
        "class": "CremaDDataset",
        "params": {},
        "n_way": 6,
    },

    "timit": {
        "class": "TimitDataset",
        "params": {"root_dir": "data/TRAIN", "n_speakers": 10, "max_files":1000},
        "n_way": 5,
    },

    "snoring": {
        "class": "SnoringDataset",
        "params": {},
        "n_way": 2,
    },
}
# src/datasets/factory.py

from src.datasets.utils import discover_datasets
from src.datasets.registry import DATASET_REGISTRY

_DATASET_CLASSES = discover_datasets()


def build_dataset(name):
    """
    Instantiate a dataset from the registry.
    """
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset: {name}")

    entry = DATASET_REGISTRY[name]
    cls_name = entry["class"]
    params = entry.get("params", {})

    if cls_name not in _DATASET_CLASSES:
        raise KeyError(f"Dataset class not found: {cls_name}")

    DatasetCls = _DATASET_CLASSES[cls_name]
    return DatasetCls(**params)
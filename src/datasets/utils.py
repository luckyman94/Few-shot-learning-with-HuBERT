import os
import importlib
import inspect
from torch.utils.data import Dataset


def discover_datasets(package="src.datasets"):

    datasets = {}

    pkg = importlib.import_module(package)

    dataset_dir = list(pkg.__path__)[0]

    for file in os.listdir(dataset_dir):
        if not file.endswith(".py"):
            continue
        if file.startswith("_"):
            continue

        module_name = file[:-3]
        module = importlib.import_module(f"{package}.{module_name}")

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Dataset) and obj is not Dataset:
                datasets[name] = obj

    return datasets
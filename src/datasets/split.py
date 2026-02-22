from torch.utils.data import Dataset
import numpy as np

class ClassSubsetDataset(Dataset):
    def __init__(self, dataset, allowed_classes):
        self.dataset = dataset
        self.allowed_classes = set(allowed_classes)

        self.indices = [
            i for i in range(len(dataset))
            if dataset[i][1] in self.allowed_classes
        ]

        # classes visibles dans ce subset
        self.classes = sorted(list(self.allowed_classes))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
    



def split_dataset_by_classes(
    dataset,
    train_ratio=0.7,
    seed=42,
):
    rng = np.random.RandomState(seed)

    classes = list(dataset.classes)
    rng.shuffle(classes)

    n_train = max(1, int(len(classes) * train_ratio))

    train_classes = classes[:n_train]
    test_classes = classes[n_train:]

    if len(test_classes) == 0:
        raise ValueError(
            "Not enough classes for class-level split"
        )

    train_dataset = ClassSubsetDataset(
        dataset, train_classes
    )
    test_dataset = ClassSubsetDataset(
        dataset, test_classes
    )

    return train_dataset, test_dataset
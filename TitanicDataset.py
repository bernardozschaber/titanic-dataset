from flautim.pytorch.Dataset import Dataset
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset as TorchDataset
from sklearn.preprocessing import StandardScaler


class TitanicTorchDataset(TorchDataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "features": self.X[idx],
            "label": self.y[idx]
        }


def dirichlet_partition(X, y, num_clients, alpha=0.5, seed=42):
    np.random.seed(seed)

    classes = np.unique(y)
    client_indices = [[] for _ in range(num_clients)]

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)

        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        counts = (proportions * len(cls_indices)).astype(int)

        diff = len(cls_indices) - counts.sum()
        for i in range(abs(diff)):
            counts[i % num_clients] += 1 if diff > 0 else -1

        start = 0
        for client_id in range(num_clients):
            end = start + counts[client_id]
            client_indices[client_id].extend(cls_indices[start:end])
            start = end

    for client_id in range(num_clients):
        np.random.shuffle(client_indices[client_id])

    return client_indices


class TitanicDataset(Dataset):
    def __init__(self, csv_path, client_id, num_clients=4, test_size=0.2, seed=42, **kwargs):
        name = kwargs.get("name", "Titanic")
        super(TitanicDataset, self).__init__(name, **kwargs)

        self.batch_size = kwargs.get("batch_size", 32)
        self.shuffle = kwargs.get("shuffle", True)

        df = pd.read_csv(csv_path)

        df = df[
            [
                "Pclass",
                "Sex",
                "Age",
                "SibSp",
                "Parch",
                "Fare",
                "Embarked",
                "Survived",
            ]
        ].copy()

        df["Age"] = df["Age"].fillna(df["Age"].median())
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

        df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
        df = pd.get_dummies(df, columns=["Embarked"], drop_first=False)

        for col in df.columns:
            if df[col].dtype == "bool":
                df[col] = df[col].astype(int)

        X = df.drop(columns=["Survived"]).astype("float32").values
        y = df["Survived"].astype("int64").values

        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype("float32")

        if (
            not hasattr(TitanicDataset, "_global_partition")
            or not hasattr(TitanicDataset, "_partition_config")
            or TitanicDataset._partition_config != (num_clients, seed)
        ):
            TitanicDataset._global_partition = dirichlet_partition(
                X, y, num_clients=num_clients, alpha=0.5, seed=seed
            )
            TitanicDataset._partition_config = (num_clients, seed)

        indices = TitanicDataset._global_partition[client_id]

        X_client = X[indices]
        y_client = y[indices]

        split_idx = int((1 - test_size) * len(X_client))
        self.X_train = X_client[:split_idx]
        self.y_train = y_client[:split_idx]
        self.X_test = X_client[split_idx:]
        self.y_test = y_client[split_idx:]

        self.train_partition = TitanicTorchDataset(self.X_train, self.y_train)
        self.test_partition = TitanicTorchDataset(self.X_test, self.y_test)

        self.input_dim = self.X_train.shape[1]

    def train(self):
        return self.train_partition

    def validation(self):
        return self.test_partition

    def dataloader(self, validation=False):
        dataset = self.validation() if validation else self.train()
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(False if validation else self.shuffle),
            num_workers=0,
        )
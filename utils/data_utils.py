from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


class TabularDataset(Dataset):
    """
    PyTorch dataset for tabular features and binary labels.
    用于表格特征和二分类标签的 PyTorch 数据集。
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


@dataclass
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


def load_dataframe(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def validate_columns(df: pd.DataFrame, feature_cols, target_col: str):
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")


def prepare_centralized_splits(
    train_csv_path: str,
    test_csv_path: str,
    feature_cols,
    target_col: str,
    val_size: float,
    random_state: int,
) -> SplitData:
    """
    Load centralized train/test data, create validation split, and standardize features.
    读取集中式训练/测试数据，划分验证集，并对特征做标准化。
    """

    train_df = load_dataframe(train_csv_path)
    test_df = load_dataframe(test_csv_path)

    validate_columns(train_df, feature_cols, target_col)
    validate_columns(test_df, feature_cols, target_col)

    X_full = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_full = train_df[target_col].to_numpy(dtype=np.int64)

    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df[target_col].to_numpy(dtype=np.int64)

    X_train, X_val, y_train, y_val = train_test_split(
        X_full,
        y_full,
        test_size=val_size,
        random_state=random_state,
        stratify=y_full,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return SplitData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        scaler=scaler,
    )


def create_dataloader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TabularDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_pos_weight(labels: np.ndarray) -> torch.Tensor:
    positive = np.sum(labels == 1)
    negative = np.sum(labels == 0)
    if positive == 0:
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(negative / max(positive, 1), dtype=torch.float32)

def load_federated_client_data(csv_path: str, feature_cols, target_col: str):
    """
    Load and standardize one federated client's local data.
    读取并标准化单个联邦客户端的本地数据。
    """

    df = load_dataframe(csv_path)
    validate_columns(df, feature_cols, target_col)

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.int64)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

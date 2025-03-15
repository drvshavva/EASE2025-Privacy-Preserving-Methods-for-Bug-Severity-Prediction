import os as os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from flwr.common import NDArrays
from flwr_datasets.partitioner import IidPartitioner
from imblearn.over_sampling import ADASYN
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from sklearn.model_selection import train_test_split


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


ROOT_DIR = get_project_root()
DATASET_DIR = os.path.join(ROOT_DIR, "datasets")

UNIQUE_LABELS = [0, 1, 2, 3]
FEATURES = ['project_name', 'project_version', 'label', 'code', 'code_comment',
            'code_no_comment', 'lc', 'pi', 'ma', 'nbd', 'ml', 'd', 'mi', 'fo', 'r',
            'e']
TRAIN_COLS = ['lc', 'pi', 'ma', 'nbd', 'ml', 'd', 'mi', 'fo', 'r', 'e']


def load_data_from_local_synthetic(partition_id: int, opr_dir: str):
    data_files = f"{DATASET_DIR}/{opr_dir}/client_{partition_id}.csv"
    file_path = os.path.join(DATASET_DIR, data_files)
    dataset = pd.read_csv(file_path)

    X = dataset.drop(columns=["label"])
    y = dataset["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=42
    )

    print(f"Client {partition_id}: GaussianCopula ile veri artırımı yapılıyor...")

    train_data = pd.concat([x_train, y_train], axis=1)

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_data)
    metadata.update_column(column_name='label', sdtype='categorical')

    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(train_data)

    target_size = y_train.value_counts().max()

    synthetic_samples = []
    for label in y_train.unique():
        count_needed = target_size - (y_train == label).sum()
        if count_needed > 0:
            synth_data = synthesizer.sample(count_needed)
            synth_data = synth_data[synth_data['label'] == label]
            synthetic_samples.append(synth_data)

    if synthetic_samples:
        synthetic_data = pd.concat(synthetic_samples)
        x_synthetic = synthetic_data.drop(columns=["label"])
        y_synthetic = synthetic_data["label"]

        # Orijinal veri ile sentetik veriyi birleştir
        x_train = pd.concat([x_train, x_synthetic], ignore_index=True)
        y_train = pd.concat([y_train, y_synthetic], ignore_index=True)

    return x_train, y_train, x_test, y_test


def load_data_from_local_imblearn(partition_id: int, opr_dir: str):
    data_files = f"{DATASET_DIR}/{opr_dir}/client_{partition_id}.csv"
    file_path = os.path.join(DATASET_DIR, data_files)
    dataset = pd.read_csv(file_path)

    x = dataset.drop(columns=["label"])
    y = dataset["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, stratify=y, random_state=42
    )

    smote = ADASYN(random_state=42)
    x_train, y_train = smote.fit_resample(x_train, y_train)

    return x_train, y_train, x_test, y_test


def set_initial_params(model, n_classes: int, n_features: int):
    """Set initial parameters as zeros.

    Required since model params are uninitialized until model.fit is called but server
    asks for initial parameters from clients at launch. Refer to
    sklearn.linear_model.LogisticRegression documentation for more information.
    """
    model.classes_ = np.array([i for i in range(n_classes)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))


def get_model_parameters(model) -> NDArrays:
    """Return the parameters of a sklearn model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params


def set_model_params(model, params: NDArrays):
    """Set the parameters of a sklean model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


partitioner = None


def load_data(partition_id: int, num_partitions: int):
    """Load the data for the given partition."""
    global partitioner
    data_files = fr"{DATASET_DIR}\train_scaled.csv"

    if partitioner is None:
        fds = load_dataset("csv", data_files=data_files)["train"]
        # fds = Dataset.from_pandas(df)
        partitioner = IidPartitioner(num_partitions=num_partitions)
        partitioner.dataset = fds

    dataset = partitioner.load_partition(partition_id).with_format("pandas")[:]
    X = dataset[TRAIN_COLS]
    y = dataset["label"]
    # Split the on-edge data: 80% train, 20% test
    x_train, x_test = X[: int(0.8 * len(X))], X[int(0.8 * len(X)):]
    y_train, y_test = y[: int(0.8 * len(y))], y[int(0.8 * len(y)):]
    return x_train, y_train, x_test, y_test


def compute_entropy(labels):
    label_counts = Counter(labels)
    total_samples = len(labels)
    probabilities = [count / total_samples for count in label_counts.values()]
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    return entropy


def load_data_from_local(partition_id: int, opr_dir: str):
    data_files = f"{DATASET_DIR}/{opr_dir}/client_{partition_id}.csv"
    file_path = os.path.join(DATASET_DIR, data_files)

    dataset = pd.read_csv(file_path)
    # X = dataset[TRAIN_COLS]
    # y = dataset["label"]
    X = dataset.drop(columns=["label"])
    y = dataset["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=42  # Stratify ile dengeli bölme
    )
    return x_train, y_train, x_test, y_test


def load_test_dataset():
    data_files = [fr"{DATASET_DIR}\data\test_processed.csv"]
    dataset = load_dataset("csv", data_files=data_files)["train"].with_format("pandas")[:]
    X = dataset.drop(columns=["label"])
    y = dataset["label"]

    # X = dataset[TRAIN_COLS]
    # y = dataset["label"]
    return X, y

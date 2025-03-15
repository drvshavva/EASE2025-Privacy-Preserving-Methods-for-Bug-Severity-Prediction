import warnings

import numpy as np
from flwr.client import NumPyClient
from flwr.common import Context
from scipy.special import expit
from sklearn.metrics import log_loss

from config import RunConf
from experiments.federated.utils import (
    UNIQUE_LABELS,
    get_model_parameters,
    load_data_from_local,
    set_model_params
)
from task import create_svc_and_instantiate_parameters


class FlowerClient(NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train.values
        self.y_train = y_train.values
        self.X_test = X_test.values
        self.y_test = y_test.values
        self.unique_labels = UNIQUE_LABELS

    def fit(self, parameters, config):
        set_model_params(self.model, parameters)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)
        print(f"İstemci modeli eğitiliyor... Loss değeri: {self.model.coef_}")
        accuracy = self.model.score(self.X_train, self.y_train)
        ndarrays = get_model_parameters(self.model)
        return (
            ndarrays,
            len(self.X_train),
            {"train_accuracy": accuracy},
        )

    def evaluate(self, parameters, config):  # type: ignore
        set_model_params(self.model, parameters)
        y_pred = self.model.predict(self.X_test)
        print("Tahmin edilen sınıflar:", np.unique(y_pred, return_counts=True))
        loss = 0.0
        accuracy = self.model.score(self.X_test, self.y_test)
        return loss, len(self.X_test), {'accuracy': accuracy}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    X_train, y_train, X_test, y_test = load_data_from_local(partition_id, RunConf.data_path)
    # Create LogisticRegression Model
    model = create_svc_and_instantiate_parameters(penalty=RunConf.penalty,
                                                  loss=RunConf.loss,
                                                  c=RunConf.c,
                                                  max_iter=RunConf.max_iter,
                                                  tol=RunConf.tol,
                                                  random_state=RunConf.random_state)

    # Return Client instance
    return FlowerClient(model, X_train, y_train, X_test, y_test).to_client()

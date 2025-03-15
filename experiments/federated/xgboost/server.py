"""xgboost_quickstart: A Flower / XGBoost app."""
import pickle
from logging import INFO
from typing import Dict, List, Optional

import numpy as np
import xgboost as xgb
from flwr.common import Context, Parameters, Scalar
from flwr.common.logger import log
from flwr.server import ServerAppComponents, ServerConfig
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
from flwr.server.strategy import FedXgbBagging, FedXgbCyclic

from config import RunConf
from experiments.federated.utils import load_test_dataset
from src.evalution import evaluatelog_result
from task import transform_dataset_to_dmatrix


class CyclicClientManager(SimpleClientManager):
    """Provides a cyclic client selection rule."""

    def sample(
            self,
            num_clients: int,
            min_num_clients: Optional[int] = None,
            criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""

        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)

        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]

        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []

        # Return all available clients
        return [self.clients[cid] for cid in available_cids]


def evaluate_metrics_aggregation(eval_metrics):
    """Return an aggregated metric (AUC) for evaluation."""
    total_num = sum([num for num, _ in eval_metrics])
    auc_aggregated = (
            sum([metrics["mlogloss"] * num for num, metrics in eval_metrics]) / total_num
    )
    metrics_aggregated = {"mlogloss": auc_aggregated}
    return metrics_aggregated


def config_func(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config


def get_evaluate_fn(params):
    """Return a function for centralised evaluation."""
    x_test, y_test = load_test_dataset()
    dtest = transform_dataset_to_dmatrix(x_test.values, y_test.values)

    def evaluate_fn(
            server_round: int, parameters: Parameters, config: Dict[str, Scalar]
    ):
        # If at the first round, skip the evaluation
        if server_round == 0:
            return 0, {}
        else:
            bst = xgb.Booster(params=params)
            for para in parameters.tensors:
                para_b = bytearray(para)

            # Load global model
            bst.load_model(para_b)
            # Run evaluation
            eval_results = bst.eval_set(
                evals=[(dtest, "valid")],
                iteration=bst.num_boosted_rounds() - 1,
            )
            mlogloss = round(float(eval_results.split("\t")[1].split(":")[1]), 4)
            y_pred_prob = bst.predict(dtest)

            # Convert probabilities to class labels (argmax for multi-class classification)
            y_pred = np.argmax(y_pred_prob, axis=1)

            res = evaluatelog_result(y_true=y_test, y_prediction=y_pred, model_name=RunConf.model,
                                     logger=RunConf.logger, prob=y_pred_prob)
            if server_round == RunConf.num_server_rounds:
                pickle.dump(bst, open(RunConf.model_name, "wb"))

            return mlogloss, res

    return evaluate_fn


def server_fn(context: Context):
    # Read from config
    num_rounds = RunConf.num_rounds
    train_method = RunConf.train_method
    params = RunConf.params
    # Init an empty Parameter
    parameters = Parameters(tensor_type="", tensors=[])
    min_available_clients = RunConf.min_available_clients
    # Define strategy
    if train_method == "bagging":
        # Bagging training
        strategy = FedXgbBagging(
            evaluate_function=get_evaluate_fn(params),
            min_available_clients=min_available_clients,
            on_evaluate_config_fn=config_func,
            on_fit_config_fn=config_func,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
            initial_parameters=parameters,
        )
    else:
        # Cyclic training
        strategy = FedXgbCyclic(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            evaluate_function=get_evaluate_fn(params),
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
            on_evaluate_config_fn=config_func,
            on_fit_config_fn=config_func,
            initial_parameters=parameters,
        )

    config = ServerConfig(num_rounds=num_rounds)
    client_manager = CyclicClientManager() if train_method == "cyclic" else None

    return ServerAppComponents(
        strategy=strategy, config=config, client_manager=client_manager
    )

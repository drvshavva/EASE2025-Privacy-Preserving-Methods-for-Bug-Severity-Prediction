import pickle
from typing import Dict, List, Tuple

import flwr as fl
from flwr.common import Context, Metrics, Scalar, ndarrays_to_parameters
from flwr.server import ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from sklearn.metrics import log_loss

from config import RunConf
from experiments.federated.utils import get_model_parameters, load_test_dataset, set_model_params, UNIQUE_LABELS
from src.evalution import evaluatelog_result
from task import create_sgd_and_instantiate_parameters


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Dict[str, Scalar]:
    """Compute weighted average.

    It is a generic implementation that averages only over floats and ints and drops the
    other data types of the Metrics.
    """
    # num_samples_list can represent the number of samples
    # or the number of batches depending on the client
    num_samples_list = [n_batches for n_batches, _ in metrics]
    num_samples_sum = sum(num_samples_list)
    metrics_lists: Dict[str, List[float]] = {}
    for num_samples, all_metrics_dict in metrics:
        #  Calculate each metric one by one
        for single_metric, value in all_metrics_dict.items():
            if isinstance(value, (float, int)):
                metrics_lists[single_metric] = []
        # Just one iteration needed to initialize the keywords
        break

    for num_samples, all_metrics_dict in metrics:
        # Calculate each metric one by one
        for single_metric, value in all_metrics_dict.items():
            # Add weighted metric
            if isinstance(value, (float, int)):
                metrics_lists[single_metric].append(float(num_samples * value))

    weighted_metrics: Dict[str, Scalar] = {}
    for metric_name, metric_values in metrics_lists.items():
        weighted_metrics[metric_name] = sum(metric_values) / num_samples_sum

    return weighted_metrics


def fit_config(server_round: int):
    print(f"Federated Round {server_round} başladı!")  # Iterasyon takibi
    return {"server_round": server_round}


def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    x_test, y_test = load_test_dataset()

    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        set_model_params(model, parameters)
        predicted = model.predict(x_test.values)
        res = evaluatelog_result(y_true=y_test, y_prediction=predicted, model_name=RunConf.model,
                                 logger=RunConf.logger, prob=None)
        if server_round == RunConf.num_server_rounds:
            pickle.dump(model, open(RunConf.model_name, "wb"))
        return 0.1, res

    return evaluate


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behavior."""

    model = create_sgd_and_instantiate_parameters(penalty=RunConf.penalty,
                                                  max_iter=RunConf.max_iter,
                                                  loss=RunConf.loss,
                                                  tol=RunConf.tol,
                                                  random_state=RunConf.random_state)
    ndarrays = get_model_parameters(model)
    global_model_init = ndarrays_to_parameters(ndarrays)

    # Define the strategy
    min_available_clients = RunConf.min_available_clients
    strategy = FedAvg(
        min_available_clients=min_available_clients,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=global_model_init,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_config
    )

    # Construct ServerConfig
    num_rounds = RunConf.num_server_rounds
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

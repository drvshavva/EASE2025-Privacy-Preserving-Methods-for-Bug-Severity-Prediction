"""xgboost_quickstart: A Flower / XGBoost app."""

import warnings

import xgboost as xgb
from flwr.client import Client
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Status,
)
from flwr.common.context import Context

from config import RunConf
from experiments.federated.utils import load_data_from_local, compute_entropy
from task import transform_dataset_to_dmatrix

warnings.filterwarnings("ignore", category=UserWarning)


# Define Flower Client and client_fn
class FlowerClient(Client):
    def __init__(
            self,
            train_dmatrix,
            valid_dmatrix,
            num_train,
            num_val,
            num_local_round,
            params,
            train_method,
    ):
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix
        self.num_train = num_train
        self.num_val = num_val
        self.num_local_round = num_local_round
        self.params = params
        self.train_method = train_method

    def _local_boost(self, bst_input):
        # Update trees based on local training data.
        for i in range(self.num_local_round):
            bst_input.update(self.train_dmatrix, bst_input.num_boosted_rounds())

        # Bagging: extract the last N=num_local_round trees for sever aggregation
        # Cyclic: return the entire model
        bst = (
            bst_input[
                bst_input.num_boosted_rounds()
                - self.num_local_round : bst_input.num_boosted_rounds()
            ]
            if self.train_method == "bagging"
            else bst_input
        )

        return bst

    def fit(self, ins: FitIns) -> FitRes:
        global_round = int(ins.config["global_round"])
        if global_round == 1:
            # First round local training
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
                evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")],
            )
        else:
            bst = xgb.Booster(params=self.params)
            global_model = bytearray(ins.parameters.tensors[0])

            # Load global model into booster
            bst.load_model(global_model)

            # Local training
            bst = self._local_boost(bst)

        # Save model
        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)

        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=self.num_train,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # Load global model
        bst = xgb.Booster(params=self.params)
        para_b = bytearray(ins.parameters.tensors[0])
        bst.load_model(para_b)

        # Run evaluation
        eval_results = bst.eval_set(
            evals=[(self.valid_dmatrix, "valid")],
            iteration=bst.num_boosted_rounds() - 1,
        )

        mlogloss = round(float(eval_results.split("\t")[1].split(":")[1]), 4)
        y_true = self.valid_dmatrix.get_label()
        entropy = compute_entropy(y_true)

        model_params_json = bst.save_raw("json").decode("utf-8")

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=0.0,
            num_examples=self.num_val,
            metrics={"mlogloss": mlogloss, "entropy": entropy, "model_params": model_params_json},
        )


def client_fn(context: Context):
    # Load model and data
    partition_id = context.node_config["partition-id"]
    X_train, y_train, X_test, y_test = load_data_from_local(partition_id, RunConf.data_path)
    train_dmatrix = transform_dataset_to_dmatrix(X_train.values, y_train.values)
    valid_dmatrix = transform_dataset_to_dmatrix(X_test.values, y_test.values)
    num_local_round = RunConf.num_rounds
    params = RunConf.params
    train_method = RunConf.train_method
    # Return Client instance
    return FlowerClient(
        train_dmatrix,
        valid_dmatrix,
        len(X_train),
        len(X_test),
        num_local_round,
        params,
        train_method
    )

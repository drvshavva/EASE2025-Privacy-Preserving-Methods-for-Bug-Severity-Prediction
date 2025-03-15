import pickle

import numpy as np
import pandas as pd

from experiments.federated.xgboost.task import transform_dataset_to_dmatrix
from experiments.federated.utils import load_test_dataset
from src.evalution import evaluatelog_result
from src.logger import Logger

MODEL_DIR = r"models/"

experiment_names = ['Linear SVC', 'Logistic_Regression', 'PAC', 'SGD', 'XGBoost']
seeds = [42, 44, 46, 48, 50]

experiment_name = experiment_names[0]
#data_dist = '_noniid'
data_dist = ''


def run_test_for_seeds():
    results = []

    for seed in seeds:
        model_name = f"{MODEL_DIR}{experiment_name}_{seed}{data_dist}.pkl"
        log_name = f"{experiment_name}_{seed}"

        try:
            with open(model_name, "rb") as f:
                model = pickle.load(f)

            x_test, y_test = load_test_dataset()

            if experiment_name == 'XGBoost':
                valid_dmatrix = transform_dataset_to_dmatrix(x_test.values, y_test.values)
                predictions = model.predict(valid_dmatrix)
                predictions = np.argmax(predictions, axis=1)
            else:
                predictions = model.predict(x_test)

            eval_result = evaluatelog_result(
                y_true=y_test,
                y_prediction=predictions,
                model_name=log_name,
                logger=Logger(log_filename=experiment_name),
                prob=None
            )

            eval_result["seed"] = seed  # Sonuçlara seed ekle
            results.append(eval_result)
        except Exception as e:
            print(f"Error processing {model_name}: {e}")

    df_results = pd.DataFrame(results)
    excel_filename = f"logs/{experiment_name}{data_dist}_results.xlsx"
    df_results.to_excel(excel_filename, index=False)


if __name__ == '__main__':
    run_test_for_seeds()

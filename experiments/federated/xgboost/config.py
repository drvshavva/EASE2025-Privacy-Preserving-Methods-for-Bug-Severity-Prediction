from src.logger import Logger


class RunConf:
    # federated learning parametreleri
    num_rounds = 25
    num_server_rounds = 25
    min_available_clients = 3

    # model parameters
    seed = 12
    params = {
        "objective": "multi:softprob",
        "num_class": 4,
        "eval_metric": "mlogloss",
        "num_parallel_tree": None,
        "n_estimators": 200,
        "tree_method": "hist",
        "random_state": 42,
    }
    train_method = 'bagging'
    # save params
    model = "XGBoost"
    data_path = "iid"
    model_name = fr"models/{model}_{str(params['random_state'])}.pkl"

    logger = Logger(log_filename=f"{model}_metrics.txt")

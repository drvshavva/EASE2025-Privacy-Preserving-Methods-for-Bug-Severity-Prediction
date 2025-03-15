from src.logger import Logger


class RunConf:
    # federated learning parameters
    num_rounds = 25
    num_server_rounds = 25
    min_available_clients = 3

    # model parameters
    penalty = "l2"
    loss = "squared_hinge"
    c = 0.5
    max_iter = 5000
    random_state = 50
    tol = 1e-4

    # save config
    model = "Linear SVC"
    data_path = "noniid"  # "robust_iid"
    model_name = fr"models/{model}_{str(random_state)}_noniid.pkl"

    logger = Logger(log_filename=f"{model}_metrics.txt")

from src.logger import Logger


class RunConf:
    # federated configs
    num_rounds = 25
    num_server_rounds = 25
    min_available_clients = 3

    # model parameters
    c = 1.0
    max_iter = 1000
    loss = "hinge"
    tol = 1e-3
    random_state = 50

    # save params
    data_path = "iid"
    model = "PAC"
    model_name = fr"models/{model}_{str(random_state)}.pkl"

    logger = Logger(log_filename=f"{model}_metrics.txt")

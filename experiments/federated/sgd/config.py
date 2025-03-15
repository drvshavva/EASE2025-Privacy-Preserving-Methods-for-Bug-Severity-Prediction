from src.logger import Logger


class RunConf:
    # federated learning parameters
    num_rounds = 25
    num_server_rounds = 25
    min_available_clients = 3

    # model parameters
    penalty = "l2"
    loss = 'hinge'
    max_iter = 1000
    tol = 1e-3
    random_state = 50

    # save parameters
    data_path = "noniid"
    model = "SGD"
    model_name = fr"models/{model}_{str(random_state)}_noniid.pkl"

    logger = Logger(log_filename=f"{model}_metrics.txt")

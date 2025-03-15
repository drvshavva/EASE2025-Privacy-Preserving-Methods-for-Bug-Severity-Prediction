from src.logger import Logger


class RunConf:
    # federated learning parameters
    num_rounds = 25
    num_server_rounds = 25
    min_available_clients = 3

    # model parameters
    penalty = "l2"
    c = 0.1
    max_iter = 1000
    solver = "saga"
    multi_class = "multinomial"
    random_state = 50

    # save params
    data_path = "iid"
    model = "Logistic_Regression"
    model_name = fr"models/{model}_{str(random_state)}.pkl"

    logger = Logger(log_filename=f"{model}_metrics.txt")

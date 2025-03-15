from sklearn.linear_model import LogisticRegression

from experiments.federated.utils import TRAIN_COLS, UNIQUE_LABELS, set_initial_params


def create_log_reg_and_instantiate_parameters(penalty, c, max_iter, solver, multi_class, random_state):
    model = LogisticRegression(
        penalty=penalty,
        C=c,
        max_iter=max_iter,
        warm_start=True,
        solver=solver,
        random_state=random_state,
        multi_class=multi_class,
        class_weight="balanced",
    )
    set_initial_params(model, n_features=130, n_classes=len(UNIQUE_LABELS))
    return model

from sklearn.linear_model import SGDClassifier

from experiments.federated.utils import TRAIN_COLS, UNIQUE_LABELS, set_initial_params


def create_sgd_and_instantiate_parameters(penalty, max_iter, loss, tol, random_state):
    model = SGDClassifier(
        penalty=penalty,
        max_iter=max_iter,
        warm_start=True,
        loss=loss,
        tol=tol,
        random_state=random_state,
        class_weight="balanced",
    )
    set_initial_params(model, n_features=130, n_classes=len(UNIQUE_LABELS))
    return model

from sklearn.linear_model import PassiveAggressiveClassifier

from experiments.federated.utils import TRAIN_COLS, UNIQUE_LABELS, set_initial_params


def create_pac_and_instantiate_parameters(c, max_iter, loss, tol, random_state):
    model = PassiveAggressiveClassifier(
        warm_start=True,
        C= c,
        max_iter=max_iter,
        class_weight="balanced",
        loss=loss,
        random_state=random_state,
        tol=tol,
    )
    set_initial_params(model, n_features=130, n_classes=len(UNIQUE_LABELS))
    return model

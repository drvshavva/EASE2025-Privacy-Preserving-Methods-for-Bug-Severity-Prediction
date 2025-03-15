from sklearn.svm import LinearSVC

from experiments.federated.utils import TRAIN_COLS, UNIQUE_LABELS, set_initial_params


def create_svc_and_instantiate_parameters(penalty, loss, c, max_iter, tol, random_state):
    model = LinearSVC(
        penalty=penalty,
        loss=loss,
        dual=True,
        C=c,
        class_weight="balanced",
        max_iter=max_iter,
        tol=tol,
        random_state=random_state
    )
    set_initial_params(model, n_features=130, n_classes=len(UNIQUE_LABELS))
    return model

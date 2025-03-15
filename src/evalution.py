from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, matthews_corrcoef, \
    cohen_kappa_score
from imblearn.metrics import geometric_mean_score


def evaluatelog_result(y_true, y_prediction, model_name, logger, prob):
    logger.info("************ " + model_name + " ************")
    eval_result = evaluate_result(y_true=y_true, y_prediction=y_prediction, prob=prob)
    for key in sorted(eval_result.keys()):
        logger.info(f"  {key} = {str(eval_result[key])}")
    return eval_result


def evaluate_result(y_true, y_prediction, prob):
    f1_weighted = f1_score(y_true, y_prediction, average='weighted')
    f1_per_class = f1_score(y_true, y_prediction, average=None).tolist()
    accuracy = accuracy_score(y_true, y_prediction)
    precision = precision_score(y_true, y_prediction, average='weighted')
    recall = recall_score(y_true, y_prediction, average='weighted')
    if prob is not None:
        roc_uac = roc_auc_score(y_true, prob, average='weighted', multi_class='ovo')
    else:
        roc_uac = 0
    mcc = matthews_corrcoef(y_true, y_prediction)
    kappa_score = cohen_kappa_score(y_true, y_prediction)
    gmean = geometric_mean_score(y_true, y_prediction, average='weighted')

    eval_result = {
        "eval_f1": float(f1_weighted),
        "eval_f1_class1": float(f1_per_class[0]),
        "eval_f1_class2": float(f1_per_class[1]),
        "eval_f1_class3": float(f1_per_class[2]),
        "eval_f1_class4": float(f1_per_class[3]),
        "eval_acc": float(accuracy),
        "eval_precision": float(precision),
        "eval_recall": float(recall),
        "eval_ROC-UAC": float(roc_uac),
        "eval_mcc": float(mcc),
        "eval_cohen_kappa_score": float(kappa_score),
        "eval_gmean": float(gmean)
    }
    return eval_result

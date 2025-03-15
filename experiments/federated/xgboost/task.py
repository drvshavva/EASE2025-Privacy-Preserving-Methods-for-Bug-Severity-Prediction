import xgboost as xgb


def transform_dataset_to_dmatrix(x, y) -> xgb.core.DMatrix:
    """Transform dataset to DMatrix format for xgboost."""
    new_data = xgb.DMatrix(x, label=y)
    return new_data



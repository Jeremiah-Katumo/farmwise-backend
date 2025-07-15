import pandas as pd
from ..dl_models import predict_yield_dl, predict_pest_dl, build_and_train_yield_model, build_train_pest_model


def retrain_yield_model(X_train, y_train, X_val, y_val):
    return build_and_train_yield_model(X_train, y_train, X_val, y_val)

def retrain_pest_model(X_train, y_train, X_val, y_val):
    return build_train_pest_model(X_train, y_train, X_val, y_val)
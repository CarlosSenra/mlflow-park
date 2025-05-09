from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np

def metricas_regressao(y_true:pd.Series,y_pred:np.array):
    """retorna e printa as metricas de avaliacao para um modelo de regressao

    Args:
        y_true: Valores reais de y
        y_pred: Valores preditos de y

    Returns:
        mae, mse, rmse, r_2: (Tuple[float, float, float, float]) mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
    """
    mae = mean_absolute_error(y_true,y_pred)
    mse = mean_squared_error(y_true,y_pred)
    rmse = (mse)**(1/2)
    r_2 = r2_score(y_true,y_pred)

    return mae, mse, rmse, r_2
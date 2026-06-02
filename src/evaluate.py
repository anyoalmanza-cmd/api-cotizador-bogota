import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from src.config import DEVICE

def calcular_metricas(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return {
        "MAE_millones": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_millones": float(root_mean_squared_error(y_true, y_pred)),
        "MAPE_porcentaje": float(mean_absolute_percentage_error(y_true, y_pred) * 100)
    }

def predecir_ensamble(modelos, X_scaled, y_mean, y_std):
    import torch
    X_t = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
    preds = []
    with torch.no_grad():
        for model in modelos:
            pred_scaled = model(X_t).cpu().numpy()
            pred_log = (pred_scaled * y_std) + y_mean
            pred_real = np.expm1(pred_log)
            preds.append(pred_real)
    preds = np.array(preds)
    return preds.mean(axis=0), preds.std(axis=0), preds

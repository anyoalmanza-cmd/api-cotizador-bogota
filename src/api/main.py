import sys
import os
import pandas as pd
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

# Forzamos a Python a reconocer la raíz absoluta del proyecto antes de cualquier otra importación
RAIZ_PROYECTO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if RAIZ_PROYECTO not in sys.path:
    sys.path.insert(0, RAIZ_PROYECTO)

# Ahora las importaciones modulares funcionarán de manera impecable
from src.config import DEVICE
from src.model import RedInmueblesMLP
from src.evaluate import predecir_ensamble
from src.api.schemas import RequestInmueble

meta_prod, columnas_modelo, redes_activas, metricas_modelo = None, None, [], {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global meta_prod, columnas_modelo, redes_activas, metricas_modelo
    
    # Rutas relativas directas desde la raíz del despliegue
    RUTA_METADATOS = os.path.join(RAIZ_PROYECTO, "data_meta.pth")
    RUTA_PESOS = os.path.join(RAIZ_PROYECTO, "ensemble_latest.pth")
    
    if not os.path.exists(RUTA_METADATOS):
        raise RuntimeError(f"No se encontró el archivo de metadatos binarios. Buscado en: {RUTA_METADATOS}")

    meta_prod = torch.load(RUTA_METADATOS, map_location=DEVICE)
    columnas_modelo = meta_prod["columnas_X"]
    metricas_modelo = meta_prod.get("metricas_test", {})
    pesos = torch.load(RUTA_PESOS, map_location=DEVICE)

    redes_activas = []
    for i in range(meta_prod["num_modelos"]):
        net = RedInmueblesMLP(len(columnas_modelo)).to(DEVICE)
        net.load_state_dict(pesos[i])
        net.eval()
        redes_activas.append(net)
    yield
    redes_activas.clear()

app = FastAPI(title="API Inmobiliaria Bogotá", version="2.5", lifespan=lifespan)

@app.post("/predecir")
def predecir(inmueble: RequestInmueble):
    if columnas_modelo is None:
        raise HTTPException(status_code=503, detail="El modelo no ha sido cargado correctamente en el servidor.")
        
    df = pd.DataFrame(0.0, index=[0], columns=columnas_modelo)
    df["Área"] = inmueble.area
    df["Habitaciones"] = inmueble.habitaciones
    df["Baños"] = inmueble.banos

    t_col, b_col, u_col = f"Tipo_{inmueble.tipo}", f"Barrio_{inmueble.barrio}", f"UPZ_{inmueble.upz}"
    if t_col not in df.columns or b_col not in df.columns or u_col not in df.columns:
        raise HTTPException(status_code=400, detail="Categoría geográfica o estructural no válida.")

    df[t_col], df[b_col], df[u_col] = 1.0, 1.0, 1.0
    scaled = (df.values.astype(np.float32) - meta_prod["X_mean"].numpy()) / meta_prod["X_std"].numpy()

    mean, std, pred_all = predecir_ensamble(redes_activas, scaled, meta_prod["y_mean"].numpy(), meta_prod["y_std"].numpy())

    return {
        "status": "success",
        "precio_estimado_millones_cop": round(float(mean[0][0]), 2),
        "incertidumbre_ensamble_millones": round(float(std[0][0]), 2),
        "predicciones_individuales_millones": [round(float(x[0][0]), 2) for x in pred_all],
        "metricas_test_modelo": {
            "MAE_millones": round(float(metricas_modelo.get("MAE_millones", 0.0)), 2),
            "RMSE_millones": round(float(metricas_modelo.get("RMSE_millones", 0.0)), 2),
            "MAPE_porcentaje": round(float(metricas_modelo.get("MAPE_porcentaje", 0.0)), 2),
        }
    }

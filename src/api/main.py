import sys
import os
import pandas as pd
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Configuración de rutas
RAIZ_PROYECTO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if RAIZ_PROYECTO not in sys.path:
    sys.path.insert(0, RAIZ_PROYECTO)

from src.model import RedInmueblesMLP
from src.evaluate import predecir_ensamble
from src.api.schemas import RequestInmueble

# Variables globales
meta_prod, columnas_modelo, redes_activas, metricas_modelo = None, None, [], {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    redes_activas.clear()

app = FastAPI(title="API Inmobiliaria Bogotá", version="2.5", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

def cargar_modelos_si_es_necesario():
    global meta_prod, columnas_modelo, redes_activas, metricas_modelo
    if meta_prod is not None: return 
    
    ruta_meta = os.path.join(RAIZ_PROYECTO, "data_meta.pth")
    ruta_pesos = os.path.join(RAIZ_PROYECTO, "ensemble_latest.pth")
    
    if not os.path.exists(ruta_meta):
        raise RuntimeError("No se encontraron los archivos del modelo.")

    meta_prod = torch.load(ruta_meta, map_location=torch.device('cpu'))
    columnas_modelo = meta_prod["columnas_X"]
    metricas_modelo = meta_prod.get("metricas_test", {})
    pesos = torch.load(ruta_pesos, map_location=torch.device('cpu'))

    for i in range(meta_prod["num_modelos"]):
        net = RedInmueblesMLP(len(columnas_modelo))
        net.load_state_dict(pesos[i])
        net.eval()
        redes_activas.append(net)

@app.post("/predecir")
def predecir(inmueble: RequestInmueble):
    cargar_modelos_si_es_necesario()
        
    df = pd.DataFrame(0.0, index=[0], columns=columnas_modelo)
    df["Área"] = inmueble.area
    df["Habitaciones"] = inmueble.habitaciones
    df["Baños"] = inmueble.banos

    # Normalización: Limpiamos los strings para asegurar que coincidan
    # Usamos .strip() para quitar espacios accidentales
    tipo_norm = inmueble.tipo.strip()
    barrio_norm = inmueble.barrio.strip()
    upz_norm = inmueble.upz.strip()

    t_col = f"Tipo_{tipo_norm}"
    b_col = f"Barrio_{barrio_norm}"
    u_col = f"UPZ_{upz_norm}"
    
    # Validación con depuración detallada
    if t_col not in df.columns or b_col not in df.columns or u_col not in df.columns:
        # Aquí verás exactamente qué estamos intentando buscar vs lo que existe
        raise HTTPException(
            status_code=400, 
            detail=f"Modelo espera estas columnas: {list(df.columns[:5])}... Pero intentamos buscar: T:{t_col}, B:{b_col}, U:{u_col}"
        )

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

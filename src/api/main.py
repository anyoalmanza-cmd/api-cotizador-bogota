import sys
import os
import logging
import traceback
import pandas as pd
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAIZ_PROYECTO = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
if RAIZ_PROYECTO not in sys.path: sys.path.insert(0, RAIZ_PROYECTO)

from src.model import RedInmueblesMLP
from src.evaluate import predecir_ensamble
from src.api.schemas import RequestInmueble

# Variables globales
meta_prod, columnas_modelo, redes_activas = None, None, []

@asynccontextmanager
async def lifespan(app: FastAPI):
    cargar_modelos()
    yield
    redes_activas.clear()

app = FastAPI(title="API Inmobiliaria", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def cargar_modelos():
    global meta_prod, columnas_modelo, redes_activas
    if meta_prod is not None: return
    
    ruta_meta = os.path.join(RAIZ_PROYECTO, "data_meta.pth")
    ruta_pesos = os.path.join(RAIZ_PROYECTO, "ensemble_latest.pth")
    
    meta_prod = torch.load(ruta_meta, map_location=torch.device('cpu'))
    columnas_modelo = list(meta_prod["columnas_X"])
    pesos = torch.load(ruta_pesos, map_location=torch.device('cpu'))

    for i in range(meta_prod["num_modelos"]):
        net = RedInmueblesMLP(len(columnas_modelo))
        net.load_state_dict(pesos[i])
        net.eval()
        redes_activas.append(net)

@app.post("/predecir")
def predecir(inmueble: RequestInmueble):
    try:
        # Preparación de variables
        t_col = f"Tipo_{inmueble.tipo.strip()}"
        b_col = f"Barrio_{inmueble.barrio.strip()}"
        u_col = f"UPZ_{inmueble.upz.strip()}"
        
        # Validación
        if t_col not in columnas_modelo or b_col not in columnas_modelo or u_col not in columnas_modelo:
            return {"status": "error", "mensaje": "Categoría no encontrada en el modelo."}

        # Construcción DataFrame
        df = pd.DataFrame(0.0, index=[0], columns=columnas_modelo)
        df["Área"] = inmueble.area
        df["Habitaciones"] = inmueble.habitaciones
        df["Baños"] = inmueble.banos
        df[t_col], df[b_col], df[u_col] = 1.0, 1.0, 1.0
        
        # Predicción
        scaled = (df.values.astype('float32') - meta_prod["X_mean"].numpy()) / meta_prod["X_std"].numpy()
        mean, std, _ = predecir_ensamble(redes_activas, scaled, meta_prod["y_mean"].numpy(), meta_prod["y_std"].numpy())

        # Extracción segura de métricas desde meta_prod
        # Aseguramos conversión a float nativo para evitar error de serialización JSON
        met = meta_prod.get("metricas", {})
        
        return {
            "status": "success",
            "precio_estimado_millones_cop": round(float(mean[0][0]), 2),
            "incertidumbre_millones": round(float(std[0][0]), 2),
            "metricas": {
                "MAE": round(float(met.get("MAE_millones", 0.0)), 2),
                "RMSE": round(float(met.get("RMSE_millones", 0.0)), 2),
                "MAPE": round(float(met.get("MAPE_porcentaje", 0.0)), 2)
            }
        }

    except Exception as e:
        return {
            "status": "error_critico",
            "mensaje": str(e),
            "traceback": traceback.format_exc()
        }

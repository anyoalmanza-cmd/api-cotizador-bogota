import sys
import os
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
mapa_normalizado = {}

def normalizar_str(s):
    return s.lower().strip().replace(" ", "").replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global meta_prod, columnas_modelo, redes_activas, mapa_normalizado
    
    ruta_meta = os.path.join(RAIZ_PROYECTO, "data_meta.pth")
    ruta_pesos = os.path.join(RAIZ_PROYECTO, "ensemble_latest.pth")
    
    meta_prod = torch.load(ruta_meta, map_location=torch.device('cpu'))
    columnas_modelo = list(meta_prod["columnas_X"])
    
    for col in columnas_modelo:
        mapa_normalizado[normalizar_str(col)] = col
        
    pesos = torch.load(ruta_pesos, map_location=torch.device('cpu'))
    for p in pesos:
        net = RedInmueblesMLP(len(columnas_modelo))
        net.load_state_dict(p)
        net.eval()
        redes_activas.append(net)
    yield
    redes_activas.clear()

app = FastAPI(title="API Inmobiliaria", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/predecir")
def predecir(inm: RequestInmueble):
    try:
        t_key = normalizar_str(f"tipo_{inm.tipo}")
        b_key = normalizar_str(f"barrio_{inm.barrio}")
        u_key = normalizar_str(f"upz_{inm.upz}")
        
        if t_key not in mapa_normalizado or b_key not in mapa_normalizado or u_key not in mapa_normalizado:
            raise HTTPException(status_code=400, detail="Categoría no encontrada.")

        df = pd.DataFrame(0.0, index=[0], columns=columnas_modelo)
        df["Área"] = float(inm.area)
        df["Habitaciones"] = float(inm.habitaciones)
        df["Baños"] = float(inm.banos)
        df[mapa_normalizado[t_key]] = 1.0
        df[mapa_normalizado[b_key]] = 1.0
        df[mapa_normalizado[u_key]] = 1.0
        
        # Normalización
        scaled = (df.values.astype('float32') - meta_prod["X_mean"].numpy()) / meta_prod["X_std"].numpy()
        
        # Predicción (Devuelve valores en escala logarítmica)
        mean_log, std_log, _ = predecir_ensamble(redes_activas, scaled, meta_prod["y_mean"].numpy(), meta_prod["y_std"].numpy())

        # CORRECCIÓN: Aplicar np.expm1 para deshacer log1p y obtener el valor real en millones
        valor_estimado = float(np.expm1(np.array(mean_log).ravel()[0]))
        valor_incertidumbre = float(np.expm1(np.array(std_log).ravel()[0]))
        
        met = meta_prod.get("metricas", {})
        
        return {
            "status": "success",
            "precio_estimado_millones_cop": round(valor_estimado, 2),
            "incertidumbre_millones": round(valor_incertidumbre, 2),
            "metricas": {
                "MAE": round(float(met.get("MAE_millones", 0.0)), 2),
                "MAPE": round(float(met.get("MAPE_porcentaje", 0.0)), 2)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import sys
import os
import pandas as pd
import numpy as np
import torch
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAIZ_PROYECTO = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
if RAIZ_PROYECTO not in sys.path: sys.path.insert(0, RAIZ_PROYECTO)

from src.model import RedInmueblesMLP
from src.evaluate import predecir_ensamble
from src.api.schemas import RequestInmueble

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
    redes_activas = [RedInmueblesMLP(len(columnas_modelo)) for _ in range(len(pesos))]
    for i, p in enumerate(pesos):
        redes_activas[i].load_state_dict(p)
        redes_activas[i].eval()
    yield

app = FastAPI(title="API Inmobiliaria", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/predecir")
def predecir(inm: RequestInmueble):
    try:
        t_key = normalizar_str(f"tipo_{inm.tipo}")
        b_key = normalizar_str(f"barrio_{inm.barrio}")
        u_key = normalizar_str(f"upz_{inm.upz}")
        
        # 1. Construcción del DataFrame con validación de existencia
        df = pd.DataFrame(0.0, index=[0], columns=columnas_modelo)
        df.loc[0, "Área"] = float(inm.area)
        df.loc[0, "Habitaciones"] = float(inm.habitaciones)
        df.loc[0, "Baños"] = float(inm.banos)
        
        # Validar claves antes de asignar
        if t_key in mapa_normalizado: df.loc[0, mapa_normalizado[t_key]] = 1.0
        if b_key in mapa_normalizado: df.loc[0, mapa_normalizado[b_key]] = 1.0
        if u_key in mapa_normalizado: df.loc[0, mapa_normalizado[u_key]] = 1.0
        
        # 2. Escalamiento
        X_input = df.values.astype(np.float32)
        scaled = (X_input - meta_prod["X_mean"].numpy()) / meta_prod["X_std"].numpy()
        
        # 3. Inferencia
        mean, std, _ = predecir_ensamble(
            redes_activas, scaled, 
            meta_prod["y_mean"].numpy(), 
            meta_prod["y_std"].numpy()
        )
        
        # 4. Cálculo final (reversión de log1p y escala)
        valor_en_millones = np.expm1(float(mean[0][0]))
        valor_final_pesos = valor_en_millones * 1_000_000
        incertidumbre_pesos = float(std[0][0]) * 1_000_000
        
        return {
            "status": "success",
            "precio_estimado_cop": f"${int(valor_final_pesos):,.0f}".replace(",", "."),
            "incertidumbre_cop": f"± ${int(incertidumbre_pesos):,.0f}".replace(",", ".")
        }
    except Exception as e:
        logger.exception("Error en predicción")
        raise HTTPException(status_code=500, detail=f"Error procesando datos: {str(e)}")

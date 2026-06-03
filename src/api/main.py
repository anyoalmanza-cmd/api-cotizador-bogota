import sys
import os
import pandas as pd
import numpy as np
import torch
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Configuración de logs para ver errores en la consola de Render/Uvicorn
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
    s = s.lower().strip().replace(" ", "").replace("á", "a").replace("é", "e")
    s = s.replace("í", "i").replace("ó", "o").replace("ú", "u")
    return s

@asynccontextmanager
async def lifespan(app: FastAPI):
    global meta_prod, columnas_modelo, redes_activas, mapa_normalizado
    
    ruta_meta = os.path.join(RAIZ_PROYECTO, "data_meta.pth")
    ruta_pesos = os.path.join(RAIZ_PROYECTO, "ensemble_latest.pth")
    
    # Cargar metadatos
    meta_prod = torch.load(ruta_meta, map_location=torch.device('cpu'))
    columnas_modelo = list(meta_prod["columnas_X"])
    
    # Crear mapa de normalización
    for col in columnas_modelo:
        mapa_normalizado[normalizar_str(col)] = col
        
    # Cargar redes
    pesos = torch.load(ruta_pesos, map_location=torch.device('cpu'))
    redes_activas = []
    for p in pesos:
        net = RedInmueblesMLP(len(columnas_modelo))
        net.load_state_dict(p)
        net.eval()
        redes_activas.append(net)
    
    logger.info(f"API iniciada. Columnas cargadas: {len(columnas_modelo)}")
    yield
    redes_activas.clear()

app = FastAPI(title="API Inmobiliaria", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/predecir")
def predecir(inm: RequestInmueble):
    try:
        # 1. Normalizar claves de entrada
        t_key = normalizar_str(f"tipo_{inm.tipo}")
        b_key = normalizar_str(f"barrio_{inm.barrio}")
        u_key = normalizar_str(f"upz_{inm.upz}")
        
        # 2. Validación estricta
        if t_key not in mapa_normalizado or b_key not in mapa_normalizado or u_key not in mapa_normalizado:
            error_msg = f"Categoría no válida. Verifique: {inm.tipo}, {inm.barrio}, {inm.upz}"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        # 3. Construcción del DataFrame (One-Hot)
        # Esto asegura que todas las columnas que el modelo espera existan
        df = pd.DataFrame(0.0, index=[0], columns=columnas_modelo)
        df["Área"] = float(inm.area)
        df["Habitaciones"] = float(inm.habitaciones)
        df["Baños"] = float(inm.banos)
        
        # Asignar 1.0 a las categorías seleccionadas
        df.loc[0, mapa_normalizado[t_key]] = 1.0
        df.loc[0, mapa_normalizado[b_key]] = 1.0
        df.loc[0, mapa_normalizado[u_key]] = 1.0
        
        # 4. Predicción
        # Convertir a numpy asegurando los mismos tipos que en el entrenamiento
        X_input = df.values.astype(np.float32)
        scaled = (X_input - meta_prod["X_mean"].numpy()) / meta_prod["X_std"].numpy()
        
        mean, std, _ = predecir_ensamble(
            redes_activas, 
            scaled, 
            meta_prod["y_mean"].numpy(), 
            meta_prod["y_std"].numpy()
        )
        
        return {
            "status": "success",
            "precio_estimado_millones_cop": round(float(mean[0][0]), 2),
            "incertidumbre_millones": round(float(std[0][0]), 2)
        }
    except Exception as e:
        logger.exception("Error interno en la predicción")
        raise HTTPException(status_code=500, detail=str(e))

import sys
import os
import pandas as pd
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Configuración de rutas para que funcione en cualquier entorno
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
    """Limpia strings para que coincidan con las columnas del modelo (sin tildes/espacios)."""
    s = s.lower().strip().replace(" ", "").replace("á", "a").replace("é", "e")
    s = s.replace("í", "i").replace("ó", "o").replace("ú", "u")
    return s

@asynccontextmanager
async def lifespan(app: FastAPI):
    global meta_prod, columnas_modelo, redes_activas, mapa_normalizado
    
    # Cargar modelos y metadatos
    ruta_meta = os.path.join(RAIZ_PROYECTO, "data_meta.pth")
    ruta_pesos = os.path.join(RAIZ_PROYECTO, "ensemble_latest.pth")
    
    meta_prod = torch.load(ruta_meta, map_location=torch.device('cpu'))
    columnas_modelo = list(meta_prod["columnas_X"])
    
    # Crear mapa de normalización
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
        # 1. Normalizar las llaves de búsqueda
        t_key = normalizar_str(f"tipo_{inm.tipo}")
        b_key = normalizar_str(f"barrio_{inm.barrio}")
        u_key = normalizar_str(f"upz_{inm.upz}")
        
        # 2. Validar existencia
        if t_key not in mapa_normalizado or b_key not in mapa_normalizado or u_key not in mapa_normalizado:
            raise HTTPException(status_code=400, detail=f"Categoría no válida. Verifica el Barrio o la UPZ.")

        # 3. Construcción del DataFrame de entrada (One-Hot)
        df = pd.DataFrame(0.0, index=[0], columns=columnas_modelo)
        df["Área"] = float(inm.area)
        df["Habitaciones"] = float(inm.habitaciones)
        df["Baños"] = float(inm.banos)
        df[mapa_normalizado[t_key]] = 1.0
        df[mapa_normalizado[b_key]] = 1.0
        df[mapa_normalizado[u_key]] = 1.0
        
        # 4. Predicción robusta
        scaled = (df.values.astype('float32') - meta_prod["X_mean"].numpy()) / meta_prod["X_std"].numpy()
        mean, std, _ = predecir_ensamble(redes_activas, scaled, meta_prod["y_mean"].numpy(), meta_prod["y_std"].numpy())

        # 5. Corrección del error de dimensionalidad: Convertir arrays a escalares puros
        valor_estimado = float(np.array(mean).flatten()[0])
        valor_incertidumbre = float(np.array(std).flatten()[0])
        
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
        # Log del error detallado para que sepas qué pasó en el servidor
        print(f"Error critico: {str(e)}") 
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

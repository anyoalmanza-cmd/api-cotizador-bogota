import sys
import os
import pandas as pd
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from src.model import RedInmueblesMLP
from src.evaluate import predecir_ensamble
from src.api.schemas import RequestInmueble

# Variables globales
meta_prod, columnas_modelo, redes_activas = None, None, []
mapa_normalizado = {} # Diccionario para encontrar columnas sin importar tildes/espacios

@asynccontextmanager
async def lifespan(app: FastAPI):
    cargar_modelos()
    yield
    redes_activas.clear()

app = FastAPI(title="API Inmobiliaria", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def cargar_modelos():
    global meta_prod, columnas_modelo, redes_activas, mapa_normalizado
    if meta_prod is not None: return
    
    # Rutas relativas
    ruta_meta = os.path.join(os.path.dirname(__file__), "..", "data_meta.pth")
    ruta_pesos = os.path.join(os.path.dirname(__file__), "..", "ensemble_latest.pth")
    
    meta_prod = torch.load(ruta_meta, map_location=torch.device('cpu'))
    columnas_modelo = list(meta_prod["columnas_X"])
    
    # CREAR MAPA: 'tipo_casa' -> 'Tipo_Casa'
    for col in columnas_modelo:
        key = col.lower().replace(" ", "").replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
        mapa_normalizado[key] = col

    pesos = torch.load(ruta_pesos, map_location=torch.device('cpu'))
    for p in pesos:
        net = RedInmueblesMLP(len(columnas_modelo))
        net.load_state_dict(p)
        net.eval()
        redes_activas.append(net)

@app.post("/predecir")
def predecir(inmueble: RequestInmueble):
    try:
        # 1. Normalizar inputs del usuario
        def norm(val):
            return val.lower().strip().replace(" ", "").replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")

        t_key = f"tipo_{norm(inmueble.tipo)}"
        b_key = f"barrio_{norm(inmueble.barrio)}"
        u_key = f"upz_{norm(inmueble.upz)}"

        # 2. Validar contra el mapa
        if t_key not in mapa_normalizado or b_key not in mapa_normalizado or u_key not in mapa_normalizado:
            raise HTTPException(status_code=400, detail=f"Categoría no válida. Verifica: {inmueble.barrio} o {inmueble.upz}")

        # 3. Construcción del DataFrame
        df = pd.DataFrame(0.0, index=[0], columns=columnas_modelo)
        df["Área"] = float(inmueble.area)
        df["Habitaciones"] = float(inmueble.habitaciones)
        df["Baños"] = float(inmueble.banos)
        
        # Activar las columnas usando los nombres reales hallados en el mapa
        df[mapa_normalizado[t_key]] = 1.0
        df[mapa_normalizado[b_key]] = 1.0
        df[mapa_normalizado[u_key]] = 1.0
        
        # 4. Predicción
        scaled = (df.values.astype('float32') - meta_prod["X_mean"].numpy()) / meta_prod["X_std"].numpy()
        mean, std, _ = predecir_ensamble(redes_activas, scaled, meta_prod["y_mean"].numpy(), meta_prod["y_std"].numpy())

        met = meta_prod.get("metricas", {})
        
        return {
            "status": "success",
            "precio_estimado_millones_cop": round(float(mean[0]), 2),
            "incertidumbre_millones": round(float(std[0]), 2),
            "metricas": {
                "MAE": round(float(met.get("MAE_millones", 0.0)), 2),
                "MAPE": round(float(met.get("MAPE_porcentaje", 0.0)), 2)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

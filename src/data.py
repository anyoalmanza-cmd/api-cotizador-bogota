import os
import kagglehub
import pandas as pd
import numpy as np
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split
from src.config import SEED

def normalizar_nombre(c):
    # Esta lógica DEBE ser idéntica a la función normalizar_str de tu main.py
    s = c.lower().strip().replace(" ", "").replace("á", "a").replace("é", "e")
    s = s.replace("í", "i").replace("ó", "o").replace("ú", "u")
    return s

def preparar_datos():
    file_path = "inmuebles_bogota.csv"
    df_raw = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "pablobravo73/real-estate-bogota", file_path)
    
    df_raw["Valor_Limpio"] = df_raw["Valor"].astype(str).str.replace(r"[$\s\.]", "", regex=True).astype(float)
    df_raw["Valor_Millones"] = df_raw["Valor_Limpio"] / 1_000_000
    
    cols_str = ["Tipo", "Barrio", "UPZ"]
    for col in cols_str:
        df_raw[col] = df_raw[col].astype(str).str.strip()

    df = df_raw[["Tipo", "Área", "Habitaciones", "Baños", "Barrio", "UPZ", "Valor_Millones"]].dropna()

    # 1. Crear dummies
    df_modelo = pd.get_dummies(df, columns=cols_str, dtype=float)
    
    # 2. Renombrar columnas con la normalización exacta de main.py
    mapa_normalizado = {}
    nuevas_cols = []
    for c in df_modelo.columns:
        if c == "Valor_Millones":
            nuevas_cols.append(c)
        else:
            nombre_limpio = normalizar_nombre(c)
            nuevas_cols.append(nombre_limpio)
            # Mapeamos cómo se debe llamar la columna en la API
            mapa_normalizado[nombre_limpio] = nombre_limpio

    df_modelo.columns = nuevas_cols
    
    # 3. Ordenamiento alfabético estricto (CRÍTICO)
    columnas_X = sorted([c for c in df_modelo.columns if c != "Valor_Millones"])
    df_modelo = df_modelo[columnas_X + ["Valor_Millones"]]

    # 4. Cálculo de parámetros
    X = df_modelo[columnas_X].values.astype(np.float32)
    y = df_modelo["Valor_Millones"].values.astype(np.float32).reshape(-1, 1)
    y_log = np.log1p(y)
    
    X_train, _, y_train, _ = train_test_split(X, y_log, test_size=0.30, random_state=SEED)
    X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
    X_std[X_std == 0] = 1.0

    # Retorno con normalización de Y neutralizada (mean=0, std=1)
    # para que main.py no altere la salida de la red.
    return {
        "columnas_X": columnas_X,
        "X_mean": X_mean,
        "X_std": X_std,
        "y_mean": np.array([0.0], dtype=np.float32), 
        "y_std": np.array([1.0], dtype=np.float32),
        "mapa_normalizado": mapa_normalizado
    }

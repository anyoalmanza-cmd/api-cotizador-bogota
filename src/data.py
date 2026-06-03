import kagglehub
import pandas as pd
import numpy as np
import torch
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split
from src.config import SEED

def normalizar_str(s):
    """Lógica universal para normalizar nombres de columnas."""
    return s.lower().strip().replace(" ", "").replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")

def preparar_datos():
    file_path = "inmuebles_bogota.csv"
    df_raw = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "pablobravo73/real-estate-bogota", file_path)
    
    # Preprocesamiento
    df_raw["Valor_Limpio"] = df_raw["Valor"].astype(str).str.replace(r"[$\s\.]", "", regex=True).astype(float)
    df_raw["Valor_Millones"] = df_raw["Valor_Limpio"] / 1_000_000
    
    # Asegurar tipos string
    for col in ["Tipo", "Barrio", "UPZ"]:
        df_raw[col] = df_raw[col].astype(str).str.strip()

    df = df_raw[["Tipo", "Área", "Habitaciones", "Baños", "Barrio", "UPZ", "Valor_Millones"]].dropna()

    # 1. Crear dummies
    df_modelo = pd.get_dummies(df, columns=["Tipo", "Barrio", "UPZ"], dtype=float)

    # 2. RENOMBRADO Y NORMALIZACIÓN: 
    # Obligamos a que las columnas se llamen como la API espera (ej: tipo_apartamento)
    def mapear_columna(c):
        if c == "Valor_Millones" or c in ["Área", "Habitaciones", "Baños"]:
            return c
        # 'Tipo_Apartamento' -> 'tipo_apartamento'
        return normalizar_str(c.replace("_", "_")) 

    df_modelo.columns = [mapear_columna(c) for c in df_modelo.columns]

    # 3. ORDENAMIENTO ALFABÉTICO ESTRICTO
    # Esto es VITAL: si el orden cambia, el modelo falla.
    columnas_X = sorted([c for c in df_modelo.columns if c != "Valor_Millones"])
    df_modelo = df_modelo[columnas_X + ["Valor_Millones"]]

    # 4. Preparación de tensores
    X = df_modelo[columnas_X].values.astype(np.float32)
    y = np.log1p(df_modelo["Valor_Millones"].values.astype(np.float32).reshape(-1, 1))

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=SEED)

    # Calcular estadísticas de escalado sobre train
    X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
    X_std[X_std == 0] = 1.0

    return {
        "X_train": (X_train - X_mean) / X_std,
        "X_val": (X_val - X_mean) / X_std,
        "X_test": (X_test - X_mean) / X_std,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "columnas_X": columnas_X, # Lista ordenada alfabéticamente
        "X_mean": X_mean,
        "X_std": X_std,
        "y_mean": y_train.mean(axis=0),
        "y_std": y_train.std(axis=0)
    }

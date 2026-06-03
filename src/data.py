import os
import kagglehub
import pandas as pd
import numpy as np
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split
from src.config import SEED

def normalizar_str(s):
    # La misma lógica que tiene main.py
    s = s.lower().strip().replace(" ", "").replace("á", "a").replace("é", "e")
    s = s.replace("í", "i").replace("ó", "o").replace("ú", "u")
    return s

def preparar_datos():
    file_path = "inmuebles_bogota.csv"
    df_raw = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "pablobravo73/real-estate-bogota", file_path)
    
    df_raw["Valor_Limpio"] = df_raw["Valor"].astype(str).str.replace(r"[$\s\.]", "", regex=True).astype(float)
    df_raw["Valor_Millones"] = df_raw["Valor_Limpio"] / 1_000_000
    df_raw["Barrio"] = df_raw["Barrio"].astype(str).str.strip()

    barrios_validos = df_raw["Barrio"].value_counts()
    barrios_validos = barrios_validos[barrios_validos > 10].index.tolist()
    df = df_raw[df_raw["Barrio"].isin(barrios_validos)][["Área", "Habitaciones", "Baños", "Barrio", "Valor_Millones"]].dropna()

    # Creación de dummies
    df_modelo = pd.get_dummies(df, columns=["Barrio"], dtype=float)

    # NORMALIZACIÓN DE NOMBRES PARA COINCIDIR CON LA API
    nuevas_cols = []
    for c in df_modelo.columns:
        if c == "Valor_Millones" or c in ["Área", "Habitaciones", "Baños"]:
            nuevas_cols.append(c)
        else:
            # Convierte 'Barrio_Chapinero Alto' -> 'barrio_chapineroalto'
            nuevas_cols.append(normalizar_str(c))
    
    df_modelo.columns = nuevas_cols
    
    columnas_X = [c for c in df_modelo.columns if c != "Valor_Millones"]
    X = df_modelo[columnas_X].values.astype(np.float32)
    y = np.log1p(df_modelo["Valor_Millones"].values.astype(np.float32).reshape(-1, 1))

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=SEED)

    X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
    X_std[X_std == 0] = 1.0

    return {
        "X_train": (X_train - X_mean) / X_std,
        "X_val": (X_val - X_mean) / X_std,
        "X_test": (X_test - X_mean) / X_std,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "y_test_real": np.expm1(y_test),
        "columnas_X": columnas_X,
        "X_mean": X_mean, "X_std": X_std,
        "y_mean": y_train.mean(axis=0), "y_std": y_train.std(axis=0)
    }

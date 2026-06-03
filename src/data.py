import kagglehub
import pandas as pd
import numpy as np
import torch
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split

def normalizar_nombre(s):
    # Esta es la ÚNICA forma de limpiar nombres para que coincidan con la API
    return s.lower().strip().replace(" ", "").replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")

def preparar_datos():
    file_path = "inmuebles_bogota.csv"
    df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "pablobravo73/real-estate-bogota", file_path)
    
    df["Valor_Limpio"] = df["Valor"].astype(str).str.replace(r"[$\s\.]", "", regex=True).astype(float)
    df["Valor_Millones"] = df["Valor_Limpio"] / 1_000_000
    
    # Crear dummies
    df_modelo = pd.get_dummies(df[["Tipo", "Área", "Habitaciones", "Baños", "Barrio", "UPZ", "Valor_Millones"]].dropna(), 
                               columns=["Tipo", "Barrio", "UPZ"], dtype=float)

    # Renombrar columnas a formato normalizado
    df_modelo.columns = [normalizar_nombre(c) if c not in ["Área", "Habitaciones", "Baños", "Valor_Millones"] else c for c in df_modelo.columns]
    
    # Ordenar estrictamente
    columnas_X = sorted([c for c in df_modelo.columns if c != "Valor_Millones"])
    df_modelo = df_modelo[columnas_X + ["Valor_Millones"]]

    X = df_modelo[columnas_X].values.astype(np.float32)
    y = np.log1p(df_modelo["Valor_Millones"].values.astype(np.float32).reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    meta = {
        "columnas_X": columnas_X,
        "X_mean": torch.tensor(X_train.mean(axis=0)),
        "X_std": torch.tensor(X_train.std(axis=0) + 1e-7),
        "y_mean": torch.tensor(y_train.mean()),
        "y_std": torch.tensor(y_train.std() + 1e-7)
    }
    torch.save(meta, "data_meta.pth")
    return X_train, y_train, X_test, y_test, meta

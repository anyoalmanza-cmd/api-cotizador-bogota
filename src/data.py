import os
import kagglehub
import pandas as pd
import numpy as np
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split
from src.config import SEED

def preparar_datos():
    file_path = "inmuebles_bogota.csv"

    print("[DATA] Descargando dataset desde Kaggle...")
    df_raw = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "pablobravo73/real-estate-bogota",
        file_path,
    )

    df_raw["Valor_Limpio"] = (
        df_raw["Valor"]
        .astype(str)
        .str.replace(r"[\$\s\.]", "", regex=True)
        .astype(float)
    )
    df_raw["Valor_Millones"] = df_raw["Valor_Limpio"] / 1_000_000

    df_raw["Barrio"] = df_raw["Barrio"].astype(str).str.strip()
    df_raw["Tipo"] = df_raw["Tipo"].astype(str).str.strip()
    df_raw["UPZ"] = df_raw["UPZ"].astype(str).str.strip()

    mapa_barrio_upz = df_raw.groupby("Barrio")["UPZ"].first().to_dict()

    barrios_validos = df_raw["Barrio"].value_counts()
    barrios_validos = barrios_validos[barrios_validos > 10].index.tolist()

    df = df_raw[df_raw["Barrio"].isin(barrios_validos)].copy()
    df = df[["Tipo", "Área", "Habitaciones", "Baños", "Barrio", "UPZ", "Valor_Millones"]].dropna()
    df = df[(df["Área"] > 0) & (df["Habitaciones"] > 0) & (df["Baños"] > 0) & (df["Valor_Millones"] > 0)].copy()

    lista_barrios = sorted(df["Barrio"].unique().tolist())
    lista_tipos = sorted(df["Tipo"].unique().tolist())
    lista_upz = sorted(df["UPZ"].unique().tolist())

    df_modelo = pd.get_dummies(df, columns=["Tipo", "Barrio", "UPZ"], dtype=float)

    columnas_X = [c for c in df_modelo.columns if c != "Valor_Millones"]
    X = df_modelo[columnas_X].values.astype(np.float32)
    y = df_modelo["Valor_Millones"].values.astype(np.float32).reshape(-1, 1)

    y_log = np.log1p(y)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y_log, test_size=0.30, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=SEED)

    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std == 0] = 1.0

    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_std[y_std == 0] = 1.0

    return {
        "X_train": ((X_train - X_mean) / X_std).astype(np.float32),
        "X_val": ((X_val - X_mean) / X_std).astype(np.float32),
        "X_test": ((X_test - X_mean) / X_std).astype(np.float32),
        "y_train": ((y_train - y_mean) / y_std).astype(np.float32),
        "y_val": ((y_val - y_mean) / y_std).astype(np.float32),
        "y_test": ((y_test - y_mean) / y_std).astype(np.float32),
        "y_test_real": np.expm1(y_test).astype(np.float32),
        "columnas_X": columnas_X,
        "lista_barrios": lista_barrios,
        "lista_tipos": lista_tipos,
        "lista_upz": lista_upz,
        "mapa_barrio_upz": mapa_barrio_upz,
        "X_mean": X_mean, "X_std": X_std, "y_mean": y_mean, "y_std": y_std
    }

import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from src.config import DEVICE, METADATOS_PATH, PESOS_PATH, MODEL_REGISTRY
from src.data import preparar_datos
from src.model import RedInmueblesMLP, EarlyStopping
from src.evaluate import predecir_ensamble, calcular_metricas

def entrenar_una_red(X_train, y_train, X_val, y_val, input_dim, epochs=500, lr=1e-3):
    model = RedInmueblesMLP(input_dim).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)

    early_stopper = EarlyStopping(patience=40, min_delta=1e-4)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t)

        if early_stopper.step(val_loss.item(), model):
            break

    model.load_state_dict(early_stopper.best_state)
    return model, early_stopper.best_loss

def pipeline_completo(num_modelos=5, epochs=500):
    data = preparar_datos()
    X_train = data["X_train"]
    y_train = data["y_train"]
    input_dim = X_train.shape[1]
    modelos = []

    print(f"[PIPELINE] Entrenando ensamble (Bagging) de {num_modelos} Modelos...")
    for i in range(num_modelos):
        idx = np.random.choice(np.arange(len(X_train)), size=len(X_train), replace=True)
        model, _ = entrenar_una_red(X_train[idx], y_train[idx], data["X_val"], data["y_val"], input_dim, epochs)
        modelos.append(model)
        print(f"-> Red {i+1} entrenada con éxito.")

    pred_mean, _, _ = predecir_ensamble(modelos, data["X_test"], data["y_mean"], data["y_std"])
    metricas = calcular_metricas(data["y_test_real"], pred_mean)

    os.makedirs(MODEL_REGISTRY, exist_ok=True)

    metadatos = {
        "X_mean": torch.tensor(data["X_mean"]), "X_std": torch.tensor(data["X_std"]),
        "y_mean": torch.tensor(data["y_mean"]), "y_std": torch.tensor(data["y_std"]),
        "columnas_X": data["columnas_X"], "lista_barrios": data["lista_barrios"],
        "lista_tipos": data["lista_tipos"], "lista_upz": data["lista_upz"],
        "mapa_barrio_upz": data["mapa_barrio_upz"], "num_modelos": num_modelos,
        "metricas_test": metricas
    }

    torch.save(metadatos, METADATOS_PATH)
    torch.save([m.cpu().state_dict() for m in modelos], PESOS_PATH)
    print(f"[OK] Entrenamiento finalizado. MAPE en Test: {metricas['MAPE_porcentaje']:.2f}%")

if __name__ == "__main__":
    pipeline_completo(num_modelos=5, epochs=150)

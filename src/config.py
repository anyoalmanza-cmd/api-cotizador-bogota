import os
import torch

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_REGISTRY = os.path.join(BASE_DIR, "model_registry")

METADATOS_PATH = os.path.join(MODEL_REGISTRY, "metadatos_latest.pt")
PESOS_PATH = os.path.join(MODEL_REGISTRY, "ensamble_latest.pth")

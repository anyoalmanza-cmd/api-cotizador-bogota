# Bogota Real Estate Machine Learning Pipeline

Proyecto modular para estimación de precios inmobiliarios en Bogotá mediante un ensamble Deep Learning (PyTorch) servido con FastAPI.

---

## Requisitos previos

- Python 3.10 o superior
- Cuenta en [Kaggle](https://www.kaggle.com) (para descargar el dataset al entrenar)

---

## Instalación en Windows

### 1. Crear el entorno virtual

```powershell
python -m venv .venv
```

### 2. Activar el entorno virtual

```powershell
.\.venv\Scripts\Activate.ps1
```

> Si PowerShell bloquea la ejecución de scripts, ejecuta primero:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 3. Instalar dependencias

```powershell
pip install -r requirements.txt
```

---

## Uso

### Entrenar el modelo

Descarga el dataset de Kaggle y entrena el ensamble de redes. La primera vez puede pedir credenciales de Kaggle.

```powershell
python -m src.train
```

Los modelos se guardan en `model_registry/`.

> **Credenciales de Kaggle:** ve a kaggle.com → Settings → API → "Create New Token".
> Coloca el archivo `kaggle.json` descargado en `C:\Users\<TU_USUARIO>\.kaggle\`.

### Lanzar la API

```powershell
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

### Correr tests

```powershell
pytest tests/
```

---

## Probar la API

Con el servidor corriendo, abre en el navegador:

```
http://127.0.0.1:8000/docs
```

O desde PowerShell:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predecir" -Method POST -ContentType "application/json" -Body '{
  "tipo": "Apartamento",
  "area": 65,
  "habitaciones": 3,
  "banos": 2,
  "barrio": "Cedritos",
  "upz": "Cedros"
}'
```

---

## Estructura del proyecto

```
api-cotizador-bogota/
├── src/
│   ├── api/
│   │   ├── main.py        # Servidor FastAPI
│   │   └── schemas.py     # Modelos de entrada/salida
│   ├── config.py          # Rutas y configuración global
│   ├── data.py            # Descarga y preparación del dataset
│   ├── model.py           # Arquitectura de la red neuronal
│   ├── train.py           # Pipeline de entrenamiento
│   └── evaluate.py        # Métricas y predicción del ensamble
├── model_registry/        # Pesos y metadatos del modelo entrenado
├── requirements.txt
└── README.md
```

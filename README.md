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

---

## Documentación detallada del proyecto

Esta sección amplía la estructura y explica el propósito de cada componente principal.

- **`src/config.py`**: configuración global del proyecto. Define `SEED`, `DEVICE` (CPU/GPU) y rutas a `MODEL_REGISTRY`, `PESOS_PATH` y `METADATOS_PATH`.
- **`src/data.py`**: descarga y preprocesamiento del dataset (usa `kagglehub`). Limpia la columna `Valor`, crea `Valor_Millones`, filtra barrios poco representados, aplica one-hot a `Tipo`/`Barrio`/`UPZ`, hace los splits train/val/test y calcula medias/std para normalización.
- **`src/model.py`**: contiene la clase `RedInmueblesMLP` (MLP secuencial) y `EarlyStopping` (patience y restauración del mejor estado).
- **`src/train.py`**: funciones de entrenamiento. `entrenar_una_red` entrena una MLP con `MSELoss` y `Adam` usando EarlyStopping. `pipeline_completo` entrena un ensamble por bagging (por defecto 5 modelos), evalúa en test y guarda `metadatos_latest.pt` y `ensamble_latest.pth` en `model_registry/`.
- **`src/evaluate.py`**: `predecir_ensamble` aplica cada modelo del ensamble al conjunto escalado, desescala y aplica `expm1` para volver a la escala real; devuelve media, desviación y todas las predicciones. `calcular_metricas` devuelve MAE, RMSE y MAPE.
- **`src/api/main.py`**: aplicación FastAPI. Al inicio carga `metadatos_latest.pt` y las listas de pesos; reconstruye cada `RedInmueblesMLP` con el `input_dim` apropiado y expone endpoints:
  - `GET /opciones`: devuelve listas de `tipos`, `barrios` y `upzs` para el frontend.
  - `POST /predecir`: valida y normaliza el input, construye vector one-hot, escala con medias/std cargadas, llama a `predecir_ensamble` y devuelve precio estimado, incertidumbre y métricas.
- **`src/api/schemas.py`**: modelos Pydantic (`RequestInmueble`) con campos `tipo`, `area`, `habitaciones`, `banos`, `barrio`, `upz`.
- **`src/api/templates/index.html`**: UI simple que consulta `/opciones` y envía una petición a `/predecir` para mostrar el resultado.

## Modelo: arquitectura de la red neuronal

Resumen:
- Tipo: MLP (red neuronal feedforward totalmente conectada).
- Entrada: vector de dimensión `input_dim` (columnas numéricas + columnas one-hot por `Tipo`/`Barrio`/`UPZ`).
- Salida: un escalar (valor objetivo en espacio log-transformado y normalizado durante el entrenamiento).

Arquitectura exacta (`RedInmueblesMLP` en `src/model.py`):

- Capa 1: `Linear(input_dim, 128)` → `ReLU` → `Dropout(0.10)`
- Capa 2: `Linear(128, 64)` → `ReLU` → `Dropout(0.10)`
- Capa salida: `Linear(64, 1)`

Detalles de entrenamiento:
- Objetivo transformado: se entrena sobre `y_log = log1p(y)`; en predicción se aplica `expm1` tras desescalar.
- Pérdida: `MSELoss`.
- Optimizador: `Adam` con `weight_decay=1e-4`.
- Ensamble: `pipeline_completo` entrena N modelos por bootstrap y usa la media de las predicciones reescaladas como resultado; la desviación entre modelos se usa como medida de incertidumbre.
- EarlyStopping: `patience=40`, `min_delta=1e-4`.

### Diagrama de la red

El siguiente diagrama muestra la topología de la red (sustituir `input_dim` por el número real de features en tiempo de ejecución):

```mermaid
flowchart LR
  A[Input<br/>dim = input_dim] --> B[Dense 128]
  B --> Bact[ReLU]
  Bact --> Bdrop[Dropout 0.10]
  Bdrop --> C[Dense 64]
  C --> Cact[ReLU]
  Cact --> Cdrop[Dropout 0.10]
  Cdrop --> D[Dense 1<br/>(Output)]
  style A fill:#f3f4f6,stroke:#222
  style B fill:#e6f2ff
  style C fill:#e6f2ff
  style D fill:#fff7e6
```

### Obtener `input_dim` y parámetros exactos
Si ya entrenaste y tienes `model_registry/metadatos_latest.pt`, el `input_dim` puede obtenerse del tamaño de `X_mean` o de las columnas guardadas (`columnas_X`).

---

Si quieres, puedo añadir una imagen SVG/PNG del diagrama al repositorio y calcular el número total de parámetros del modelo automáticamente.

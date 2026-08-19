# MLOPS_TF — Predicción de masa invariante de pares de electrones

**Asignatura:** Operaciones con Máquinas I

**Integrantes:**

a2413, César Hernán Ruggeri

a2521, Andrea Tatiana Duran

a2525, Pablo David Gorosito

a2542, Federico Tombesi

## 1. Objetivo del proyecto

Este proyecto aplica técnicas de aprendizaje automático a datos de física de partículas. El objetivo es predecir la masa invariante de pares de electrones a partir del dataset `dielectron.csv`.

Como modelo principal se utiliza `GradientBoostingRegressor`, con búsqueda de hiperparámetros mediante Optuna. El proyecto implementa una arquitectura de MLOps que automatiza la preparación de datos, el entrenamiento, el registro de modelos y su exposición como servicio de predicción.

## 2. Componentes del proyecto

| Componente | Función |
|---|---|
| Notebook (`Trabajo_Final_—_Aprendizaje_de_Máquina.ipynb`) | Exploración del dataset, preprocesamiento y evaluación inicial del modelo. |
| Apache Airflow (`airflow/dags/process_etl_split.py`) | DAG `process_etl_split`: descarga el dataset desde Kaggle, lo sube a MinIO, lo limpia, lo divide en train/test, busca hiperparámetros con Optuna, entrena el modelo campeón y lo registra en MLflow. |
| MinIO | Almacenamiento S3-compatible. Bucket `data` con las etapas `raw/`, `processed/`, `final/train`, `final/test` y `data_info/data.json`; bucket `mlflow` con los artefactos de los modelos. |
| MLflow + SQLite | Registra parámetros, métricas y modelos de los experimentos. Mantiene el modelo campeón bajo el alias `thebest` en el registro `dielectron_mass_regressor`. |
| FastAPI (`dockerfiles/fastapi/app.py`) | Expone el endpoint `POST /predict/`. Carga el modelo desde el registro de MLflow (alias `thebest`); si no está disponible, usa un fallback local (`MODEL_URI` o pickle). |
| Streamlit (`dockerfiles/fastapi/streamlit_app.py`) | Interfaz web simple que arma el payload y consume la API de FastAPI. |
| Docker Compose | Levanta y orquesta toda la infraestructura: Postgres, Redis, los servicios de Airflow, MinIO, MLflow, FastAPI y Streamlit. |

## 3. Flujo de datos y entrenamiento

El DAG `process_etl_split` implementa las siguientes tareas, en orden:

1. **`download_and_upload`**: descarga el dataset desde Kaggle (`fedesoriano/cern-electron-collision-data`) y lo sube a `s3://data/raw/dielectron.csv`.
2. **`etl_data`**: lee el archivo raw, elimina nulos/duplicados y las columnas `Run` y `Event` (identificadores, no variables predictoras), guarda el resultado en `s3://data/processed/processed_data.csv` y actualiza los metadatos en `s3://data/data_info/data.json`.
3. **`split_data`**: separa variables explicativas de la variable objetivo (`M`) y divide en train/test con semilla reproducible, guardando las particiones en:

```text
s3://data/final/train/dielectron_X_train.csv
s3://data/final/train/dielectron_y_train.csv
s3://data/final/test/dielectron_X_test.csv
s3://data/final/test/dielectron_y_test.csv
```

4. **`train_model`**: lee las particiones desde MinIO, busca hiperparámetros con Optuna, entrena el `GradientBoostingRegressor` campeón, calcula métricas (RMSE, MAE, R²) sobre el set de test, y registra el modelo en MLflow bajo el nombre `dielectron_mass_regressor` con el alias `thebest`.

La API de FastAPI y la interfaz de Streamlit consumen ese modelo registrado en MLflow.

> **Nota:** esta descripción refleja lo que implementa el código del DAG. La ejecución end-to-end depende de tener acceso a Kaggle y recursos suficientes de Docker; verificalo en tu entorno antes de asumir que el pipeline corre sin intervención.

## 4. Puesta en marcha

### Requisitos

- Docker y Docker Compose.
- Conexión a Internet para descargar el dataset desde Kaggle (`kagglehub`). Si el dataset requiere autenticación, puede ser necesario configurar credenciales de Kaggle en el entorno donde corre el worker de Airflow.

### Configuración inicial

1. Clonar el repositorio y situarse en la raíz del proyecto:

```bash
git clone <repo-url>
cd MLOPS_TF
```

2. Crear el archivo de variables de entorno a partir del ejemplo y completar los valores locales:

```bash
cp .env.example .env
```

3. Crear los archivos de secrets de Airflow a partir de los ejemplos:

```bash
cp airflow/secrets/variables.example.yaml airflow/secrets/variables.yaml
cp airflow/secrets/connections.example.yaml airflow/secrets/connections.yaml
```

### Levantar la infraestructura

Construir las imágenes y levantar todos los servicios:

```bash
docker compose up -d --build
```

Esto levanta Postgres, Redis, los servicios de Airflow (`apiserver`, `scheduler`, `dag-processor`, `worker`, `triggerer`, `init`), MinIO, la creación de buckets (`data` y `mlflow`), MLflow, FastAPI y Streamlit.

Verificar que los servicios estén saludables:

```bash
docker compose ps
```

### Ejecutar el pipeline de Airflow

1. Abrir Airflow en el navegador: `http://localhost:8080` (usuario/contraseña definidos en `.env`).
2. Desencadenar el DAG `process_etl_split` desde la interfaz.
3. Alternativamente, usar el CLI de Airflow (el servicio `airflow-cli` está definido bajo el profile `debug`):

```bash
docker compose --profile debug run --rm airflow-cli airflow dags trigger process_etl_split
```

4. Confirmar que las particiones se generaron en MinIO bajo `s3://data/final/` y que el modelo `dielectron_mass_regressor` quedó registrado en MLflow con el alias `thebest`.

### Acceso a servicios

| Servicio | URL | Notas |
|---|---|---|
| Airflow | http://localhost:8080 | Usuario/contraseña definidos en `.env`. |
| MinIO (consola) | http://localhost:9001 | Access/secret key definidos en `.env`. |
| MLflow | http://localhost:5000 | Tracking server y model registry. |
| FastAPI | http://localhost:8800 | Documentación interactiva en `/docs`. |
| Streamlit | http://localhost:8501 | Formulario de predicción. |

### Apagar el entorno

```bash
docker compose down
```

Agregar `-v` si además se quieren eliminar los volúmenes (datos de Postgres, MinIO y MLflow):

```bash
docker compose down -v
```

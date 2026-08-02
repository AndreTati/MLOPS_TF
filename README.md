# MLOPS_TF — Predicción de masa invariante de pares de electrones

**Asignatura:** Operaciones con Máquinas I

**Integrantes:**

a2413, César Hernán Ruggeri

a2521, Andrea Tatiana Duran

a2525, Pablo David Gorosito

a2542, Federico Tombesi

## 1. Objetivo del proyecto

Este proyecto aplica técnicas de aprendizaje automático a datos de física de partículas. El objetivo es predecir la masa invariante de pares de electrones a partir del dataset `dielectron.csv`.

Como modelo principal se utiliza `GradientBoostingRegressor`. Además del desarrollo del modelo, el proyecto incorpora una arquitectura inicial de MLOps para organizar, automatizar y hacer reproducible parte del flujo de datos y entrenamiento.

## 2. Alcance de esta entrega parcial

La entrega presenta avances en:

- Exploración, preprocesamiento y evaluación inicial del dataset en una notebook.
- Entrenamiento modular mediante `app/app.py`.
- Orquestación de la preparación de datos con Apache Airflow.
- Almacenamiento de datos procesados en MinIO, compatible con S3.
- Infraestructura local reproducible mediante Docker Compose.
- Registro local de experimentos de entrenamiento mediante MLflow con SQLite.

> **Estado actual:** se validaron dos circuitos complementarios:
> 1. Airflow → MinIO, para la preparación automatizada y el almacenamiento de datos.
> 2. Entrenamiento → MLflow/SQLite, para el entrenamiento reproducible y el registro de métricas.
>
> Aún no se validó la integración completa de punta a punta, es decir, que `app/app.py` consuma directamente las particiones `train/test` producidas por Airflow y almacenadas en MinIO.

## 3. Componentes principales

| Componente | Función |
|---|---|
| Notebook | Exploración del dataset, preprocesamiento, experimentación y evaluación inicial del modelo. |
| `app/app.py` | Script modular para entrenar, evaluar y guardar artefactos del modelo. |
| Apache Airflow | Orquesta la descarga, limpieza y división reproducible de los datos. |
| MinIO | Almacena los datos en las etapas `raw`, `processed` y `final`. |
| MLflow + SQLite | Registra parámetros, métricas y modelos de los experimentos de entrenamiento. |
| Docker Compose | Levanta la infraestructura local requerida por Airflow, MinIO y sus servicios auxiliares. |

## 4. Flujo de datos validado

El DAG `process_etl_split` implementa el siguiente flujo:

1. Descarga el dataset de origen.
2. Guarda los datos originales en MinIO.
3. Realiza tareas de limpieza y preparación.
4. Excluye las variables `Run` y `Event`, que actúan como identificadores y no como variables predictoras.
5. Separa variables explicativas y variable objetivo.
6. Divide los datos en conjuntos de entrenamiento y prueba mediante una semilla reproducible.
7. Guarda las particiones finales en MinIO.

La ejecución del DAG fue validada correctamente en el entorno local.

Los archivos generados se almacenan en el bucket `data` bajo las siguientes rutas:

```text
final/train/dielectron_X_train.csv
final/train/dielectron_y_train.csv
final/test/dielectron_X_test.csv
final/test/dielectron_y_test.csv

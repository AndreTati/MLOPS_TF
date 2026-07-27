# MLOPS_TF
# MLOPS_TF

## Estado actual del proyecto — entrega parcial

Este documento describe el estado de avance del proyecto de Operaciones con Máquina. Su objetivo es documentar lo que se encuentra implementado, diferenciarlo de lo que todavía debe revisarse y dejar explicitada la hoja de ruta hacia la entrega final.

> **Importante:** al momento de redactar este README no se afirma que el pipeline funcione de extremo a extremo. El repositorio contiene avances en experimentación, entrenamiento modular, orquestación e infraestructura, pero todavía deben verificarse y/o conectarse varias piezas para confirmar un flujo integrado y reproducible.

## 1. Integrantes

a2413, César Hernán Ruggeri

a2512, Armando Tomás Civini

a2521, Andrea Tatiana Duran

a2525, Pablo David Gorosito

a2542, Federico Tombesi


## 2. Objetivo del proyecto

El proyecto desarrolla un modelo de aprendizaje automático para predecir la **masa invariante de pares de electrones** a partir de datos de eventos de física de partículas.

El trabajo busca avanzar desde un experimento de machine learning hacia una estructura inicial de MLOps que incluya:

- desarrollo y evaluación del modelo;
- separación entre datos originales, procesados y particiones de entrenamiento/prueba;
- orquestación de tareas mediante Apache Airflow;
- almacenamiento de datos mediante MinIO, utilizando una interfaz compatible con S3;
- ejecución reproducible de servicios mediante Docker Compose.

La entrega parcial documenta el estado alcanzado hasta el momento. No pretende presentar todavía un sistema productivo completo.

## 3. Problema de aprendizaje automático

### Variable objetivo

La variable objetivo es `M`, asociada a la masa invariante del par de electrones.

### Tratamiento de variables

El desarrollo excluye `Run` y `Event` como variables predictoras, ya que funcionan como identificadores del evento y no como características físicas destinadas al aprendizaje. La variable `M` se utiliza como objetivo y no como feature.

### Modelo seleccionado

El modelo elegido es **Gradient Boosting Regressor**. La notebook incluye comparación de modelos, búsqueda de hiperparámetros con Optuna, evaluación y validación cruzada.

En la ejecución cuyos resultados quedaron conservados en la notebook, el modelo optimizado alcanzó aproximadamente:

- RMSE: `3,44`;
- R²: `0,9815`.

Estos valores documentan esa ejecución concreta. No deben interpretarse como una garantía de que cualquier ejecución futura producirá exactamente los mismos resultados, especialmente mientras no estén completamente fijadas las versiones, semillas, rutas y artefactos del flujo.

## 4. Estructura del repositorio

La estructura actual separa parcialmente las etapas de experimentación, entrenamiento, orquestación e infraestructura:

```text
MLOPS_TF/
├── Trabajo_Final_—_Aprendizaje_de_Máquina.ipynb
├── datasets/
│   └── dielectron.csv
├── app/
│   └── app.py
├── airflow/
│   ├── dags/
│   │   └── process_etl_split.py
│   └── secrets/
│       ├── connections.yaml
│       └── variables.yaml
├── dockerfiles/
│   ├── airflow/
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── app/
│       └── requirements.txt
├── docker-compose.yaml
├── .env
└── README.md
```

### Función de cada componente

| Componente | Función actual |
|---|---|
| `Trabajo_Final_—_Aprendizaje_de_Máquina.ipynb` | Experimentación, análisis exploratorio, preprocesamiento, comparación, optimización y evaluación de modelos. |
| `datasets/dielectron.csv` | Copia local del conjunto de datos utilizada por el flujo local. |
| `app/app.py` | Script modular para cargar datos, preprocesar, optimizar, entrenar y evaluar el modelo. No es todavía una API. |
| `airflow/dags/process_etl_split.py` | DAG para descargar, limpiar, dividir y almacenar los datos en MinIO. |
| `airflow/secrets/` | Archivos de configuración de conexiones y variables de Airflow para el entorno local. |
| `dockerfiles/airflow/` | Construcción de la imagen personalizada de Airflow y declaración de sus dependencias. |
| `dockerfiles/app/requirements.txt` | Dependencias previstas para el script de entrenamiento. Actualmente no está asociada a un Dockerfile ni a un servicio de Compose. |
| `docker-compose.yaml` | Definición de servicios de Airflow, PostgreSQL, Redis/Valkey y MinIO. |
| `.env` | Variables de configuración utilizadas por Docker Compose en el entorno local. Debe revisarse antes de compartir el repositorio. |

## 5. Notebook de experimentación

La notebook representa la etapa de desarrollo del modelo y contiene, entre otros elementos:

- descripción del problema y del dataset;
- análisis exploratorio de datos;
- limpieza y preprocesamiento;
- separación entre variables predictoras y objetivo;
- comparación de modelos;
- selección de Gradient Boosting;
- optimización de hiperparámetros con Optuna;
- métricas RMSE, MAE y R²;
- validación cruzada;
- interpretación de resultados y conclusiones.

La notebook evidencia que existe un proceso de experimentación desarrollado en la notebook. Sin embargo, el modelo queda en memoria y no se observa actualmente un artefacto entrenado versionado, como `joblib`, `pickle` o un pipeline serializado con sus metadatos.

## 6. Entrenamiento modular: `app/app.py`

`app/app.py` transforma parte de la lógica de la notebook en funciones reutilizables, entre ellas:

- `download_and_load_dataset()`;
- `preprocess_data()`;
- `optimize_hyperparameters()`;
- `train_model()`;
- `evaluate_model()`;
- `save_artifacts()`;
- `main()`.

Esta modularización es un avance importante porque permite separar el código de entrenamiento de la notebook.

El estado actual presenta estas consideraciones:

1. El archivo es un script de entrenamiento, no una aplicación de inferencia ni una API REST.
2. La función `save_artifacts()` existe, pero su uso en `main()` aparece comentado; por eso no debe afirmarse todavía que el modelo se está persistiendo efectivamente.
3. Algunas rutas utilizan una notación dependiente de Windows, por lo que deben revisarse antes de ejecutar el script dentro de un contenedor Linux.
4. `app/app.py` lee el CSV local y realiza su propia división de datos; actualmente no consume los archivos de train/test generados por Airflow en MinIO.

## 7. DAG de Apache Airflow

El archivo `airflow/dags/process_etl_split.py` expresa conceptualmente la siguiente secuencia:

```text
Descarga del dataset
        ↓
Carga del archivo original en MinIO
        ↓
Limpieza y procesamiento
        ↓
Separación de X e y
        ↓
División train/test
        ↓
Carga de las particiones en MinIO
```

El DAG organiza las tareas en dependencias sucesivas: descarga/carga, ETL y división de datos.

### Rutas previstas en MinIO

Según la estructura actual del DAG, el DAG intenta guardar las particiones con una organización equivalente a:

```text
s3://data/final/train/dielectron_X_train.csv
s3://data/final/train/dielectron_y_train.csv
s3://data/final/test/dielectron_X_test.csv
s3://data/final/test/dielectron_y_test.csv
```

La utilización de capas `raw`, `processed` y `final` es adecuada como primera aproximación a una organización de datos para MLOps.

### Dependencia pendiente de verificar

El DAG importa `awswrangler`, pero esa dependencia no aparece declarada en `dockerfiles/airflow/requirements.txt`, que actualmente contiene, entre otras, `pandas`, `kagglehub` y `scikit-learn`.

Por esa razón, la ejecución del DAG todavía debe verificarse en el entorno de Airflow. Antes de presentar el DAG como componente funcional, es necesario comprobar que todas sus dependencias estén instaladas y que la conexión con MinIO/S3 esté correctamente configurada.

## 8. MinIO

MinIO se utiliza como almacenamiento de objetos compatible con S3. Dentro del diseño actual, su función es conservar distintas versiones o etapas de los datos:

- datos originales descargados;
- datos procesados;
- particiones de entrenamiento y prueba.

Esto permite separar el almacenamiento de datos de la lógica de procesamiento y constituye un avance coherente con una arquitectura MLOps.

La integración debe considerarse todavía pendiente de validación completa: no se documenta aquí que la descarga, carga, transformación y posterior consumo por el modelo hayan sido ejecutados exitosamente como una única cadena.

## 9. Docker Compose e infraestructura

`docker-compose.yaml` define una infraestructura local que incluye servicios de apoyo para Airflow y MinIO, entre ellos:

- PostgreSQL para los metadatos de Airflow;
- Redis/Valkey como componente de mensajería;
- API server de Airflow;
- scheduler;
- DAG processor;
- worker;
- triggerer;
- servicio de inicialización de Airflow;
- MinIO;
- creación del bucket de almacenamiento.

También se utilizan volúmenes persistentes, healthchecks, variables de entorno y una red interna para los servicios.

### Alcance actual

Docker Compose demuestra un avance significativo en infraestructura local. Sin embargo:

- el script `app/app.py` no está integrado como servicio de Compose;
- no existe actualmente un `Dockerfile` para `app`;
- no se observa todavía una API FastAPI;
- no se documenta todavía un flujo de inferencia mediante REST;
- no se afirma que todos los servicios hayan sido levantados y probados conjuntamente.

## 10. Forma de ejecución prevista

Las siguientes instrucciones describen las formas de ejecución previstas a partir de la estructura actual. Deben considerarse instrucciones de desarrollo local pendientes de verificación completa.

### 10.1 Experimentación mediante notebook

1. Abrir `Trabajo_Final_—_Aprendizaje_de_Máquina.ipynb`.
2. Instalar las dependencias requeridas por la notebook.
3. Ejecutar las celdas en orden.
4. Verificar la disponibilidad del dataset y de las credenciales de Kaggle si la notebook realiza una descarga.
5. Revisar los resultados de las métricas y las conclusiones.

### 10.2 Ejecución del script modular

Desde el directorio raíz del repositorio, crear un entorno virtual e instalar las dependencias de `dockerfiles/app/requirements.txt`. Luego ejecutar el script con:

```bash
python app/app.py
```

Esta forma de ejecución todavía debe comprobarse en Windows y Linux, porque las rutas actuales no son completamente portables.

### 10.3 Levantamiento de servicios con Docker Compose

Desde el directorio raíz del repositorio:

```bash
docker compose up -d
docker compose ps
```

La salida de `docker compose ps` debe utilizarse para comprobar qué servicios iniciaron correctamente. En caso de errores, revisar los logs de los servicios correspondientes:

```bash
docker compose logs airflow-scheduler
docker compose logs airflow-worker
docker compose logs minio
```

Los nombres exactos de los servicios deben confirmarse contra `docker-compose.yaml` antes de incorporar esta sección al README definitivo.

### 10.4 Ejecución del DAG

Una vez iniciados los servicios, el DAG `process_etl_split.py` debe localizarse en la interfaz de Airflow, habilitarse y ejecutarse manualmente para la prueba inicial.

La verificación debe comprobar, como mínimo:

- que el DAG sea reconocido por Airflow;
- que todas sus importaciones estén disponibles;
- que Kaggle permita la descarga;
- que la conexión con MinIO funcione;
- que se creen los objetos `raw`, `processed` y `final` esperados;
- que no existan errores en ninguna tarea.

Hasta realizar esas comprobaciones, esta sección describe una ejecución prevista, no una ejecución certificada.

## 11. Estado de avance

| Área | Estado documentado |
|---|---|
| Problema y dataset | Definidos en la notebook y representados por `dielectron.csv`. |
| Exploración y preprocesamiento | Implementados en la notebook. |
| Selección de modelo | Gradient Boosting seleccionado y optimizado con Optuna. |
| Evaluación | Se conservan métricas y resultados de una ejecución de la notebook. |
| Código de entrenamiento modular | Iniciado en `app/app.py`. |
| Persistencia del modelo | Preparada mediante una función, pero no confirmada como ejecutada. |
| ETL y división de datos | Modeladas en un DAG de Airflow. |
| Almacenamiento de objetos | MinIO incluido en la infraestructura y utilizado por el DAG según el código. |
| Contenerización | Docker Compose avanzado para servicios de Airflow y MinIO. |
| Integración completa | Pendiente de verificación; no debe presentarse como funcionando de extremo a extremo. |
| Documentación | Este README documenta el estado actual del proyecto correspondiente a la entrega parcial. |

## 12. Pendientes prioritarios para consolidar la entrega parcial

1. Completar los nombres y aportes de todos los integrantes.
2. Verificar y declarar todas las dependencias del DAG, especialmente `awswrangler`.
3. Comprobar las conexiones, variables y credenciales necesarias para MinIO y Kaggle.
4. Ejecutar el DAG y documentar el resultado real, incluyendo errores si los hubiera.
5. Activar y verificar el guardado de un artefacto entrenado, por ejemplo un modelo serializado junto con sus metadatos.
6. Revisar las rutas de `app/app.py` para que sean portables entre Windows y Linux.
7. Definir cuál es el flujo principal: el CSV local o los datos procesados y particionados en MinIO.
8. Incorporar un `.env.example` y revisar la conveniencia de excluir `.env` y otros archivos sensibles del control de versiones.
9. Aclarar en la estructura que `app/app.py` es actualmente un script de entrenamiento y no una API.

## 13. Componentes que pueden quedar para la entrega final

Sin que su ausencia implique necesariamente un incumplimiento de la entrega parcial, pueden desarrollarse posteriormente:

- integración con MLflow;
- seguimiento formal de experimentos;
- registro y versionado de modelos;
- DAG de entrenamiento o reentrenamiento;
- comparación champion/challenger;
- API FastAPI;
- endpoints de predicción;
- Dockerfile y servicio contenerizado para la aplicación;
- inferencia integrada con Docker Compose;
- monitoreo y controles productivos;
- pruebas de integración completas;
- endurecimiento de la seguridad para producción.

## 14. Conclusión

El proyecto presenta un avance concreto y coherente para una entrega parcial: contiene una notebook con experimentación del modelo, un script de entrenamiento modular, un DAG de Airflow, almacenamiento con MinIO y una infraestructura Docker Compose de desarrollo.

La situación actual debe describirse como una **arquitectura inicial de MLOps en proceso de integración**. Todavía no corresponde afirmar que exista un pipeline ejecutable de extremo a extremo, porque el entrenamiento local no consume actualmente los splits generados por Airflow en MinIO, falta verificar dependencias como `awswrangler` y no está confirmado el guardado efectivo del artefacto del modelo.

El próximo paso técnico, luego de aprobar esta documentación, será revisar y asegurar las dependencias del DAG y documentar los resultados reales de su ejecución.

---

**Repositorio:** https://github.com/AndreTati/MLOPS_TF  
**Documento:** README de estado actual para la entrega parcial  
**Fecha de referencia del análisis:** 25 de julio de 2026

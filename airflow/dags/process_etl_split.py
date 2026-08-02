import datetime

from airflow.decorators import dag, task


default_args = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": datetime.timedelta(minutes=1),

}
@dag(
    start_date=datetime.datetime(2026, 6, 1),
    catchup=False,
    tags=["download", "etl"],
    default_args=default_args,
)
def process_etl_split():

    #@task(task_id="download_and_upload")
    @task.virtualenv(
        requirements=["awswrangler","pandas", "kagglehub"],  
        system_site_packages=False
    )
    def download_and_upload() -> str:

        """
        Descarga el dataset de Kaggle (por slug) y lo sube a S3 (bucket 'data', key 'raw/<filename>').
        Devuelve la URI S3 del archivo subido.
        """
        import os
        import pandas as pd
        import kagglehub
        import awswrangler as wr

        dataset_slug: str = "fedesoriano/cern-electron-collision-data"
        filename: str = "dielectron.csv"
        target_dir: str = "/tmp/data"

        data_path = "s3://data/raw/dielectron.csv"

        os.makedirs(target_dir, exist_ok=True)
        os.environ["KAGGLE_CACHE_DIR"] = target_dir

        #Descargar el dataset desde Kaggle
        try:
            print(f"Descargando dataset {dataset_slug} desde Kaggle...")
            download_path = kagglehub.dataset_download(dataset_slug)
            candidate = os.path.join(download_path, filename)
            print(candidate)
            if not os.path.exists(candidate):
                # intentar buscar archivos CSV dentro del directorio descargado
                files = [f for f in os.listdir(download_path) if f.lower().endswith(".csv")]
                if len(files) == 0:
                    raise FileNotFoundError(f"No se encontró {filename} ni otros CSV en {download_path}")
                candidate = os.path.join(download_path, files[0])
                print(f"Archivo solicitado no encontrado; usando {candidate}")
            dataframe = pd.read_csv(candidate)
        except Exception as exc:
            print(f"Error descargando dataset desde Kaggle: {exc}")
            raise
        

        # Subir a S3 
        wr.s3.to_csv(df=dataframe,
                    path=data_path,
                    index=False)
        
        return data_path


    
    @task.virtualenv(
        requirements=["awswrangler"],  
        system_site_packages=False
    )
    def etl_data(s3_uri: str) -> str:
        """
        Realiza un ETL simple sobre el dataset descargado y subido a S3.
        Devuelve la URI S3 del archivo procesado.
        """
        import awswrangler as wr

        
        dataset = wr.s3.read_csv(s3_uri)

        # Procesamiento simple: eliminar filas con valores nulos
        df_cleaned = dataset.dropna()
        print("Filas después de limpiar nulos: %s", df_cleaned.shape)

        #Eliminar columnas innecesarias
        df_cleaned = df_cleaned.drop(columns=['Run', 'Event'], errors='ignore')
        print("Columnas después de eliminar innecesarias: %s", df_cleaned.shape)

        # Guardar el DataFrame procesado en un nuevo archivo CSV en S3
        processed_s3_uri = "s3://data/processed/processed_data.csv"
        wr.s3.to_csv(df=df_cleaned, path=processed_s3_uri, index=False)
        return processed_s3_uri
    

    
    @task.virtualenv(
        requirements=["awswrangler", "scikit-learn"],  
        system_site_packages=False
    )
    def split_data(s3_uri):
        """
        Función para dividir el dataset procesado en conjuntos de entrenamiento y prueba.
        """
        import awswrangler as wr
        from sklearn.model_selection import train_test_split

        target = 'M'
        test_size = 0.2

        # Leer el dataset procesado desde S3
        df = wr.s3.read_csv(s3_uri)

        # Separar target de features
        X = df.drop(columns=target)
        y = df[[target]]

        # Dividir en conjunto de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Guardar los conjuntos de entrenamiento y prueba en S3
        wr.s3.to_csv(df=X_train, path="s3://data/final/train/dielectron_X_train.csv", index=False)
        wr.s3.to_csv(df=y_train, path="s3://data/final/train/dielectron_y_train.csv", index=False)
        wr.s3.to_csv(df=X_test, path="s3://data/final/test/dielectron_X_test.csv", index=False)
        wr.s3.to_csv(df=y_test, path="s3://data/final/test/dielectron_y_test.csv", index=False)

       

    @task.virtualenv(
        requirements=["awswrangler", "scikit-learn", "mlflow", "optuna"],
        system_site_packages=False
    )
    def train_model():
        """
        Lee train/test desde MinIO, busca hiperparámetros con Optuna,
        entrena el modelo campeón y lo registra en MLflow.
        """
        import os
        import awswrangler as wr
        import mlflow
        import optuna
        from mlflow.models import infer_signature
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import numpy as np

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        mlflow.set_experiment("Dielectron Mass Regression")

        X_train = wr.s3.read_csv("s3://data/final/train/dielectron_X_train.csv")
        y_train = wr.s3.read_csv("s3://data/final/train/dielectron_y_train.csv")
        X_test = wr.s3.read_csv("s3://data/final/test/dielectron_X_test.csv")
        y_test = wr.s3.read_csv("s3://data/final/test/dielectron_y_test.csv")

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "random_state": 42,
            }
            model = GradientBoostingRegressor(**params)
            scores = cross_val_score(
                model, X_train, y_train.values.ravel(),
                cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1,
            )
            return -scores.mean()

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=15)

        with mlflow.start_run(run_name="airflow_training_run"):
            best_model = GradientBoostingRegressor(**study.best_params, random_state=42)
            best_model.fit(X_train, y_train.values.ravel())

            y_pred = best_model.predict(X_test)
            metrics = {
                "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "test_mae": float(mean_absolute_error(y_test, y_pred)),
                "test_r2": float(r2_score(y_test, y_pred)),
            }

            mlflow.log_params(study.best_params)
            mlflow.log_metrics(metrics)

            signature = infer_signature(X_train, best_model.predict(X_train))
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="model",
                signature=signature,
                registered_model_name="dielectron_mass_regressor",
            )

            print(f"Métricas finales: {metrics}")

    download_task = download_and_upload()
    etl_task = etl_data(download_task)
    split_task = split_data(etl_task)
    train_task = train_model()

    split_task >> train_task

dag = process_etl_split()
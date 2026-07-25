import datetime

from airflow.decorators import dag, task

DEFAULT_DIR = "../data"
DEFAULT_FILENAME = "dielectron.csv"
DEFAULT_DATASET = "fedesoriano/cern-electron-collision-data" 
S3_BUCKET = "data"
S3_KEY_PREFIX = "raw"

default_args = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": datetime.timedelta(minutes=1),

}
@dag(schedule_interval="*/60 * * * *",
    start_date=datetime.datetime(2026, 6, 1),
    catchup=False,
    tags=["download", "etl"],
    default_args=default_args,
)
def process_etl_split():

    @task(task_id="download_and_upload")
    def download_and_upload(dataset_slug: str = DEFAULT_DATASET,
                            filename: str = DEFAULT_FILENAME,
                            target_dir: str = DEFAULT_DIR) -> str:

        """
        Descarga el dataset de Kaggle (por slug) y lo sube a S3 (bucket 'data', key 'raw/<filename>').
        Devuelve la URI S3 del archivo subido.
        """
        import os
        import pandas as pd
        import kagglehub
        import awswrangler as wr

        data_path = "s3://data/raw/dielectron.csv"

        os.makedirs(target_dir, exist_ok=True)
        os.environ["KAGGLE_CACHE_DIR"] = target_dir

        #Descargar el dataset desde Kaggle
        try:
            print(f"Descargando dataset {dataset_slug} desde Kaggle...")
            download_path = kagglehub.dataset_download(dataset_slug)
            candidate = os.path.join(download_path, filename)
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


    @task(task_id="etl_data")
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
    

    @task(task_id="split_data")
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

       

    download_task = download_and_upload()
    etl_task = etl_data(download_task)
    split_task = split_data(etl_task)

    download_task

dag = process_etl_split()
FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow==2.14.3 boto3

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////mlflow_data/mlflow.db", \
     "--default-artifact-root", "s3://mlflow/", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
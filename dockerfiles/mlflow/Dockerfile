FROM python:3.12-slim

# Ensure packaging tools are present so `pkg_resources` (setuptools) is available
# Pin setuptools below 81 as required
# Install minimal OS packages required by healthchecks/runtime
RUN apt-get update \
     && apt-get install -y --no-install-recommends wget ca-certificates \
     && rm -rf /var/lib/apt/lists/*

# Install Python packaging tools and mlflow
RUN pip install --no-cache-dir --upgrade pip 'setuptools<81' wheel \
     && pip install --no-cache-dir mlflow==2.14.3 boto3==1.43.68

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////mlflow_data/mlflow.db", \
     "--serve-artifacts", \
     "--artifacts-destination", "s3://mlflow/", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
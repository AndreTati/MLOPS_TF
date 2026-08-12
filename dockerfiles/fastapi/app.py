import json
import os
import pickle
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import mlflow
import numpy as np
import boto3
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

MODEL_NAME = "dielectron_mass_regressor"
MODEL_ALIAS = "thebest"
PREPROCESS_CONFIG_PATH = Path(__file__).resolve().parent / "files" / "data.json"
FALLBACK_MODEL_PATH = Path(__file__).resolve().parent / "files" / "model.pkl"


def _load_pickle_model(path: Path) -> Any:
    try:
        return pickle.loads(path.read_bytes())
    except Exception as exc:
        raise RuntimeError(f"Failed to load pickle model from {path}: {exc}") from exc


def _resolve_file_path(model_uri: str) -> Path | None:
    parsed = urlparse(model_uri)
    if parsed.scheme == "file":
        return Path(parsed.path)
    if parsed.scheme == "":
        return Path(model_uri)
    return None


def load_model(model_name: str = MODEL_NAME, alias: str = MODEL_ALIAS) -> tuple[Any, int, dict]:
    """
    Load model (from MODEL_URI env, fallback pickle, or MLflow registry) and the preprocessing
    data dictionary from S3 (or local fallback file).

    Returns: (model, version, data_dict)
    """
    global loaded_source
    # Prefer loading from MLflow registry first
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.MlflowClient()

        try:
            registered_version = client.get_model_version_by_alias(model_name, alias)
            model_source = registered_version.source
            model_version = int(registered_version.version)
        except Exception:
            versions = client.get_latest_versions(model_name)
            if not versions:
                raise RuntimeError(
                    f"No registered versions found for model '{model_name}' at {tracking_uri}."
                )
            latest_version = max(versions, key=lambda version: int(version.version))
            model_source = latest_version.source
            model_version = int(latest_version.version)

        model = mlflow.sklearn.load_model(model_source)
        loaded_source = model_source
    except Exception as mlflow_exc:
        # If MLflow loading fails, fallback to MODEL_URI env or local file
        model_uri = os.getenv("MODEL_URI")
        if model_uri:
            path = _resolve_file_path(model_uri)
            if path is not None and path.suffix == ".pkl":
                if path.exists():
                    model = _load_pickle_model(path)
                    loaded_source = str(path)
                    return model, 0, load_preprocessing_config()
                # MODEL_URI points at a local pickle that isn't there yet;
                # fall through to FALLBACK_MODEL_PATH below instead of failing outright.
            else:
                model = mlflow.sklearn.load_model(model_uri)
                loaded_source = model_uri
                return model, 0, load_preprocessing_config()

        if FALLBACK_MODEL_PATH.exists():
            model = _load_pickle_model(FALLBACK_MODEL_PATH)
            loaded_source = str(FALLBACK_MODEL_PATH)
            return model, 0, load_preprocessing_config()

        raise RuntimeError(
            f"No se pudo cargar el modelo: la carga desde MLflow falló ({mlflow_exc}) y no "
            f"hay un modelo utilizable en MODEL_URI ni en {FALLBACK_MODEL_PATH}."
        ) from mlflow_exc

    # Load preprocessing/data dictionary from S3 (bucket 'data', key 'data_info/data.json')
    data_dictionary = None
    try:
        s3 = boto3.client("s3")
        resp = s3.get_object(Bucket="data", Key="data_info/data.json")
        text = resp["Body"].read().decode("utf-8")
        data_dictionary = json.loads(text)

        # Convert scaler lists to numpy arrays if present
        if "standard_scaler_mean" in data_dictionary:
            data_dictionary["standard_scaler_mean"] = np.array(data_dictionary["standard_scaler_mean"])
        if "standard_scaler_std" in data_dictionary:
            data_dictionary["standard_scaler_std"] = np.array(data_dictionary["standard_scaler_std"])
    except Exception:
        # Fallback to local preprocessing config
        data_dictionary = load_preprocessing_config()

    return model, model_version, data_dictionary


def check_model() -> None:
    """
    Check the MLflow registry for an updated 'champion' version and reload model/data_dict if changed.
    """
    global model, version_model, loaded_source, data_dict
    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        client = MlflowClient()
        new_version = None
        try:
            mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
            new_version = int(mv.version)
        except Exception:
            versions = client.get_latest_versions(MODEL_NAME)
            if versions:
                new_version = max(versions, key=lambda v: int(v.version)).version

        if new_version is not None and version_model is not None and int(new_version) != int(version_model):
            # reload
            m, v, new_data_dict = load_model(MODEL_NAME, MODEL_ALIAS)
            model = m
            version_model = v
            data_dict = new_data_dict
    except Exception:
        # silently ignore errors during asynchronous checks
        pass


def load_preprocessing_config(path: Path | str = PREPROCESS_CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(path)
    try:
        raw_bytes = config_path.read_bytes()
    except FileNotFoundError as exc:
        raise RuntimeError(f"Preprocessing config not found at {config_path}") from exc

    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        raw_bytes = raw_bytes[3:]

    try:
        config = json.loads(raw_bytes.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            f"Failed to decode preprocessing config {config_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse preprocessing config {config_path}: {exc}"
        ) from exc

    return config


class DielectronInput(BaseModel):
    E1: float = Field(description="Energy of the first electron")
    px1: float = Field(description="x component of the first electron momentum")
    py1: float = Field(description="y component of the first electron momentum")
    pz1: float = Field(description="z component of the first electron momentum")
    pt1: float = Field(description="Transverse momentum of the first electron")
    eta1: float = Field(description="Pseudorapidity of the first electron")
    phi1: float = Field(description="Azimuthal angle of the first electron")
    Q1: float = Field(description="Charge of the first electron")
    E2: float = Field(description="Energy of the second electron")
    px2: float = Field(description="x component of the second electron momentum")
    py2: float = Field(description="y component of the second electron momentum")
    pz2: float = Field(description="z component of the second electron momentum")
    pt2: float = Field(description="Transverse momentum of the second electron")
    eta2: float = Field(description="Pseudorapidity of the second electron")
    phi2: float = Field(description="Azimuthal angle of the second electron")
    Q2: float = Field(description="Charge of the second electron")

    class Config:
        json_schema_extra = {
            "example": {
                "E1": 50.0,
                "px1": 15.0,
                "py1": -8.0,
                "pz1": 40.0,
                "pt1": 17.0,
                "eta1": 1.2,
                "phi1": 0.5,
                "Q1": -1.0,
                "E2": 45.0,
                "px2": -12.0,
                "py2": 5.0,
                "pz2": -35.0,
                "pt2": 13.0,
                "eta2": -0.8,
                "phi2": -0.2,
                "Q2": 1.0,
            }
        }


class PredictionOutput(BaseModel):
    prediction: float = Field(description="Predicted invariant mass M")


model = None
version_model = None
loaded_source = None
data_dict: dict[str, Any] | None = None

# Cargar modelo y configuración al inicio (intentar, pero no fallar la importación)
try:
    model, version_model, data_dict = load_model(MODEL_NAME, MODEL_ALIAS)
except Exception:
    model = None
    version_model = None
    data_dict = None


def get_model() -> Any:
    global model, version_model, data_dict
    if model is None:
        model, version_model, data_dict = load_model()
    return model


def get_preprocessing_config() -> dict[str, Any]:
    global data_dict
    if data_dict is None:
        # try to reload from S3 or local
        try:
            _, _, data_dict = load_model()
        except Exception:
            data_dict = load_preprocessing_config()
    return data_dict


app = FastAPI(title="Dielectron Mass Regression API")


@app.on_event("startup")
def startup_event() -> None:
    try:
        get_model()
        get_preprocessing_config()
    except Exception as exc:
        # Don't fail application startup if MLflow or model isn't ready yet.
        # Log the error so the container stays up and we can retry loading on demand.
        import sys
        print(f"Warning: failed to initialize FastAPI resources: {exc}", file=sys.stderr)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Dielectron mass prediction API is ready."}


@app.get("/model-info")
def model_info() -> dict[str, str]:
    return {
        "model_name": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "version": str(version_model),
        "source": str(loaded_source),
    }


@app.post("/predict/", response_model=PredictionOutput)
def predict(features: DielectronInput, background_tasks: BackgroundTasks) -> PredictionOutput:
    # Ensure model is loaded
    if model is None:
        try:
            get_model()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Schedule an asynchronous check for model updates
    background_tasks.add_task(check_model)

    config = data_dict or load_preprocessing_config()
    columns = config["columns"]
    columns_after_dummy = config.get("columns_after_dummy", columns)
    categorical_columns = config.get("categorical_columns", [])
    categories_map = config.get("categories_values_per_categorical", {})

    input_df = pd.DataFrame([features.dict()])

    if categorical_columns:
        for categorical_col in categorical_columns:
            input_df[categorical_col] = input_df[categorical_col].astype(int)
            categories = categories_map.get(categorical_col)
            if categories is not None:
                input_df[categorical_col] = pd.Categorical(
                    input_df[categorical_col], categories=categories
                )

        input_df = pd.get_dummies(
            data=input_df,
            columns=categorical_columns,
            drop_first=True,
        )

    input_df = input_df.reindex(columns=columns_after_dummy, fill_value=0)

    # Apply scaling if provided
    if config.get("standard_scaler_mean") is not None and config.get("standard_scaler_std") is not None:
        mean = np.array(config["standard_scaler_mean"])
        std = np.array(config["standard_scaler_std"])
        # Ensure alignment with columns_after_dummy
        mean = pd.Series(mean, index=columns_after_dummy)
        std = pd.Series(std, index=columns_after_dummy)
        input_df = (input_df - mean) / std

    try:
        prediction = model.predict(input_df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictionOutput(prediction=float(prediction[0]))

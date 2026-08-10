import json
import os
from typing import Any

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

MODEL_NAME = "dielectron_mass_regressor"
MODEL_ALIAS = "champion"
PREPROCESS_CONFIG_PATH = "/app/files/data.json"


def load_model(model_name: str = MODEL_NAME, alias: str = MODEL_ALIAS) -> tuple[Any, str, str]:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    try:
        registered_version = client.get_model_version_by_alias(model_name, alias)
        model_source = registered_version.source
        model_version = registered_version.version
    except Exception:
        versions = client.get_latest_versions(model_name)
        if not versions:
            raise RuntimeError(
                f"No registered versions found for model '{model_name}' at {tracking_uri}."
            )
        latest_version = max(versions, key=lambda version: int(version.version))
        model_source = latest_version.source
        model_version = latest_version.version

    model = mlflow.sklearn.load_model(model_source)
    return model, model_version, model_source


def load_preprocessing_config(path: str = PREPROCESS_CONFIG_PATH) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as file:
            config = json.load(file)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Preprocessing config not found at {path}") from exc
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
        schema_extra = {
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
loaded_version = None
loaded_source = None
preprocess_config: dict[str, Any] | None = None


def get_model() -> Any:
    global model, loaded_version, loaded_source
    if model is None:
        model, loaded_version, loaded_source = load_model()
    return model


def get_preprocessing_config() -> dict[str, Any]:
    global preprocess_config
    if preprocess_config is None:
        preprocess_config = load_preprocessing_config()
    return preprocess_config


app = FastAPI(title="Dielectron Mass Regression API")


@app.on_event("startup")
def startup_event() -> None:
    try:
        get_model()
        get_preprocessing_config()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize FastAPI: {exc}"
        ) from exc


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Dielectron mass prediction API is ready."}


@app.get("/model-info")
def model_info() -> dict[str, str]:
    return {
        "model_name": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "version": str(loaded_version),
        "source": str(loaded_source),
    }


@app.post("/predict/", response_model=PredictionOutput)
def predict(features: DielectronInput) -> PredictionOutput:
    model = get_model()
    config = get_preprocessing_config()
    columns = config["columns"]
    columns_after_dummy = config.get("columns_after_dummy", columns)
    categorical_columns = config.get("categorical_columns", [])
    categories_map = config.get("categories_values_per_categorical", {})

    input_df = pd.DataFrame([features.dict()])[columns]

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

    if config.get("standard_scaler_mean") and config.get("standard_scaler_std"):
        mean = pd.Series(config["standard_scaler_mean"], index=columns_after_dummy)
        std = pd.Series(config["standard_scaler_std"], index=columns_after_dummy)
        input_df = (input_df - mean) / std

    try:
        prediction = model.predict(input_df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictionOutput(prediction=float(prediction[0]))

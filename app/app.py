"""Entrenamiento reproducible del modelo de masa invariante dielectron.

Este módulo constituye el circuito de entrenamiento de la entrega parcial.
Recibe el dataset desde un directorio local (y lo descarga de Kaggle solo si no
está disponible), entrena un Gradient Boosting Regressor, guarda sus artefactos
y registra la ejecución en MLflow con un backend SQLite local.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import kagglehub
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    """Parámetros que identifican y hacen reproducible una ejecución."""

    dataset_dir: Path = PROJECT_ROOT / "datasets"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    tracking_uri: str = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    experiment_name: str = "dielectron-gradient-boosting"
    random_state: int = 42
    test_size: float = 0.20
    n_trials: int = 10
    cv_folds: int = 5


def configure_logging() -> None:
    """Configura mensajes breves y comparables entre ejecuciones."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)


def download_and_load_dataset(dataset_dir: Path) -> pd.DataFrame:
    """Carga ``dielectron.csv`` o lo descarga desde Kaggle si no existe."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    csv_file = dataset_dir / "dielectron.csv"
    os.environ["KAGGLE_CACHE_DIR"] = str(dataset_dir)

    if not csv_file.exists():
        LOGGER.info("Dataset no encontrado; se descarga desde Kaggle.")
        try:
            downloaded_dir = Path(
                kagglehub.dataset_download("fedesoriano/cern-electron-collision-data")
            )
            shutil.copy2(downloaded_dir / "dielectron.csv", csv_file)
        except Exception as exc:
            raise RuntimeError(
                "No se pudo obtener dielectron.csv. Verificá la conexión y la "
                "configuración local de Kaggle, o colocá el archivo en datasets/."
            ) from exc
    else:
        LOGGER.info("Se reutiliza el dataset local: %s", csv_file)

    dataframe = pd.read_csv(csv_file)
    LOGGER.info("Dataset cargado: %s filas y %s columnas.", *dataframe.shape)
    return dataframe


def preprocess_data(
    dataframe: pd.DataFrame,
    random_state: int,
    test_size: float,
    target_col: str = "M",
    exclude_cols: tuple[str, ...] = ("Run", "Event"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Selecciona variables, elimina nulos y separa train/test reproduciblemente."""
    if target_col not in dataframe.columns:
        raise ValueError(f"No se encontró la columna objetivo {target_col!r}.")

    features = [
        column
        for column in dataframe.columns
        if column != target_col and column not in exclude_cols
    ]
    clean_dataframe = dataframe[features + [target_col]].dropna().copy()
    removed_rows = len(dataframe) - len(clean_dataframe)
    if clean_dataframe.empty:
        raise ValueError("No quedaron observaciones luego de eliminar valores nulos.")

    x = clean_dataframe[features]
    y = clean_dataframe[target_col]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    LOGGER.info(
        "Preprocesamiento: %s variables; %s filas eliminadas; train=%s, test=%s.",
        len(features),
        removed_rows,
        len(x_train),
        len(x_test),
    )
    return x_train, x_test, y_train, y_test, features


def optimize_hyperparameters(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    n_trials: int,
) -> dict[str, Any]:
    """Selecciona hiperparámetros mediante una búsqueda Optuna acotada."""
    x_subset, _, y_subset, _ = train_test_split(
        x_train, y_train, train_size=0.30, random_state=random_state
    )

    def objective(trial: optuna.Trial) -> float:
        parameters = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.20),
            "random_state": random_state,
        }
        model = GradientBoostingRegressor(**parameters)
        scores = cross_val_score(
            model, x_subset, y_subset, cv=2, scoring="r2", n_jobs=-1
        )
        return float(scores.mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_parameters = {**study.best_params, "random_state": random_state}
    LOGGER.info("Mejor R² de búsqueda: %.6f.", study.best_value)
    return best_parameters


def evaluate_model(
    model: GradientBoostingRegressor,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int,
    random_state: int,
) -> dict[str, float]:
    """Evalúa el modelo en test y mediante validación cruzada sobre train."""
    predictions = model.predict(x_test)
    nonzero_target = y_test.to_numpy() != 0
    mape = float(
        np.mean(
            np.abs(
                (y_test.to_numpy()[nonzero_target] - predictions[nonzero_target])
                / y_test.to_numpy()[nonzero_target]
            )
        )
        * 100
    ) if nonzero_target.any() else float("nan")

    cv_model = GradientBoostingRegressor(**model.get_params())
    folds = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_rmse = np.sqrt(
        -cross_val_score(
            cv_model,
            x_train,
            y_train,
            cv=folds,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
    )
    cv_r2 = cross_val_score(cv_model, x_train, y_train, cv=folds, scoring="r2", n_jobs=-1)

    metrics = {
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "test_mae": float(mean_absolute_error(y_test, predictions)),
        "test_r2": float(r2_score(y_test, predictions)),
        "test_mape": mape,
        "cv_rmse_mean": float(cv_rmse.mean()),
        "cv_rmse_std": float(cv_rmse.std()),
        "cv_r2_mean": float(cv_r2.mean()),
        "cv_r2_std": float(cv_r2.std()),
    }
    LOGGER.info(
        "Métricas test: RMSE=%.4f | MAE=%.4f | R²=%.6f.",
        metrics["test_rmse"],
        metrics["test_mae"],
        metrics["test_r2"],
    )
    return metrics


def save_artifacts(
    model: GradientBoostingRegressor,
    features: list[str],
    metrics: dict[str, float],
    best_parameters: dict[str, Any],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    artifacts_dir: Path,
) -> list[Path]:
    """Guarda modelo, metadatos y relevancia de variables en rutas portables."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "gradient_boosting_model.joblib"
    metadata_path = artifacts_dir / "model_metadata.json"
    importance_path = artifacts_dir / "feature_importances.csv"

    joblib.dump(model, model_path)
    metadata = {
        "target": "M",
        "features": features,
        "best_parameters": best_parameters,
        "metrics": metrics,
        "data_shapes": {"x_train": list(x_train.shape), "x_test": list(x_test.shape)},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (
        pd.DataFrame({"feature": features, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .to_csv(importance_path, index=False)
    )
    LOGGER.info("Artefactos guardados en: %s", artifacts_dir)
    return [model_path, metadata_path, importance_path]


def run_training(config: TrainingConfig) -> None:
    """Ejecuta y registra el flujo de entrenamiento completo."""
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "random_state": config.random_state,
                "test_size": config.test_size,
                "n_trials": config.n_trials,
                "cv_folds": config.cv_folds,
                "model": "GradientBoostingRegressor",
            }
        )
        dataframe = download_and_load_dataset(config.dataset_dir)
        x_train, x_test, y_train, y_test, features = preprocess_data(
            dataframe, config.random_state, config.test_size
        )
        best_parameters = optimize_hyperparameters(
            x_train, y_train, config.random_state, config.n_trials
        )
        model = GradientBoostingRegressor(**best_parameters)
        model.fit(x_train, y_train)
        metrics = evaluate_model(
            model,
            x_test,
            y_test,
            x_train,
            y_train,
            config.cv_folds,
            config.random_state,
        )
        artifact_paths = save_artifacts(
            model,
            features,
            metrics,
            best_parameters,
            x_train,
            x_test,
            config.artifacts_dir,
        )
        mlflow.log_params({f"model_{key}": value for key, value in best_parameters.items()})
        mlflow.log_metrics({key: value for key, value in metrics.items() if np.isfinite(value)})
        mlflow.log_artifacts(str(config.artifacts_dir))
        LOGGER.info("Ejecución registrada en MLflow; artefactos: %s", len(artifact_paths))


def parse_args() -> TrainingConfig:
    """Permite modificar los parámetros de prueba sin editar el código."""
    parser = argparse.ArgumentParser(description="Entrena el modelo dielectron.")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "datasets")
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"),
    )
    args = parser.parse_args()
    return TrainingConfig(
        dataset_dir=args.dataset_dir,
        artifacts_dir=args.artifacts_dir,
        tracking_uri=args.tracking_uri,
        random_state=args.random_state,
        test_size=args.test_size,
        n_trials=args.n_trials,
    )


if __name__ == "__main__":
    configure_logging()
    run_training(parse_args())

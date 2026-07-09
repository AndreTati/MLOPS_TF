import kagglehub
import os
import shutil
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import optuna


def print_section(title):
    """Imprime un encabezado de sección formateado."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def download_and_load_dataset(dataset_dir, csv_file):
    """
    Descarga el dataset desde Kaggle si no existe, o lo carga si ya está disponible.
    
    Args:
        dataset_dir: Directorio donde se almacena el dataset
        csv_file: Ruta completa al archivo CSV
    
    Returns:
        DataFrame con los datos cargados
    """
    print_section("DESCARGA Y VALIDACIÓN DEL DATASET")
    
    os.makedirs(dataset_dir, exist_ok=True)
    os.environ["KAGGLE_CACHE_DIR"] = dataset_dir
    
    # Verificar si el archivo ya existe
    if os.path.exists(csv_file):
        print(f"✓ Dataset encontrado en: {csv_file}")
        print("  (usando dataset existente, no descargando de nuevo)")
    else:
        print("Dataset no encontrado. Descargando desde Kaggle...")
        try:
            path = kagglehub.dataset_download("fedesoriano/cern-electron-collision-data")
            data_file_path = os.path.join(path, 'dielectron.csv')
            shutil.copy(data_file_path, csv_file)
            print(f"✓ Dataset descargado y guardado en: {csv_file}")
        except Exception as e:
            print(f"✗ Error descargando dataset: {e}")
            raise
    
    df = pd.read_csv(csv_file)
    
    print(f"\n✓ Dimensiones del dataset: {df.shape}")
    print(f"\nColumnas disponibles:")
    for col in df.columns:
        print(f"  - {col}")
    
    return df


def preprocess_data(df, target_col='M', exclude_cols=None):
    """
    Preprocesa el dataset: selecciona features, limpia NaN y realiza split train/test.
    
    Args:
        df: DataFrame con los datos
        target_col: Nombre de la columna objetivo
        exclude_cols: Lista de columnas a excluir
    
    Returns:
        Tupla: (X_train, X_test, y_train, y_test, scaler, X, y, features, target)
    """
    print_section("PREPROCESAMIENTO")
    
    if exclude_cols is None:
        exclude_cols = ['Run', 'Event']
    
    # Selección de features
    features = [col for col in df.columns 
                if col != target_col and col not in exclude_cols]
    
    print(f"\nFeatures ({len(features)}): {features}")
    print(f"Target: {target_col}")
    
    # Limpieza de NaN
    df_clean = df[features + [target_col]].dropna()
    filas_eliminadas = len(df) - len(df_clean)
    
    print(f"\nFilas originales : {len(df)}")
    print(f"Filas con NaN    : {filas_eliminadas}")
    print(f"Filas utilizables: {len(df_clean)}")
    
    X = df_clean[features].copy()
    y = df_clean[target_col].copy()
    
    # Verificación final
    assert X.isnull().sum().sum() == 0, "Aún hay NaN en X"
    assert y.isnull().sum() == 0, "Aún hay NaN en y"
    print("\n✓ Verificación OK: sin valores NaN en X ni en y.")
    
    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTamaño entrenamiento: {X_train.shape[0]} muestras")
    print(f"Tamaño prueba:        {X_test.shape[0]} muestras")
    
    # Escalado
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    print(f"\n✓ Datos escalados y listos para entrenamiento")
    
    return X_train, X_test, y_train, y_test, scaler, X, y, features, target_col


def optimize_hyperparameters(X_train, y_train, n_trials=10):
    """
    Busca hiperparámetros óptimos para Gradient Boosting usando Optuna.
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        n_trials: Número de trials para la búsqueda
    
    Returns:
        Diccionario con los mejores parámetros
    """
    print_section("BÚSQUEDA DE HIPERPARÁMETROS (OPTUNA)")
    
    # Subconjunto para búsqueda rápida
    X_train_sub, _, y_train_sub, _ = train_test_split(
        X_train, y_train, train_size=0.3, random_state=42
    )
    
    # Función objetivo
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
            'random_state': 42
        }
        model = GradientBoostingRegressor(**params)
        score = cross_val_score(
            model, X_train_sub, y_train_sub, cv=2, scoring='r2', n_jobs=-1
        )
        return score.mean()
    
    # Ejecutar optimización
    print(f"\nIniciando búsqueda de hiperparámetros ({n_trials} trials)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = study.best_params
    print(f"\n✓ Mejor R² en búsqueda: {study.best_value:.6f}")
    print(f"✓ Mejores parámetros: {best_params}")
    
    return best_params


def train_model(X_train, y_train, best_params):
    """
    Entrena el modelo Gradient Boosting con parámetros optimizados.
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        best_params: Diccionario con los mejores parámetros
    
    Returns:
        Modelo entrenado
    """
    print_section("ENTRENAMIENTO DEL MODELO FINAL")
    
    print("\nEntrenando Gradient Boosting con set completo de entrenamiento...")
    gb_model = GradientBoostingRegressor(**best_params, random_state=42)
    gb_model.fit(X_train, y_train)
    print("✓ Modelo entrenado exitosamente")
    
    return gb_model


def evaluate_model(gb_model, X_test, y_test, X, y):
    """
    Calcula las métricas de evaluación del modelo.
    
    Args:
        gb_model: Modelo entrenado
        X_test: Features de test
        y_test: Target de test
        X: Features completo (para validación cruzada)
        y: Target completo (para validación cruzada)
    
    Returns:
        Diccionario con todas las métricas
    """
    print_section("MÉTRICAS FINALES — GRADIENT BOOSTING")
    
    # Predicciones en conjunto de test
    y_pred = gb_model.predict(X_test)
    
    # Cálculo de métricas
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # MAPE
    residuos = y_test.values - y_pred
    mape = np.mean(np.abs(residuos / y_test.values)) * 100
    
    print(f"\n  RMSE (Root Mean Squared Error) : {rmse:.4f} GeV/c²")
    print(f"  MAE  (Mean Absolute Error)     : {mae:.4f} GeV/c²")
    print(f"  R²   (Coefficient of Determination): {r2:.6f}")
    print(f"  MAPE (Mean Absolute Percentage Error): {mape:.2f} %")
    
    # Validación cruzada 5-fold
    print("\nValidación cruzada (5-fold)...")
    gb_cv = GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                       learning_rate=0.1, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rmse = np.sqrt(-cross_val_score(gb_cv, X, y, cv=kf,
                                        scoring='neg_mean_squared_error', n_jobs=-1))
    cv_r2 = cross_val_score(gb_cv, X, y, cv=kf, scoring='r2', n_jobs=-1)
    
    print(f"  RMSE CV promedio: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f} GeV/c²")
    print(f"  R² CV promedio  : {cv_r2.mean():.6f} ± {cv_r2.std():.6f}")
    
    metrics = {
        'y_pred': y_pred,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'cv_rmse_mean': cv_rmse.mean(),
        'cv_rmse_std': cv_rmse.std(),
        'cv_r2_mean': cv_r2.mean(),
        'cv_r2_std': cv_r2.std()
    }
    
    return metrics


def display_feature_importance(gb_model, features):
    """
    Muestra la importancia de las features del modelo.
    
    Args:
        gb_model: Modelo entrenado
        features: Lista de nombres de features
    """
    print_section("IMPORTANCIA DE FEATURES")
    
    importancias = pd.Series(gb_model.feature_importances_, index=features)
    importancias_sorted = importancias.sort_values(ascending=False)
    
    print("\nLas 10 variables más importantes:")
    for i, (feature, importance) in enumerate(importancias_sorted.head(10).items(), 1):
        print(f"  {i:2d}. {feature:8s} : {importance:.6f}")
    
    return importancias_sorted


def save_artifacts(gb_model, scaler, best_params, metrics, features, target, 
                   X_train, X_test, artifacts_dir="..\\artifacts"):
    """
    Guarda el modelo, scaler y metadatos en archivos.
    
    Args:
        gb_model: Modelo entrenado
        scaler: Scaler utilizado
        best_params: Parámetros optimizados
        metrics: Diccionario con métricas
        features: Lista de features
        target: Nombre de la variable objetivo
        X_train: Features de entrenamiento
        X_test: Features de test
        artifacts_dir: Directorio donde guardar los artefactos
    """
    print_section("GUARDANDO ARTEFACTOS")
    
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Guardar modelo
    model_path = os.path.join(artifacts_dir, "gb_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(gb_model, f)
    print(f"✓ Modelo guardado en: {model_path}")
    
    # Guardar scaler
    scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler guardado en: {scaler_path}")
    
    # Guardar metadatos
    metadata = {
        'features': features,
        'target': target,
        'best_params': best_params,
        'metrics': {
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'r2': metrics['r2'],
            'mape': metrics['mape']
        },
        'cv_metrics': {
            'cv_rmse_mean': metrics['cv_rmse_mean'],
            'cv_rmse_std': metrics['cv_rmse_std'],
            'cv_r2_mean': metrics['cv_r2_mean'],
            'cv_r2_std': metrics['cv_r2_std']
        },
        'data_shapes': {
            'X_train': X_train.shape,
            'X_test': X_test.shape
        }
    }
    
    metadata_path = os.path.join(artifacts_dir, "model_metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✓ Metadatos guardados en: {metadata_path}")



def main():
    """Ejecuta el pipeline completo de entrenamiento del modelo."""
    
    # Configuración
    DATASET_DIR = "..\\datasets"
    CSV_FILE = os.path.join(DATASET_DIR, "dielectron.csv")
    #ARTIFACTS_DIR = "..\\artifacts"
    
    # 1. Descarga y carga del dataset
    df = download_and_load_dataset(DATASET_DIR, CSV_FILE)
    
    # 2. Preprocesamiento
    X_train, X_test, y_train, y_test, scaler, X, y, features, target = preprocess_data(df)
    
    # 3. Búsqueda de hiperparámetros
    best_params = optimize_hyperparameters(X_train, y_train, n_trials=10)
    
    # 4. Entrenamiento del modelo
    gb_model = train_model(X_train, y_train, best_params)
    
    # 5. Evaluación
    metrics = evaluate_model(gb_model, X_test, y_test, X, y)
    
    # 6. Feature importance
    #display_feature_importance(gb_model, features)
    
    # 7. Guardar artefactos
    #save_artifacts(gb_model, scaler, best_params, metrics, features, target,
    #              X_train, X_test, ARTIFACTS_DIR)
    
    print_section("PIPELINE COMPLETADO EXITOSAMENTE")


if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/xgboost_riesgo.joblib")
ENCODER_PATH = Path("models/label_encoder.joblib")


def extraer_features(historial: pd.DataFrame, stock_actual: int, stock_minimo: int) -> dict:
    """
    Extrae las features que el modelo XGBoost necesita para clasificar el riesgo.
    Todas se calculan a partir del historial real de la BD.
    """
    if historial.empty:
        return _features_vacias(stock_actual, stock_minimo)

    salidas = historial["salidas"]
    entradas = historial["entradas"]

    # Consumos en diferentes ventanas de tiempo
    consumo_7d  = salidas.tail(7).sum()
    consumo_30d = salidas.tail(30).sum()
    consumo_90d = salidas.tail(90).sum()

    # Promedios diarios
    promedio_7d  = salidas.tail(7).mean()
    promedio_30d = salidas.tail(30).mean()

    # Tendencia: comparar consumo reciente vs anterior
    consumo_reciente  = salidas.tail(15).mean()
    consumo_anterior  = salidas.iloc[-30:-15].mean() if len(salidas) >= 30 else promedio_30d
    tendencia = float(consumo_reciente - consumo_anterior) if consumo_anterior > 0 else 0.0

    # Entradas recientes (si hay reposicion reciente, el riesgo baja)
    entradas_30d = entradas.tail(30).sum()

    # Dias de stock restante (con consumo promedio de 30 dias)
    dias_stock = (stock_actual / promedio_30d) if promedio_30d > 0 else 999

    # Ratio stock actual vs stock minimo
    ratio_stock_minimo = stock_actual / stock_minimo if stock_minimo > 0 else 10.0

    # Variabilidad del consumo (coeficiente de variacion)
    cv_consumo = (salidas.tail(30).std() / promedio_30d) if promedio_30d > 0 else 0.0

    return {
        "stock_actual":      stock_actual,
        "stock_minimo":      stock_minimo,
        "ratio_stock":       round(float(ratio_stock_minimo), 4),
        "consumo_7d":        int(consumo_7d),
        "consumo_30d":       int(consumo_30d),
        "consumo_90d":       int(consumo_90d),
        "promedio_diario_7d":  round(float(promedio_7d), 4),
        "promedio_diario_30d": round(float(promedio_30d), 4),
        "tendencia_consumo":   round(float(tendencia), 4),
        "entradas_30d":        int(entradas_30d),
        "dias_stock_restante": round(float(min(dias_stock, 999)), 2),
        "cv_consumo":          round(float(cv_consumo), 4),
    }


def _features_vacias(stock_actual: int, stock_minimo: int) -> dict:
    ratio = stock_actual / stock_minimo if stock_minimo > 0 else 10.0
    return {
        "stock_actual": stock_actual, "stock_minimo": stock_minimo,
        "ratio_stock": float(ratio), "consumo_7d": 0, "consumo_30d": 0,
        "consumo_90d": 0, "promedio_diario_7d": 0.0, "promedio_diario_30d": 0.0,
        "tendencia_consumo": 0.0, "entradas_30d": 0,
        "dias_stock_restante": 999.0, "cv_consumo": 0.0,
    }


def clasificar_riesgo_reglas(features: dict) -> str:
    """
    Clasificacion basada en reglas cuando no hay modelo entrenado aun.
    Siempre disponible — no requiere datos de entrenamiento.
    """
    ratio      = features["ratio_stock"]
    dias       = features["dias_stock_restante"]
    tendencia  = features["tendencia_consumo"]

    if features["stock_actual"] == 0:
        return "CRITICO"
    if ratio < 0.5 or dias < 7:
        return "ALTO"
    if ratio < 1.0 or dias < 15:
        if tendencia > 0:   # consumo creciendo
            return "ALTO"
        return "MEDIO"
    if ratio < 1.5 or dias < 30:
        return "MEDIO"
    return "BAJO"


def entrenar_modelo(datos_entrenamiento: list[dict]) -> bool:
    """
    Entrena el modelo XGBoost con datos historicos reales.

    datos_entrenamiento: lista de dicts con features + etiqueta 'riesgo_real'
    La etiqueta se genera automaticamente segun lo que ocurrio despues:
      - CRITICO: el insumo llego a 0
      - ALTO:    llego a stock_minimo en los siguientes 7 dias
      - MEDIO:   llego a stock_minimo en los siguientes 30 dias
      - BAJO:    no tuvo problema de stock
    """
    if len(datos_entrenamiento) < 20:
        logger.warning(f"Datos insuficientes para entrenar: {len(datos_entrenamiento)} registros")
        return False

    df = pd.DataFrame(datos_entrenamiento)

    feature_cols = [
        "stock_actual", "stock_minimo", "ratio_stock",
        "consumo_7d", "consumo_30d", "consumo_90d",
        "promedio_diario_7d", "promedio_diario_30d",
        "tendencia_consumo", "entradas_30d",
        "dias_stock_restante", "cv_consumo"
    ]

    X = df[feature_cols].fillna(0)
    y = df["riesgo_real"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    modelo = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42
    )

    modelo.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Reporte en log
    y_pred = modelo.predict(X_test)
    reporte = classification_report(y_test, y_pred, target_names=le.classes_)
    logger.info(f"Modelo entrenado:\n{reporte}")

    # Guardar modelo
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(modelo, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)

    logger.info(f"Modelo guardado en {MODEL_PATH}")
    return True


def predecir_riesgo_xgboost(features: dict) -> dict:
    """
    Clasifica el nivel de riesgo usando XGBoost si hay modelo entrenado,
    o reglas simples si aun no hay suficientes datos.
    """
    if not MODEL_PATH.exists():
        nivel = clasificar_riesgo_reglas(features)
        return {
            "nivel_riesgo": nivel,
            "probabilidades": None,
            "modelo_entrenado": False,
            "metodo": "reglas"
        }

    try:
        modelo = joblib.load(MODEL_PATH)
        le     = joblib.load(ENCODER_PATH)

        feature_cols = [
            "stock_actual", "stock_minimo", "ratio_stock",
            "consumo_7d", "consumo_30d", "consumo_90d",
            "promedio_diario_7d", "promedio_diario_30d",
            "tendencia_consumo", "entradas_30d",
            "dias_stock_restante", "cv_consumo"
        ]

        X = pd.DataFrame([features])[feature_cols].fillna(0)
        pred_idx  = modelo.predict(X)[0]
        proba     = modelo.predict_proba(X)[0]
        nivel     = le.inverse_transform([pred_idx])[0]

        probabilidades = {
            le.classes_[i]: round(float(p), 4)
            for i, p in enumerate(proba)
        }

        return {
            "nivel_riesgo":     nivel,
            "probabilidades":   probabilidades,
            "modelo_entrenado": True,
            "metodo":           "xgboost"
        }

    except Exception as e:
        logger.error(f"Error en XGBoost predict: {e}")
        nivel = clasificar_riesgo_reglas(features)
        return {
            "nivel_riesgo": nivel,
            "probabilidades": None,
            "modelo_entrenado": False,
            "metodo": "reglas_fallback"
        }
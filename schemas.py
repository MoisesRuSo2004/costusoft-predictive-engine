from pydantic import BaseModel
from typing import Optional


class PrediccionRequest(BaseModel):
    insumo_id: int


class ProhetResultado(BaseModel):
    dias_hasta_stock_minimo:    Optional[int]
    dias_hasta_cero:            Optional[int]
    consumo_diario_promedio:    float
    fecha_alerta_estimada:      Optional[str]
    fecha_agotamiento_estimada: Optional[str]
    confianza:                  float
    suficiente_historial:       bool
    metodo:                     str


class XGBoostResultado(BaseModel):
    nivel_riesgo:     str
    probabilidades:   Optional[dict]
    modelo_entrenado: bool
    metodo:           str


class PrediccionResponse(BaseModel):
    insumo_id:     int
    nombre:        str
    stock_actual:  int
    stock_minimo:  int
    unidad_medida: str

    # Resultado Prophet
    prophet: ProhetResultado

    # Resultado XGBoost
    xgboost: XGBoostResultado

    # Features usadas (para trazabilidad)
    features: dict

    # Resumen ejecutivo para el frontend
    alerta:         bool
    mensaje:        str
    recomendacion:  str


class PrediccionMasivaResponse(BaseModel):
    total:         int
    en_riesgo:     int
    predicciones:  list[PrediccionResponse]


class EntrenamientoResponse(BaseModel):
    exito:    bool
    mensaje:  str
    registros: int
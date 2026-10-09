"""
Contratos de entrada y salida del modulo de mineria de datos.
"""
from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ── Deteccion de anomalias ───────────────────────────────────────────────────

class AnomaliasRequest(BaseModel):
    # Periodo a revisar; por defecto, los 12 meses previos al ultimo movimiento.
    desde: Optional[date] = None
    hasta: Optional[date] = None
    insumo_ids: Optional[list[int]] = Field(default=None, max_length=50)
    tipo: Literal["salidas", "entradas", "ambos"] = "salidas"
    sensibilidad: Literal["baja", "media", "alta"] = "media"
    limite: int = Field(default=100, ge=1, le=500)
    # Validacion con anomalias simuladas en memoria (no toca la base de datos).
    validar: bool = True
    anomalias_simuladas: int = Field(default=60, ge=10, le=300)


class AnomaliaMovimiento(BaseModel):
    movimiento: Literal["SALIDA", "ENTRADA"]
    movimiento_id: int
    # Un movimiento puede tener varios renglones del mismo insumo: este id es unico.
    renglon_id: int
    fecha: str
    insumo_id: int
    insumo: str
    unidad_medida: str
    cantidad: float
    mediana_habitual: float
    rango_habitual: list[float]      # percentiles 10 y 90 del insumo
    veces_mediana: float
    dias_desde_anterior: Optional[int]
    tercero: Optional[str]           # colegio (salidas) o proveedor (entradas)
    descripcion: Optional[str]
    nivel: Literal["alta", "media"]  # alta = 2 o mas metodos coinciden
    tipo_anomalia: Literal["cantidad_alta", "cantidad_baja", "repetida", "combinacion"]
    razon: str
    metodos: list[str]
    puntaje: float                   # 0 a 1, para ordenar


class ResumenAnomalias(BaseModel):
    movimientos_analizados: int
    alta: int
    media: int
    por_tipo: dict[str, int]


class MetricaMetodo(BaseModel):
    metodo: str
    nombre: str
    precision: float     # % de las alertas que eran anomalias inyectadas
    recall: float        # % de las anomalias inyectadas que se detectaron
    f1: float
    falsas_alarmas: int
    # Recall por tipo de anomalia inyectada: alta, baja y rafaga.
    recall_por_tipo: dict[str, float]


class ValidacionDetector(BaseModel):
    anomalias_inyectadas: int
    movimientos_evaluados: int
    metricas: list[MetricaMetodo]


class AnomaliasResponse(BaseModel):
    desde: str
    hasta: str
    sensibilidad: str
    resumen: ResumenAnomalias
    anomalias: list[AnomaliaMovimiento]
    validacion: Optional[ValidacionDetector]

"""
Contratos de entrada y salida del modulo de mineria de datos.
"""
from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ── Pronostico dinamico ──────────────────────────────────────────────────────

class PrendaEscenario(BaseModel):
    uniforme_id: int
    talla: Optional[str] = None
    cantidad: int = Field(gt=0, le=100_000)


class Escenario(BaseModel):
    """'¿Y si?': un pedido hipotetico que consumiria insumos en una fecha."""
    prendas: list[PrendaEscenario] = Field(min_length=1, max_length=20)
    # Si no se indica, el consumo se aplica en la fecha de corte.
    fecha: Optional[date] = None


class PronosticoRequest(BaseModel):
    insumo_ids: list[int] = Field(min_length=1, max_length=5)
    horizonte_dias: int = Field(default=90, ge=7, le=365)
    agrupacion: Literal["dia", "semana", "mes"] = "semana"
    # Dias que tarda el proveedor en entregar: la compra debe hacerse antes.
    plazo_proveedor_dias: int = Field(default=7, ge=0, le=90)
    escenario: Optional[Escenario] = None


class PuntoSerie(BaseModel):
    fecha: str
    valor: float
    minimo: Optional[float] = None
    maximo: Optional[float] = None


class ComparacionModelo(BaseModel):
    modelo: str
    descripcion: str
    mae: float
    wape: Optional[float]
    error_total: Optional[float]


class Precision(BaseModel):
    """
    Backtesting: cada modelo candidato se entrena sin los ultimos meses, los
    pronostica y se compara con lo real. Gana el de menor error.
    """
    meses_evaluados: int
    modelo_elegido: str
    mae: float            # error absoluto medio por mes (en la unidad del insumo)
    wape: Optional[float] # error total / consumo total, en % (None si no hubo consumo)
    error_total: Optional[float]  # error sobre la suma del periodo, en %
    interpretacion: str
    comparacion: list[ComparacionModelo]


class CompraSugerida(BaseModel):
    cantidad: float
    fecha_limite: Optional[str]
    motivo: str


class PronosticoInsumo(BaseModel):
    insumo_id: int
    nombre: str
    unidad_medida: str
    stock_actual: float
    stock_minimo: float

    historial: list[PuntoSerie]
    pronostico: list[PuntoSerie]
    stock_proyectado: list[PuntoSerie]
    stock_proyectado_escenario: Optional[list[PuntoSerie]] = None

    consumo_pronosticado: float
    consumo_escenario: float
    fecha_alerta: Optional[str]
    fecha_agotamiento: Optional[str]
    fecha_alerta_escenario: Optional[str] = None
    fecha_agotamiento_escenario: Optional[str] = None

    compra_sugerida: CompraSugerida
    precision: Optional[Precision]
    metodo: str


class PronosticoResponse(BaseModel):
    fecha_corte: str
    datos_desactualizados: bool
    horizonte_dias: int
    agrupacion: str
    insumos: list[PronosticoInsumo]
    no_encontrados: list[int]


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

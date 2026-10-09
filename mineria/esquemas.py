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

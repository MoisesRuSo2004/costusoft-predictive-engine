"""
Endpoints del modulo de mineria de datos (/mineria/*).
"""
import logging
import time

from fastapi import APIRouter, Depends, HTTPException

from mineria.anomalias import detectar_anomalias
from mineria.esquemas import AnomaliasRequest, AnomaliasResponse, PronosticoRequest, PronosticoResponse
from mineria.pronostico import calcular_pronostico
from seguridad import verificar_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mineria", tags=["Mineria de datos"], dependencies=[Depends(verificar_token)])


@router.post("/pronostico", response_model=PronosticoResponse)
def pronostico(solicitud: PronosticoRequest):
    """
    Prediccion dinamica: el usuario elige insumos (1 a 5), horizonte,
    agrupacion, plazo del proveedor y opcionalmente un escenario "¿y si?".
    Cada insumo usa el modelo que mejor resulto en el backtesting.
    """
    inicio = time.monotonic()
    respuesta = calcular_pronostico(solicitud)
    if not respuesta.insumos:
        raise HTTPException(status_code=404, detail="Ninguno de los insumos indicados existe")
    logger.info(
        f"Pronostico dinamico — insumos: {solicitud.insumo_ids} | horizonte: {solicitud.horizonte_dias} d | "
        f"escenario: {'si' if solicitud.escenario else 'no'} | {time.monotonic() - inicio:.1f} s"
    )
    return respuesta


@router.post("/anomalias", response_model=AnomaliasResponse)
def anomalias(solicitud: AnomaliasRequest):
    """
    Movimientos que se alejan de lo habitual para su insumo (cantidades muy
    altas o muy bajas, repeticiones, combinaciones raras), con estadistica
    robusta, Isolation Forest y LOF. Incluye la validacion del detector con
    anomalias simuladas en memoria.
    """
    inicio = time.monotonic()
    respuesta = detectar_anomalias(solicitud)
    logger.info(
        f"Anomalias — {solicitud.tipo} {respuesta.desde}..{respuesta.hasta} | sensibilidad: {solicitud.sensibilidad} | "
        f"alta: {respuesta.resumen.alta} · media: {respuesta.resumen.media} | {time.monotonic() - inicio:.1f} s"
    )
    return respuesta

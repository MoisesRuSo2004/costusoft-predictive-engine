"""
Endpoints del modulo de mineria de datos (/mineria/*).
"""
import logging
import time

from fastapi import APIRouter, Depends, HTTPException

from mineria.esquemas import PronosticoRequest, PronosticoResponse
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

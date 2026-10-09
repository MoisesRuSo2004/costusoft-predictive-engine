"""
Endpoints del modulo de mineria de datos (/mineria/*).
"""
import logging
import time

from fastapi import APIRouter, Depends

from mineria.anomalias import detectar_anomalias
from mineria.esquemas import AnomaliasRequest, AnomaliasResponse
from seguridad import verificar_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mineria", tags=["Mineria de datos"], dependencies=[Depends(verificar_token)])


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

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
import logging
import threading
import time
import pandas as pd
from datetime import date, timedelta

from config import settings
from database import (
    obtener_historial_insumo,
    obtener_stock_actual,
    obtener_todos_los_insumos,
    engine
)
from prophet_model import predecir_agotamiento_prophet
from xgboost_model import (
    extraer_features,
    predecir_riesgo_xgboost,
    entrenar_modelo
)
from schemas import (
    PrediccionResponse,
    PrediccionMasivaResponse,
    EntrenamientoResponse,
    ProhetResultado,
    XGBoostResultado
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO if settings.ENVIRONMENT == "production" else logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Prediccion Service",
    description="Microservicio de prediccion de agotamiento de insumos — Prophet + XGBoost",
    version="1.0.0",
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Seguridad: token interno ─────────────────────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-Token", auto_error=False)

def verificar_token(token: str = Security(api_key_header)):
    if token != settings.API_SECRET_TOKEN:
        raise HTTPException(status_code=403, detail="Token invalido")
    return token

# ── Scheduler para reentrenamiento automatico ─────────────────────────────────
scheduler = BackgroundScheduler()

@app.on_event("startup")
def startup_event():
    logger.info("Iniciando prediccion-service...")
    # Reentrenar modelo cada dia a las 2 AM
    scheduler.add_job(
        reentrenar_modelo_automatico,
        "cron",
        hour=2,
        minute=0,
        id="reentrenamiento_diario"
    )
    # Mantener fresca la cache de la prediccion masiva
    scheduler.add_job(
        _recalcular_cache,
        "interval",
        seconds=settings.CACHE_PREDICCIONES_SEGUNDOS,
        id="refresco_cache_predicciones"
    )
    scheduler.start()
    # Precalcular al arrancar para que la primera consulta no espere
    _recalcular_cache_en_segundo_plano()
    logger.info("Scheduler iniciado — reentrenamiento diario a las 2 AM")

@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown()

# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "prediccion-service", "version": "1.0.0"}

# ── Cache de la prediccion masiva ─────────────────────────────────────────────
# Calcular todos los insumos (Prophet + XGBoost por cada uno) tarda ~20 s.
# Se guarda el ultimo resultado y se aplica "stale-while-revalidate": se
# responde al instante con el calculo guardado y, si esta vencido, se
# recalcula en segundo plano. Se precalcula al arrancar, cada 10 minutos y
# despues de reentrenar el modelo.
_cache_lock = threading.Lock()
_cache_predicciones: dict = {"resultado": None, "calculado_en": 0.0}
_recalculo_en_curso = threading.Event()


def _guardar_en_cache(resultado: "PrediccionMasivaResponse") -> None:
    with _cache_lock:
        _cache_predicciones["resultado"] = resultado
        _cache_predicciones["calculado_en"] = time.monotonic()


def _recalcular_cache() -> None:
    """Recalcula la prediccion masiva y la guarda (una sola vez a la vez)."""
    if _recalculo_en_curso.is_set():
        return
    _recalculo_en_curso.set()
    try:
        inicio = time.monotonic()
        _guardar_en_cache(_calcular_predicciones_todos())
        logger.info(f"Cache de predicciones actualizada en {time.monotonic() - inicio:.1f} s")
    except Exception as e:
        logger.error(f"No se pudo actualizar la cache de predicciones: {e}")
    finally:
        _recalculo_en_curso.clear()


def _recalcular_cache_en_segundo_plano() -> None:
    threading.Thread(target=_recalcular_cache, name="recalculo-predicciones", daemon=True).start()


# ── Prediccion masiva ─────────────────────────────────────────────────────────
@app.get("/predict/todos", response_model=PrediccionMasivaResponse)
def predecir_todos(refrescar: bool = False, token: str = Depends(verificar_token)):
    """
    Predice agotamiento para todos los insumos (ordenados por riesgo).
    Responde desde la cache; con `?refrescar=true` fuerza un calculo nuevo.
    """
    with _cache_lock:
        resultado = _cache_predicciones["resultado"]
        edad = time.monotonic() - _cache_predicciones["calculado_en"]

    if resultado is None or refrescar:
        resultado = _calcular_predicciones_todos()
        _guardar_en_cache(resultado)
        return resultado

    if edad > settings.CACHE_PREDICCIONES_SEGUNDOS:
        _recalcular_cache_en_segundo_plano()
    return resultado


def _calcular_predicciones_todos() -> PrediccionMasivaResponse:
    """Calcula Prophet + XGBoost para cada insumo (operacion costosa)."""
    insumos = obtener_todos_los_insumos()
    predicciones = []
    en_riesgo = 0

    for insumo in insumos:
        try:
            historial = obtener_historial_insumo(insumo["id"], dias=180)
            features  = extraer_features(historial, insumo["stock"], insumo["stock_minimo"])
            r_prophet = predecir_agotamiento_prophet(
                historial, insumo["stock"], insumo["stock_minimo"])
            r_xgboost = predecir_riesgo_xgboost(features)

            alerta, mensaje, recomendacion = _construir_resumen(insumo, r_prophet, r_xgboost)

            pred = PrediccionResponse(
                insumo_id=insumo["id"],
                nombre=insumo["nombre"],
                stock_actual=insumo["stock"],
                stock_minimo=insumo["stock_minimo"],
                unidad_medida=insumo["unidad_medida"],
                prophet=ProhetResultado(**r_prophet),
                xgboost=XGBoostResultado(**r_xgboost),
                features=features,
                alerta=alerta,
                mensaje=mensaje,
                recomendacion=recomendacion
            )
            predicciones.append(pred)

            if r_xgboost["nivel_riesgo"] in ["ALTO", "CRITICO"]:
                en_riesgo += 1

        except Exception as e:
            logger.error(f"Error prediciendo insumo {insumo['id']}: {e}")
            continue

    # Ordenar por nivel de riesgo (critico primero)
    orden = {"CRITICO": 0, "ALTO": 1, "MEDIO": 2, "BAJO": 3}
    predicciones.sort(key=lambda p: orden.get(p.xgboost.nivel_riesgo, 4))

    return PrediccionMasivaResponse(
        total=len(predicciones),
        en_riesgo=en_riesgo,
        predicciones=predicciones
    )

# ── Prediccion individual ─────────────────────────────────────────────────────
@app.get("/predict/{insumo_id}", response_model=PrediccionResponse)
def predecir_insumo(
    insumo_id: int,
    token: str = Depends(verificar_token)
):
    """
    Predice el agotamiento de un insumo especifico usando Prophet + XGBoost.
    """
    # 1. Obtener datos del insumo
    insumo = obtener_stock_actual(insumo_id)
    if not insumo:
        raise HTTPException(status_code=404, detail=f"Insumo {insumo_id} no encontrado")

    # 2. Obtener historial (ultimo año)
    historial = obtener_historial_insumo(insumo_id, dias=365)

    # 3. Extraer features para XGBoost
    features = extraer_features(
        historial,
        stock_actual=insumo["stock"],
        stock_minimo=insumo["stock_minimo"]
    )

    # 4. Prophet: predecir dias hasta agotamiento
    resultado_prophet = predecir_agotamiento_prophet(
        historial,
        stock_actual=insumo["stock"],
        stock_minimo=insumo["stock_minimo"]
    )

    # 5. XGBoost: clasificar nivel de riesgo
    resultado_xgboost = predecir_riesgo_xgboost(features)

    # 6. Construir resumen ejecutivo
    alerta, mensaje, recomendacion = _construir_resumen(
        insumo, resultado_prophet, resultado_xgboost
    )

    logger.info(
        f"Prediccion — insumo: '{insumo['nombre']}' | "
        f"riesgo: {resultado_xgboost['nivel_riesgo']} | "
        f"dias hasta 0: {resultado_prophet.get('dias_hasta_cero')}"
    )

    return PrediccionResponse(
        insumo_id=insumo_id,
        nombre=insumo["nombre"],
        stock_actual=insumo["stock"],
        stock_minimo=insumo["stock_minimo"],
        unidad_medida=insumo["unidad_medida"],
        prophet=ProhetResultado(**resultado_prophet),
        xgboost=XGBoostResultado(**resultado_xgboost),
        features=features,
        alerta=alerta,
        mensaje=mensaje,
        recomendacion=recomendacion
    )

# ── Entrenamiento manual ──────────────────────────────────────────────────────
@app.post("/entrenar", response_model=EntrenamientoResponse)
def entrenar(token: str = Depends(verificar_token)):
    """
    Reentrena el modelo XGBoost con todos los datos historicos actuales.
    Spring Boot puede llamar esto despues de cargar datos historicos iniciales.
    """
    exito, registros = reentrenar_modelo_automatico()
    return EntrenamientoResponse(
        exito=exito,
        mensaje="Modelo entrenado exitosamente" if exito else "Datos insuficientes para entrenar",
        registros=registros
    )

# ── Helpers privados ──────────────────────────────────────────────────────────

def reentrenar_modelo_automatico() -> tuple[bool, int]:
    """
    Genera datos de entrenamiento a partir del historial real
    y reentrena el modelo XGBoost.
    """
    logger.info("Iniciando reentrenamiento automatico del modelo XGBoost...")
    insumos = obtener_todos_los_insumos()
    datos_entrenamiento = []

    for insumo in insumos:
        try:
            historial = obtener_historial_insumo(insumo["id"], dias=365)
            if historial.empty or len(historial) < 30:
                continue

            # Generar snapshots historicos con etiqueta real
            for ventana_dias in [30, 60, 90, 120]:
                if len(historial) < ventana_dias + 30:
                    continue

                h_pasado   = historial.iloc[:ventana_dias]
                h_futuro   = historial.iloc[ventana_dias:ventana_dias + 30]

                # Calcular stock simulado en ese momento historico
                stock_pasado = insumo["stock"] + h_futuro["salidas"].sum() - h_futuro["entradas"].sum()
                stock_pasado = max(0, int(stock_pasado))

                features = extraer_features(h_pasado, stock_pasado, insumo["stock_minimo"])

                # Etiqueta real: que paso en los siguientes 30 dias
                salidas_futuras = h_futuro["salidas"].sum()
                stock_al_final  = stock_pasado - salidas_futuras

                if stock_al_final <= 0:
                    riesgo_real = "CRITICO"
                elif stock_al_final <= insumo["stock_minimo"]:
                    riesgo_real = "ALTO"
                elif stock_al_final <= insumo["stock_minimo"] * 1.5:
                    riesgo_real = "MEDIO"
                else:
                    riesgo_real = "BAJO"

                features["riesgo_real"] = riesgo_real
                datos_entrenamiento.append(features)

        except Exception as e:
            logger.error(f"Error generando datos entrenamiento insumo {insumo['id']}: {e}")

    if not datos_entrenamiento:
        logger.warning("No se generaron datos de entrenamiento")
        return False, 0

    exito = entrenar_modelo(datos_entrenamiento)
    logger.info(f"Reentrenamiento completado — {len(datos_entrenamiento)} registros | exito: {exito}")
    if exito:
        # El modelo cambio: las predicciones guardadas ya no son validas.
        _recalcular_cache_en_segundo_plano()
    return exito, len(datos_entrenamiento)


def _construir_resumen(insumo: dict, r_prophet: dict, r_xgboost: dict) -> tuple[str, str, str]:
    """Genera mensaje y recomendacion legible para el frontend."""
    nivel  = r_xgboost["nivel_riesgo"]
    dias_0 = r_prophet.get("dias_hasta_cero")
    dias_m = r_prophet.get("dias_hasta_stock_minimo")
    nombre = insumo["nombre"]

    alerta = nivel in ["ALTO", "CRITICO"]

    if nivel == "CRITICO":
        mensaje = f"'{nombre}' sin stock — requiere reposicion urgente."
        recomendacion = "Realizar pedido de emergencia al proveedor inmediatamente."
    elif nivel == "ALTO":
        if dias_0 and dias_0 <= 14:
            mensaje = f"'{nombre}' se agotara en aproximadamente {dias_0} dias."
        else:
            mensaje = f"'{nombre}' tiene stock bajo — riesgo alto de agotamiento."
        recomendacion = f"Realizar pedido esta semana. Stock actual: {insumo['stock']} {insumo['unidad_medida']}."
    elif nivel == "MEDIO":
        if dias_m:
            mensaje = f"'{nombre}' alcanzara el stock minimo en ~{dias_m} dias."
        else:
            mensaje = f"'{nombre}' tiene consumo moderado. Monitorear."
        recomendacion = "Planificar reposicion en los proximos 15 dias."
    else:
        mensaje = f"'{nombre}' tiene stock suficiente."
        recomendacion = "Sin accion requerida por ahora."

    return alerta, mensaje, recomendacion
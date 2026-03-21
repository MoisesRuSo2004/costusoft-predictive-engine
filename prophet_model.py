import pandas as pd
import numpy as np
from prophet import Prophet
import logging
from datetime import date, timedelta

logger = logging.getLogger(__name__)


def predecir_agotamiento_prophet(
    historial: pd.DataFrame,
    stock_actual: int,
    stock_minimo: int
) -> dict:
    """
    Usa Prophet para predecir cuantos dias faltan para que el insumo
    llegue al stockMinimo (punto de alerta) o a cero (agotamiento).

    Prophet necesita un DataFrame con columnas 'ds' (fecha) y 'y' (valor).
    Usamos las salidas acumuladas para modelar el consumo.

    Retorna:
        dias_hasta_stock_minimo: int  (puede ser None si no se predice agotamiento)
        dias_hasta_cero:         int  (puede ser None)
        consumo_diario_promedio: float
        fecha_alerta_estimada:   str | None
        fecha_agotamiento_estimada: str | None
        confianza:               float  (0.0 - 1.0)
        suficiente_historial:    bool
    """
    resultado = {
        "dias_hasta_stock_minimo":    None,
        "dias_hasta_cero":            None,
        "consumo_diario_promedio":    0.0,
        "fecha_alerta_estimada":      None,
        "fecha_agotamiento_estimada": None,
        "confianza":                  0.0,
        "suficiente_historial":       False,
        "metodo":                     "prophet"
    }

    if historial.empty or len(historial) < 14:
        logger.warning("Historial insuficiente para Prophet — usando promedio simple")
        return _fallback_promedio_simple(historial, stock_actual, stock_minimo, resultado)

    # Preparar datos para Prophet: modelamos el consumo acumulado
    df_prophet = historial[["fecha", "salidas"]].copy()
    df_prophet.columns = ["ds", "y"]
    df_prophet = df_prophet[df_prophet["y"] >= 0]

    # Si no hay salidas reales, no hay que predecir agotamiento
    total_salidas = df_prophet["y"].sum()
    if total_salidas == 0:
        logger.info("Sin salidas registradas — sin riesgo de agotamiento")
        resultado["confianza"] = 0.9
        resultado["suficiente_historial"] = True
        return resultado

    try:
        # Configurar Prophet con estacionalidad semanal
        # (los uniformes tienen picos al inicio de año escolar)
        model = Prophet(
            yearly_seasonality=len(historial) >= 180,  # solo si hay 6+ meses
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.05,   # no sobreajustar
            interval_width=0.80
        )

        model.fit(df_prophet)

        # Predecir los proximos 90 dias
        futuro = model.make_future_dataframe(periods=90)
        forecast = model.predict(futuro)

        # Calcular consumo diario promedio de los ultimos 30 dias
        ultimos_30 = df_prophet.tail(30)
        consumo_diario = ultimos_30["y"].mean()
        resultado["consumo_diario_promedio"] = round(float(consumo_diario), 2)

        # Simular stock futuro dia a dia
        stock_simulado = float(stock_actual)
        hoy = pd.Timestamp.now().normalize()

        predicciones_futuras = forecast[forecast["ds"] > hoy].copy()

        dias_hasta_minimo = None
        dias_hasta_cero   = None

        for i, row in predicciones_futuras.iterrows():
            consumo_dia = max(0, row["yhat"])
            stock_simulado -= consumo_dia

            dias_transcurridos = (row["ds"] - hoy).days

            if dias_hasta_minimo is None and stock_simulado <= stock_minimo:
                dias_hasta_minimo = dias_transcurridos
                resultado["fecha_alerta_estimada"] = row["ds"].strftime("%Y-%m-%d")

            if dias_hasta_cero is None and stock_simulado <= 0:
                dias_hasta_cero = dias_transcurridos
                resultado["fecha_agotamiento_estimada"] = row["ds"].strftime("%Y-%m-%d")
                break

        resultado["dias_hasta_stock_minimo"]    = dias_hasta_minimo
        resultado["dias_hasta_cero"]            = dias_hasta_cero
        resultado["suficiente_historial"]       = True

        # Confianza basada en cantidad de datos disponibles
        resultado["confianza"] = min(0.95, len(historial) / 180)

        logger.info(
            f"Prophet completado — consumo/dia: {consumo_diario:.2f} | "
            f"dias hasta minimo: {dias_hasta_minimo} | hasta cero: {dias_hasta_cero}"
        )

    except Exception as e:
        logger.error(f"Error en Prophet: {e}")
        return _fallback_promedio_simple(historial, stock_actual, stock_minimo, resultado)

    return resultado


def _fallback_promedio_simple(
    historial: pd.DataFrame,
    stock_actual: int,
    stock_minimo: int,
    resultado: dict
) -> dict:
    """
    Calculo de respaldo cuando no hay suficiente historial para Prophet.
    Usa promedio simple de consumo diario.
    """
    resultado["metodo"] = "promedio_simple"

    if historial.empty or historial["salidas"].sum() == 0:
        resultado["confianza"] = 0.5
        return resultado

    consumo_diario = historial["salidas"].mean()
    resultado["consumo_diario_promedio"] = round(float(consumo_diario), 2)

    if consumo_diario > 0:
        dias_hasta_minimo = int((stock_actual - stock_minimo) / consumo_diario)
        dias_hasta_cero   = int(stock_actual / consumo_diario)

        if dias_hasta_minimo > 0:
            resultado["dias_hasta_stock_minimo"] = dias_hasta_minimo
            fecha_alerta = date.today() + timedelta(days=dias_hasta_minimo)
            resultado["fecha_alerta_estimada"] = fecha_alerta.isoformat()

        if dias_hasta_cero > 0:
            resultado["dias_hasta_cero"] = dias_hasta_cero
            fecha_agot = date.today() + timedelta(days=dias_hasta_cero)
            resultado["fecha_agotamiento_estimada"] = fecha_agot.isoformat()

    resultado["confianza"] = 0.4
    return resultado
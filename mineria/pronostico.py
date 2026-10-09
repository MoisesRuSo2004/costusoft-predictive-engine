"""
Prediccion dinamica de consumo de insumos.

A diferencia de /predict/todos (todos los insumos, 90 dias fijos), aqui el
usuario elige que predecir y como:
  - uno o varios insumos para compararlos,
  - el horizonte (7 a 365 dias) y la agrupacion (dia, semana, mes),
  - un escenario "¿y si?": un pedido hipotetico cuyas prendas se traducen a
    insumos con la receta de cada uniforme (uniforme_insumos).

Por cada insumo se entrega el pronostico con su banda de confianza, el stock
proyectado (con y sin escenario), cuando se llegaria al minimo y a cero, la
compra sugerida y la precision del modelo medida con backtesting.
"""
import logging
import math
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from prophet import Prophet

from mineria.datos import consumo_diario, consumo_por_prendas, fecha_ultimo_movimiento, obtener_insumos
from mineria.esquemas import (
    ComparacionModelo,
    CompraSugerida,
    Precision,
    PronosticoInsumo,
    PronosticoRequest,
    PronosticoResponse,
    PuntoSerie,
)

logger = logging.getLogger(__name__)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

# El consumo de cada insumo es esporadico (unas pocas salidas al mes), asi que
# a nivel diario la serie es casi toda ceros. Se modela por MES, la escala de
# las temporadas escolares, con seis años de historia para la estacionalidad.
DIAS_ENTRENAMIENTO = 6 * 365
# Meses que se reservan para medir la precision (backtesting).
MESES_BACKTEST = 6
# Historial que se muestra en la grafica antes del pronostico.
DIAS_HISTORIAL_VISIBLE = 180

_FRECUENCIA = {"dia": "D", "semana": "W-MON", "mes": "MS"}


# ── Modelos candidatos ───────────────────────────────────────────────────────
# Para cada insumo se prueban varios modelos con backtesting y se usa el que
# menos se equivoco. Asi el sistema nunca usa un modelo mas complejo que pierda
# contra una regla simple (seleccion de modelo por validacion).

def _mensual(historia: pd.DataFrame) -> pd.DataFrame:
    """Consumo total por mes calendario (ds = primer dia del mes)."""
    return historia.set_index("ds")["y"].resample("MS").sum().reset_index()


def _meses_futuros(meses: pd.DataFrame, cantidad: int) -> pd.DatetimeIndex:
    return pd.date_range(meses["ds"].max() + pd.offsets.MonthBegin(1), periods=cantidad, freq="MS")


def _modelo_prophet(meses: pd.DataFrame, cantidad: int) -> pd.DataFrame:
    modelo = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
        interval_width=0.80,
    )
    modelo.fit(meses)
    return modelo.predict(pd.DataFrame({"ds": _meses_futuros(meses, cantidad)}))[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def _modelo_promedio(meses: pd.DataFrame, cantidad: int) -> pd.DataFrame:
    """Promedio de los ultimos 12 meses; banda entre los percentiles 10 y 90."""
    base = meses["y"].tail(12)
    return pd.DataFrame({
        "ds": _meses_futuros(meses, cantidad),
        "yhat": float(base.mean()),
        "yhat_lower": float(base.quantile(0.1)),
        "yhat_upper": float(base.quantile(0.9)),
    })


def _modelo_estacional(meses: pd.DataFrame, cantidad: int) -> pd.DataFrame:
    """Promedio del mismo mes calendario en los ultimos 3 anios; banda = minimo y maximo."""
    filas = []
    for mes in _meses_futuros(meses, cantidad):
        mismos = meses[meses["ds"].dt.month == mes.month]["y"].tail(3)
        filas.append((mes, float(mismos.mean()), float(mismos.min()), float(mismos.max())))
    return pd.DataFrame(filas, columns=["ds", "yhat", "yhat_lower", "yhat_upper"])


MODELOS = {
    "prophet": (_modelo_prophet, "Prophet: tendencia y estacionalidad anual"),
    "promedio_12m": (_modelo_promedio, "Promedio de los últimos 12 meses"),
    "estacional": (_modelo_estacional, "Promedio del mismo mes en los últimos 3 años"),
}


def _error_total(real: np.ndarray, predicho: np.ndarray) -> Optional[float]:
    """Error sobre el consumo TOTAL del periodo, en %: es el que importa para decidir cuanto comprar."""
    total = real.sum()
    return float(abs(predicho.sum() - total) / total * 100) if total > 0 else None


def _evaluar(real: np.ndarray, predicho: np.ndarray) -> tuple[float, Optional[float]]:
    """
    MAE y WAPE. El WAPE (error total / consumo total) es el porcentaje usado
    para demanda irregular: a diferencia del MAPE no se dispara en los meses de
    poco consumo, y ordena los modelos igual que el MAE.
    """
    absolutos = np.abs(real - predicho)
    total = real.sum()
    wape = float(absolutos.sum() / total * 100) if total > 0 else None
    return float(absolutos.mean()), wape


def _interpretar(wape: Optional[float], error_total: Optional[float]) -> str:
    """
    Lectura para el usuario. Para comprar importa acertar el TOTAL del periodo:
    un insumo puede variar mucho mes a mes y aun asi sumar lo esperado.
    """
    if wape is None:
        return "Sin consumo en los meses evaluados: no se puede medir el error."
    if wape <= 20:
        return "Muy buena: se equivocó menos del 20 % del consumo real."
    if wape <= 40:
        return "Aceptable: útil para planear, con margen de error moderado."
    if error_total is not None and error_total <= 20:
        return "Útil para comprar: mes a mes varía mucho, pero acierta el total del periodo."
    return "Baja: el consumo de este insumo es muy irregular; tómalo como referencia."


def _elegir_modelo(meses: pd.DataFrame) -> tuple[str, Optional[Precision]]:
    """
    Backtesting de cada candidato sobre los ultimos 6 meses completos; gana el
    de menor error absoluto medio. Con poca historia se usa el promedio.
    """
    if len(meses) < MESES_BACKTEST + 24 or (meses["y"] > 0).sum() < 12:
        return "promedio_12m", None
    entrenamiento, real = meses.iloc[:-MESES_BACKTEST], meses["y"].iloc[-MESES_BACKTEST:].to_numpy()

    comparacion = []
    for nombre, (modelo, descripcion) in MODELOS.items():
        try:
            predicho = modelo(entrenamiento, MESES_BACKTEST)["yhat"].clip(lower=0).to_numpy()
        except Exception as e:  # noqa: BLE001 — un candidato que falla simplemente no compite
            logger.warning(f"Modelo {nombre} fallo en el backtesting: {e}")
            continue
        mae, wape = _evaluar(real, predicho)
        total = _error_total(real, predicho)
        comparacion.append(ComparacionModelo(
            modelo=nombre, descripcion=descripcion, mae=round(mae, 2),
            wape=None if wape is None else round(wape, 1),
            error_total=None if total is None else round(total, 1),
        ))
    if not comparacion:
        return "promedio_12m", None

    comparacion.sort(key=lambda c: c.mae)
    ganador = comparacion[0]
    return ganador.modelo, Precision(
        meses_evaluados=MESES_BACKTEST,
        modelo_elegido=ganador.modelo,
        mae=ganador.mae,
        wape=ganador.wape,
        error_total=ganador.error_total,
        interpretacion=_interpretar(ganador.wape, ganador.error_total),
        comparacion=comparacion,
    )


def _pronosticar(historia: pd.DataFrame, dias: int) -> tuple[pd.DataFrame, str, Optional[Precision]]:
    """
    Consumo diario esperado para los `dias` siguientes al ultimo dia de la
    historia. Se pronostica por mes (sin el mes en curso, que esta incompleto)
    con el modelo que gano el backtesting, y cada mes se reparte por igual
    entre sus dias. Columnas: ds, yhat, yhat_lower, yhat_upper.
    """
    corte = historia["ds"].max()
    mes_en_curso = corte.to_period("M").to_timestamp()
    completos = _mensual(historia[historia["ds"] < mes_en_curso])
    fin = corte + pd.Timedelta(days=dias)
    cantidad_meses = (fin.year - mes_en_curso.year) * 12 + fin.month - mes_en_curso.month + 1

    nombre, precision = _elegir_modelo(completos)
    mensual = MODELOS[nombre][0](completos, cantidad_meses)
    for columna in ("yhat", "yhat_lower", "yhat_upper"):
        mensual[columna] = mensual[columna].clip(lower=0)

    dias_futuros = pd.date_range(corte + pd.Timedelta(days=1), periods=dias, freq="D")
    por_mes = mensual.set_index("ds")
    diario = pd.DataFrame({"ds": dias_futuros})
    inicio_mes = diario["ds"].dt.to_period("M").dt.to_timestamp()
    dias_del_mes = diario["ds"].dt.days_in_month.to_numpy()
    for columna in ("yhat", "yhat_lower", "yhat_upper"):
        diario[columna] = inicio_mes.map(por_mes[columna]).fillna(0).to_numpy() / dias_del_mes
    return diario, nombre, precision


# ── Stock y compra ────────────────────────────────────────────────────────────

def _proyectar_stock(
    stock_inicial: float,
    stock_minimo: float,
    pronostico: pd.DataFrame,
    consumo_extra: float = 0.0,
    fecha_extra: Optional[pd.Timestamp] = None,
) -> tuple[pd.DataFrame, Optional[str], Optional[str]]:
    """Stock dia a dia descontando el consumo esperado (y el del escenario en su fecha)."""
    consumo = pronostico.set_index("ds")["yhat"].copy()
    if consumo_extra and fecha_extra is not None and fecha_extra in consumo.index:
        consumo.loc[fecha_extra] += consumo_extra
    stock = stock_inicial - consumo.cumsum()

    def primera_fecha(condicion: pd.Series) -> Optional[str]:
        fechas = condicion[condicion].index
        return fechas[0].strftime("%Y-%m-%d") if len(fechas) else None

    alerta = primera_fecha(stock <= stock_minimo) if stock_inicial > stock_minimo else pronostico["ds"].iloc[0].strftime("%Y-%m-%d")
    agotamiento = primera_fecha(stock <= 0)
    return stock.reset_index().rename(columns={0: "stock", "yhat": "stock"}), alerta, agotamiento


def _compra_sugerida(
    stock: float,
    stock_minimo: float,
    consumo_total: float,
    fecha_alerta: Optional[str],
    corte: date,
    horizonte: int,
    plazo_proveedor: int,
) -> CompraSugerida:
    """Cuanto comprar para terminar el horizonte sin bajar del minimo, y antes de cuando."""
    necesidad = consumo_total + stock_minimo - stock
    if necesidad <= 0:
        return CompraSugerida(cantidad=0, fecha_limite=None, motivo="El stock alcanza para todo el horizonte sin bajar del mínimo.")
    cantidad = math.ceil(necesidad)
    referencia = date.fromisoformat(fecha_alerta) if fecha_alerta else corte + timedelta(days=horizonte)
    limite = max(corte, referencia - timedelta(days=plazo_proveedor))
    motivo = (
        f"Cubre el consumo esperado del horizonte y mantiene el stock mínimo. "
        f"Pídelo a más tardar {plazo_proveedor} días antes de llegar al mínimo (plazo del proveedor)."
    )
    return CompraSugerida(cantidad=cantidad, fecha_limite=limite.isoformat(), motivo=motivo)


# ── Series para la grafica ───────────────────────────────────────────────────

def _agrupar(df: pd.DataFrame, columnas: dict[str, str], agrupacion: str, suma: bool = True) -> list[PuntoSerie]:
    """Convierte una serie diaria a puntos agrupados por dia, semana o mes."""
    if df.empty:
        return []
    serie = df.set_index("ds")[list(columnas)]
    serie = serie.resample(_FRECUENCIA[agrupacion]).sum() if suma else serie.resample(_FRECUENCIA[agrupacion]).last()
    puntos = []
    for fecha, fila in serie.iterrows():
        datos = {destino: round(float(fila[origen]), 2) for origen, destino in columnas.items()}
        puntos.append(PuntoSerie(fecha=fecha.strftime("%Y-%m-%d"), **datos))
    return puntos


# ── Orquestacion ─────────────────────────────────────────────────────────────

def calcular_pronostico(solicitud: PronosticoRequest) -> PronosticoResponse:
    hoy = date.today()
    ultimo = fecha_ultimo_movimiento() or hoy
    # Si los datos no llegan hasta hoy, se pronostica desde el ultimo dia con
    # movimientos: tomar los dias sin registros como "consumo cero" falsearia el modelo.
    corte = min(hoy, ultimo)
    desactualizados = (hoy - ultimo).days > 30

    insumos = obtener_insumos(solicitud.insumo_ids)
    consumo_escenario = consumo_por_prendas([p.model_dump() for p in solicitud.escenario.prendas]) if solicitud.escenario else {}
    fecha_escenario = None
    if solicitud.escenario:
        pedida = solicitud.escenario.fecha or corte + timedelta(days=1)
        fecha_escenario = pd.Timestamp(min(max(pedida, corte + timedelta(days=1)), corte + timedelta(days=solicitud.horizonte_dias)))

    resultados = []
    for insumo_id in solicitud.insumo_ids:
        insumo = insumos.get(insumo_id)
        if insumo is None:
            continue
        historia = consumo_diario(insumo_id, corte - timedelta(days=DIAS_ENTRENAMIENTO), corte)
        pronostico, metodo, precision = _pronosticar(historia, solicitud.horizonte_dias)

        stock = float(insumo["stock"])
        minimo = float(insumo["stock_minimo"])
        proyeccion, alerta, agotamiento = _proyectar_stock(stock, minimo, pronostico)
        extra = round(consumo_escenario.get(insumo_id, 0.0), 2)
        con_escenario = None
        alerta_esc = agotamiento_esc = None
        if extra > 0:
            con_escenario, alerta_esc, agotamiento_esc = _proyectar_stock(stock, minimo, pronostico, extra, fecha_escenario)

        consumo_total = float(pronostico["yhat"].sum())
        resultados.append(PronosticoInsumo(
            insumo_id=insumo_id,
            nombre=insumo["nombre"],
            unidad_medida=insumo["unidad_medida"],
            stock_actual=stock,
            stock_minimo=minimo,
            historial=_agrupar(historia.tail(DIAS_HISTORIAL_VISIBLE), {"y": "valor"}, solicitud.agrupacion),
            pronostico=_agrupar(pronostico, {"yhat": "valor", "yhat_lower": "minimo", "yhat_upper": "maximo"}, solicitud.agrupacion),
            stock_proyectado=_agrupar(proyeccion, {"stock": "valor"}, solicitud.agrupacion, suma=False),
            stock_proyectado_escenario=None if con_escenario is None else _agrupar(con_escenario, {"stock": "valor"}, solicitud.agrupacion, suma=False),
            consumo_pronosticado=round(consumo_total, 2),
            consumo_escenario=extra,
            fecha_alerta=alerta,
            fecha_agotamiento=agotamiento,
            fecha_alerta_escenario=alerta_esc,
            fecha_agotamiento_escenario=agotamiento_esc,
            compra_sugerida=_compra_sugerida(
                stock, minimo, consumo_total + extra, alerta_esc or alerta, corte,
                solicitud.horizonte_dias, solicitud.plazo_proveedor_dias,
            ),
            precision=precision,
            metodo=metodo,
        ))

    return PronosticoResponse(
        fecha_corte=corte.isoformat(),
        datos_desactualizados=desactualizados,
        horizonte_dias=solicitud.horizonte_dias,
        agrupacion=solicitud.agrupacion,
        insumos=resultados,
        no_encontrados=[i for i in solicitud.insumo_ids if i not in insumos],
    )

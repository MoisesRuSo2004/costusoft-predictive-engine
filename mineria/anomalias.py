"""
Deteccion de anomalias en los movimientos de inventario.

Cada renglon de movimiento (una salida o entrada de un insumo) se compara con
la historia DE ESE MISMO INSUMO, con dos tipos de senales:
  - cantidad relativa: z-score robusto (mediana / MAD) y veces la mediana,
  - repeticion: dias desde el movimiento anterior del insumo y cuantos hubo
    en los 7 dias previos (salidas repetidas),
y tres metodos:
  - estadistico por colas (criterio de Iglewicz-Hoaglin): z robusto de la
    cantidad para las cantidades ALTAS y z robusto del logaritmo para las
    BAJAS (la cantidad no baja de 0: en escala lineal las bajas no se ven),
  - Isolation Forest y Local Outlier Factor sobre las senales combinadas.

Niveles, elegidos con la validacion: alerta ALTA = la prueba estadistica;
alerta MEDIA = Isolation Forest y LOF coinciden. Lo que marca un solo modelo
se descarta: en la validacion resulto ser casi todo ruido.

Una anomalia NO es un fraude: es un movimiento que se aleja de lo habitual y
merece revision (error de registro, perdida, pedido extraordinario...).

Validacion: sobre una copia en memoria se inyectan anomalias simuladas de tres
tipos (cantidad alta, cantidad baja, rafaga de repetidos) y se mide cuantas
detecta cada metodo. La base de datos no se modifica.
"""
import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from mineria.datos import fecha_ultimo_movimiento, movimientos_detalle
from mineria.esquemas import (
    AnomaliaMovimiento,
    AnomaliasRequest,
    AnomaliasResponse,
    MetricaMetodo,
    ResumenAnomalias,
    ValidacionDetector,
)

logger = logging.getLogger(__name__)

# Constante de Iglewicz-Hoaglin: z modificado = 0,6745 * (x - mediana) / MAD.
_K_MAD = 0.6745

# Umbrales del z robusto (cantidades altas / bajas en escala log) y proporcion
# esperada de rarezas para los modelos, segun la sensibilidad elegida.
_SENSIBILIDAD = {
    "baja": {"z_alto": 4.5, "z_bajo": 4.5, "proporcion": 0.01},
    "media": {"z_alto": 3.5, "z_bajo": 4.0, "proporcion": 0.02},
    "alta": {"z_alto": 3.0, "z_bajo": 3.5, "proporcion": 0.04},
}

_SENALES = ["z_cantidad", "log_veces_mediana", "z_ritmo", "movimientos_7d"]
_NOMBRE_METODO = {
    "estadistico": "Estadístico (z robusto por colas)",
    "isolation_forest": "Isolation Forest",
    "lof": "Local Outlier Factor",
    "alertas": "Alertas del sistema (alta + media)",
}


# ── Senales ──────────────────────────────────────────────────────────────────

def _z_robusto(valores: pd.Series) -> pd.Series:
    mediana = valores.median()
    mad = (valores - mediana).abs().median()
    if mad == 0:
        mad = valores.std() or 1.0
    return _K_MAD * (valores - mediana) / mad


def _calcular_senales(movimientos: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega a cada renglon sus senales, calculadas por insumo y tipo de
    movimiento con TODA su historia (no solo el periodo analizado).
    """
    df = movimientos.sort_values(["tipo", "insumo_id", "fecha", "renglon_id"]).copy()
    grupos = df.groupby(["tipo", "insumo_id"], group_keys=False)

    df["mediana"] = grupos["cantidad"].transform("median")
    df["p10"] = grupos["cantidad"].transform(lambda s: s.quantile(0.10))
    df["p90"] = grupos["cantidad"].transform(lambda s: s.quantile(0.90))
    df["veces_mediana"] = df["cantidad"] / df["mediana"].replace(0, np.nan)
    df["log_veces_mediana"] = np.log(df["veces_mediana"].clip(lower=1e-3)).fillna(0)
    df["z_cantidad"] = grupos["cantidad"].transform(_z_robusto)
    df["z_log"] = grupos["cantidad"].transform(lambda s: _z_robusto(np.log(s.clip(lower=0.5))))

    # Ritmo: dias desde el movimiento anterior del mismo insumo frente a su
    # ritmo habitual (negativo = mucho antes de lo normal).
    df["dias_desde_anterior"] = grupos["fecha"].diff().dt.days
    df["ritmo_habitual"] = grupos["dias_desde_anterior"].transform("median").fillna(30)
    habitual = df["ritmo_habitual"]
    df["z_ritmo"] = ((df["dias_desde_anterior"].fillna(habitual) - habitual) / habitual.clip(lower=1)).clip(-1, 3)

    # Movimientos del mismo insumo en los 7 dias anteriores (sin contar este).
    def _ventana_7d(grupo: pd.DataFrame) -> pd.Series:
        serie = pd.Series(1, index=grupo["fecha"])
        return pd.Series(serie.rolling("7D").sum().to_numpy() - 1, index=grupo.index)

    df["movimientos_7d"] = grupos[["fecha"]].apply(_ventana_7d)
    return df


# ── Metodos ──────────────────────────────────────────────────────────────────

def _normalizar(x: np.ndarray) -> np.ndarray:
    rango = np.ptp(x)
    return (x - x.min()) / rango if rango else np.zeros_like(x, dtype=float)


def _detectar(df: pd.DataFrame, sensibilidad: str) -> pd.DataFrame:
    """Marca cada renglon con lo que dice cada metodo y un puntaje (0 a 1) para ordenar."""
    config = _SENSIBILIDAD[sensibilidad]
    resultado = df.copy()
    resultado["estadistico"] = (resultado["z_cantidad"] > config["z_alto"]) | (resultado["z_log"] < -config["z_bajo"])

    senales = StandardScaler().fit_transform(resultado[_SENALES].fillna(0).to_numpy())
    bosque = IsolationForest(n_estimators=200, contamination=config["proporcion"], random_state=42)
    resultado["isolation_forest"] = bosque.fit_predict(senales) == -1
    vecinos = LocalOutlierFactor(n_neighbors=35, contamination=config["proporcion"])
    resultado["lof"] = vecinos.fit_predict(senales) == -1

    rareza_cantidad = np.clip(np.maximum(resultado["z_cantidad"].to_numpy(), -resultado["z_log"].to_numpy()), 0, None)
    resultado["puntaje"] = (
        _normalizar(rareza_cantidad) + _normalizar(-bosque.score_samples(senales)) + _normalizar(-vecinos.negative_outlier_factor_)
    ) / 3
    resultado["nivel"] = np.where(
        resultado["estadistico"], "alta",
        np.where(resultado["isolation_forest"] & resultado["lof"], "media", None),
    )
    return resultado


def _cantidad(valor: float, unidad: str) -> str:
    """'1 unidad', '2 unidades', '1 metro'..."""
    if round(valor) == 1:
        singular = unidad[:-2] if unidad.endswith("des") else unidad[:-1] if unidad.endswith("s") else unidad
        return f"1 {singular}"
    return f"{valor:.0f} {unidad}"


def _explicar(fila: pd.Series) -> tuple[str, str]:
    """Tipo de anomalia y explicacion en lenguaje claro."""
    unidad = fila["unidad_medida"].lower()
    verbo = "Salida" if fila["tipo"] == "SALIDA" else "Entrada"
    rango = f"{fila['p10']:.0f}–{fila['p90']:.0f} {unidad}"
    cantidad = _cantidad(fila["cantidad"], unidad)
    if fila["z_cantidad"] > 3 or fila["veces_mediana"] >= 2.5:
        return "cantidad_alta", (
            f"{verbo} de {cantidad}, cuando lo habitual es {rango} "
            f"({fila['veces_mediana']:.1f} veces la mediana)."
        )
    if fila["z_log"] < -3 or fila["veces_mediana"] <= 0.3:
        return "cantidad_baja", f"{verbo} de solo {cantidad}; lo habitual es {rango}."
    if fila["movimientos_7d"] >= 2:
        return "repetida", (
            f"{int(fila['movimientos_7d']) + 1} movimientos del insumo en 7 días; lo habitual es uno cada "
            f"{fila['ritmo_habitual']:.0f} días."
        )
    dias = "sin movimiento previo" if pd.isna(fila["dias_desde_anterior"]) else f"{fila['dias_desde_anterior']:.0f} días desde el anterior"
    return "combinacion", (
        f"Combinación poco común de cantidad ({cantidad}; habitual {rango}) "
        f"y ritmo ({dias}; habitual cada {fila['ritmo_habitual']:.0f})."
    )


# ── Validacion con anomalias inyectadas ──────────────────────────────────────

def _inyectar(df: pd.DataFrame, cantidad: int, semilla: int = 7) -> pd.DataFrame:
    """
    Copia en memoria con anomalias simuladas de tres tipos, por partes iguales
    (la columna `inyectada` dice cual):
      - "alta": cantidades de 4 a 10 veces la mediana del insumo,
      - "baja": cantidades de 3 % a 10 % de la mediana,
      - "rafaga": el mismo insumo repetido en los 2 dias siguientes.
    """
    rng = np.random.default_rng(semilla)
    base = df[["tipo", "insumo_id", "fecha", "cantidad", "renglon_id"]].copy()
    base["inyectada"] = ""
    medianas = base.groupby(["tipo", "insumo_id"])["cantidad"].median()

    nuevas = []
    siguiente_id = int(base["renglon_id"].max()) + 1
    for i, (_, fila) in enumerate(base.sample(n=min(cantidad, len(base)), random_state=semilla).iterrows()):
        mediana = medianas[(fila["tipo"], fila["insumo_id"])]
        tipo = ("alta", "baja", "rafaga")[i % 3]
        if tipo == "rafaga":
            for desfase in (1, 2):
                nuevas.append({**fila, "fecha": fila["fecha"] + pd.Timedelta(days=desfase),
                               "cantidad": round(mediana * rng.uniform(0.8, 1.2)), "renglon_id": siguiente_id, "inyectada": tipo})
                siguiente_id += 1
            continue
        factor = rng.uniform(4, 10) if tipo == "alta" else rng.uniform(0.03, 0.1)
        nuevas.append({**fila, "cantidad": max(1, round(mediana * factor)), "renglon_id": siguiente_id, "inyectada": tipo})
        siguiente_id += 1
    return pd.concat([base, pd.DataFrame(nuevas)], ignore_index=True)


def _validar(movimientos: pd.DataFrame, sensibilidad: str, cantidad: int) -> ValidacionDetector:
    detectado = _detectar(_calcular_senales(_inyectar(movimientos, cantidad)), sensibilidad)
    tipo = detectado["inyectada"].to_numpy()
    verdad = tipo != ""

    marcas = {
        "estadistico": detectado["estadistico"].to_numpy(),
        "isolation_forest": detectado["isolation_forest"].to_numpy(),
        "lof": detectado["lof"].to_numpy(),
        "alertas": detectado["nivel"].notna().to_numpy(),
    }
    metricas = []
    for metodo, marca in marcas.items():
        vp = int((marca & verdad).sum())
        fp = int((marca & ~verdad).sum())
        precision = vp / (vp + fp) if vp + fp else 0.0
        recall = vp / int(verdad.sum()) if verdad.any() else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metricas.append(MetricaMetodo(
            metodo=metodo,
            nombre=_NOMBRE_METODO[metodo],
            precision=round(precision * 100, 1),
            recall=round(recall * 100, 1),
            f1=round(f1 * 100, 1),
            falsas_alarmas=fp,
            recall_por_tipo={
                t: round(float((marca & (tipo == t)).sum()) / max(1, int((tipo == t).sum())) * 100, 1)
                for t in ("alta", "baja", "rafaga")
            },
        ))
    return ValidacionDetector(
        anomalias_inyectadas=int(verdad.sum()),
        movimientos_evaluados=len(detectado),
        metricas=metricas,
    )


# ── Orquestacion ─────────────────────────────────────────────────────────────

def detectar_anomalias(solicitud: AnomaliasRequest) -> AnomaliasResponse:
    hasta = solicitud.hasta or fecha_ultimo_movimiento() or date.today()
    desde = min(solicitud.desde or hasta - timedelta(days=365), hasta)
    tipos = ["SALIDA", "ENTRADA"] if solicitud.tipo == "ambos" else [solicitud.tipo.upper()[:-1]]

    movimientos = movimientos_detalle(tipos)
    if solicitud.insumo_ids:
        movimientos = movimientos[movimientos["insumo_id"].isin(solicitud.insumo_ids)]
    if movimientos.empty:
        return AnomaliasResponse(
            desde=desde.isoformat(), hasta=hasta.isoformat(), sensibilidad=solicitud.sensibilidad,
            resumen=ResumenAnomalias(movimientos_analizados=0, alta=0, media=0, por_tipo={}),
            anomalias=[], validacion=None,
        )

    # Las senales y los modelos usan toda la historia (asi "lo habitual" no
    # depende del periodo elegido); se informa solo lo del periodo.
    detectado = _detectar(_calcular_senales(movimientos), solicitud.sensibilidad)
    en_periodo = detectado[(detectado["fecha"].dt.date >= desde) & (detectado["fecha"].dt.date <= hasta)]
    alertas = en_periodo[en_periodo["nivel"].notna()].sort_values(["estadistico", "puntaje"], ascending=False)

    anomalias = []
    por_tipo: dict[str, int] = {}
    for _, fila in alertas.iterrows():
        tipo_anomalia, razon = _explicar(fila)
        por_tipo[tipo_anomalia] = por_tipo.get(tipo_anomalia, 0) + 1
        if len(anomalias) >= solicitud.limite:
            continue
        anomalias.append(AnomaliaMovimiento(
            movimiento=fila["tipo"],
            movimiento_id=int(fila["movimiento_id"]),
            renglon_id=int(fila["renglon_id"]),
            fecha=fila["fecha"].strftime("%Y-%m-%d"),
            insumo_id=int(fila["insumo_id"]),
            insumo=fila["insumo"],
            unidad_medida=fila["unidad_medida"],
            cantidad=float(fila["cantidad"]),
            mediana_habitual=round(float(fila["mediana"]), 2),
            rango_habitual=[round(float(fila["p10"]), 2), round(float(fila["p90"]), 2)],
            veces_mediana=round(float(fila["veces_mediana"]), 2),
            dias_desde_anterior=None if pd.isna(fila["dias_desde_anterior"]) else int(fila["dias_desde_anterior"]),
            tercero=fila["tercero"],
            descripcion=fila["descripcion"],
            nivel=fila["nivel"],
            tipo_anomalia=tipo_anomalia,
            razon=razon,
            metodos=[m for m in ("estadistico", "isolation_forest", "lof") if bool(fila[m])],
            puntaje=round(float(fila["puntaje"]), 3),
        ))

    return AnomaliasResponse(
        desde=desde.isoformat(),
        hasta=hasta.isoformat(),
        sensibilidad=solicitud.sensibilidad,
        resumen=ResumenAnomalias(
            movimientos_analizados=len(en_periodo),
            alta=int((en_periodo["nivel"] == "alta").sum()),
            media=int((en_periodo["nivel"] == "media").sum()),
            por_tipo=por_tipo,
        ),
        anomalias=anomalias,
        validacion=_validar(movimientos, solicitud.sensibilidad, solicitud.anomalias_simuladas) if solicitud.validar else None,
    )

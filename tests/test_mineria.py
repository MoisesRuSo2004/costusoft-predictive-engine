"""
Pruebas del modulo de mineria de datos con datos sinteticos de comportamiento
conocido (no usan la base de datos).

Ejecutar:  python -m pytest -q
"""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from mineria import anomalias as A
from mineria import pronostico as P


# ── Pronostico ──────────────────────────────────────────────────────────────

def _historia_diaria(meses_valores: list[float], fin: str = "2026-05-29") -> pd.DataFrame:
    """Historia diaria cuyo total mensual es el indicado (todo el consumo el dia 10)."""
    fin_ts = pd.Timestamp(fin)
    inicio = (fin_ts.to_period("M") - len(meses_valores)).to_timestamp()
    dias = pd.date_range(inicio, fin_ts, freq="D")
    y = pd.Series(0.0, index=dias)
    for i, valor in enumerate(meses_valores):
        mes = (inicio.to_period("M") + i).to_timestamp()
        y[mes + pd.Timedelta(days=9)] = valor
    return pd.DataFrame({"ds": dias, "y": y.to_numpy()})


def test_errores_mae_wape_y_total():
    real = np.array([10.0, 20.0, 30.0])
    predicho = np.array([12.0, 18.0, 30.0])
    mae, wape = P._evaluar(real, predicho)
    assert mae == pytest.approx(4 / 3)
    assert wape == pytest.approx(4 / 60 * 100)          # error total / consumo total
    assert P._error_total(real, predicho) == pytest.approx(0.0)   # 60 vs 60
    assert P._evaluar(np.zeros(3), predicho)[1] is None  # sin consumo no hay porcentaje


def test_seleccion_elige_estacional_cuando_el_patron_se_repite():
    # Un año que se repite exacto: el estacional acierta y debe ganar.
    anio = [5, 5, 40, 40, 5, 5, 60, 5, 5, 5, 5, 20]
    meses = pd.DataFrame({
        "ds": pd.date_range("2020-01-01", periods=6 * 12, freq="MS"),
        "y": [float(v) for v in anio * 6],
    })
    nombre, precision = P._elegir_modelo(meses)
    assert nombre == "estacional"
    assert precision.mae == pytest.approx(0.0, abs=1e-9)
    assert precision.wape == pytest.approx(0.0, abs=1e-9)
    assert [c.modelo for c in precision.comparacion][0] == "estacional"


def test_ante_un_empate_gana_el_modelo_mas_simple():
    # Consumo constante: los tres modelos aciertan; por parsimonia gana el promedio.
    meses = pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=60, freq="MS"), "y": 25.0})
    nombre, precision = P._elegir_modelo(meses)
    assert precision.mae == pytest.approx(0.0, abs=1e-6)
    assert nombre == "promedio_12m"


def test_poca_historia_usa_promedio_sin_backtesting():
    nombre, precision = P._elegir_modelo(pd.DataFrame({"ds": pd.date_range("2025-01-01", periods=10, freq="MS"), "y": 3.0}))
    assert (nombre, precision) == ("promedio_12m", None)


def test_pronostico_reparte_cada_mes_entre_sus_dias():
    historia = _historia_diaria([30.0] * 40)
    diario, modelo, _ = P._pronosticar(historia, 33)
    assert len(diario) == 33
    assert diario["ds"].iloc[0] == pd.Timestamp("2026-05-30")
    # Junio completo dentro del horizonte suma lo de un mes (30).
    junio = diario[diario["ds"].dt.month == 6]["yhat"].sum()
    assert junio == pytest.approx(30.0, rel=1e-6)


def test_proyeccion_de_stock_alerta_y_agotamiento():
    pronostico = pd.DataFrame({"ds": pd.date_range("2026-06-01", periods=10, freq="D"), "yhat": 2.0})
    proyeccion, alerta, agotamiento = P._proyectar_stock(10.0, 4.0, pronostico)
    assert proyeccion["stock"].tolist()[:3] == [8.0, 6.0, 4.0]
    assert alerta == "2026-06-03"        # 10 - 3*2 = 4 = minimo
    assert agotamiento == "2026-06-05"   # 10 - 5*2 = 0
    _, alerta_pedido, agot_pedido = P._proyectar_stock(10.0, 4.0, pronostico, 6.0, pd.Timestamp("2026-06-01"))
    assert alerta_pedido == "2026-06-01" and agot_pedido == "2026-06-02"


def test_compra_sugerida():
    sin_compra = P._compra_sugerida(100, 10, 50, None, date(2026, 5, 29), 90, 7)
    assert sin_compra.cantidad == 0 and sin_compra.fecha_limite is None
    compra = P._compra_sugerida(20, 10, 45.2, "2026-06-20", date(2026, 5, 29), 90, 7)
    assert compra.cantidad == 36                 # ceil(45.2 + 10 - 20)
    assert compra.fecha_limite == "2026-06-13"   # 7 dias antes de llegar al minimo
    urgente = P._compra_sugerida(20, 10, 45.2, "2026-05-30", date(2026, 5, 29), 90, 7)
    assert urgente.fecha_limite == "2026-05-29"  # nunca antes de la fecha de corte


# ── Anomalias ───────────────────────────────────────────────────────────────

def _movimientos_sinteticos(semilla: int = 3) -> pd.DataFrame:
    """Tres insumos con cantidades normales (alrededor de 30) y ritmo regular."""
    rng = np.random.default_rng(semilla)
    filas, renglon = [], 1
    for insumo in (1, 2, 3):
        for fecha in pd.date_range("2022-01-01", periods=300, freq="5D"):
            filas.append({"tipo": "SALIDA", "movimiento_id": renglon, "renglon_id": renglon, "fecha": fecha,
                          "insumo_id": insumo, "insumo": f"Insumo {insumo}", "unidad_medida": "Unidades",
                          "cantidad": float(max(5, round(rng.normal(30, 6)))), "tercero": None, "descripcion": None})
            renglon += 1
    return pd.DataFrame(filas)


def test_z_robusto_ignora_valores_extremos():
    valores = pd.Series([10.0, 11, 9, 10, 12, 8, 10, 1000])
    z = A._z_robusto(valores)
    assert abs(z.iloc[:-1]).max() < 2      # los normales siguen siendo normales
    assert z.iloc[-1] > 100                # el extremo resalta muchisimo


def test_detecta_cantidades_altas_y_bajas_inyectadas():
    movimientos = _movimientos_sinteticos()
    validacion = A._validar(movimientos, "media", cantidad=60)
    estadistico = next(m for m in validacion.metricas if m.metodo == "estadistico")
    assert estadistico.recall_por_tipo["alta"] == 100.0
    assert estadistico.recall_por_tipo["baja"] >= 90.0
    assert validacion.anomalias_inyectadas == 80   # 20 altas + 20 bajas + 20 rafagas x 2


def test_sin_anomalias_casi_no_hay_alertas_altas():
    detectado = A._detectar(A._calcular_senales(_movimientos_sinteticos()), "media")
    assert (detectado["nivel"] == "alta").mean() < 0.01


def test_explicacion_y_singular():
    assert A._cantidad(1, "unidades") == "1 unidad"
    assert A._cantidad(1, "metros") == "1 metro"
    assert A._cantidad(3, "unidades") == "3 unidades"
    fila = pd.Series({"tipo": "SALIDA", "unidad_medida": "Metros", "cantidad": 180.0, "p10": 10.0, "p90": 20.0,
                      "veces_mediana": 12.0, "z_cantidad": 25.0, "z_log": 6.0, "movimientos_7d": 0,
                      "dias_desde_anterior": 30.0, "ritmo_habitual": 30.0})
    tipo, razon = A._explicar(fila)
    assert tipo == "cantidad_alta"
    assert razon == "Salida de 180 metros, cuando lo habitual es 10–20 metros (12.0 veces la mediana)."

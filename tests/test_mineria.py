"""
Pruebas de la deteccion de anomalias con datos sinteticos de comportamiento
conocido (no usan la base de datos).

Ejecutar:  python -m pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from mineria import anomalias as A


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

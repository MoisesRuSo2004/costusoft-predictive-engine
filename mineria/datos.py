"""
Consultas de solo lectura para el modulo de mineria de datos.
"""
import logging
from datetime import date
from typing import Optional

import pandas as pd
from sqlalchemy import text

from database import engine

logger = logging.getLogger(__name__)


def fecha_ultimo_movimiento() -> Optional[date]:
    """Ultima fecha con entradas o salidas registradas (la "fecha de corte" de los datos)."""
    with engine.connect() as conn:
        return conn.execute(text("""
            SELECT GREATEST(
                (SELECT MAX(fecha) FROM salidas),
                (SELECT MAX(fecha) FROM entradas)
            )
        """)).scalar()


def obtener_insumos(ids: list[int]) -> dict[int, dict]:
    """Datos basicos de los insumos pedidos, por id."""
    with engine.connect() as conn:
        filas = conn.execute(text("""
            SELECT id, nombre, stock, stock_minimo, unidad_medida
            FROM insumos
            WHERE id = ANY(:ids)
        """), {"ids": ids}).mappings().all()
    return {f["id"]: dict(f) for f in filas}


def consumo_diario(insumo_id: int, desde: date, hasta: date) -> pd.DataFrame:
    """
    Salidas diarias de un insumo entre dos fechas, con los dias sin
    movimiento en cero. Columnas: ds (fecha), y (cantidad).
    """
    consulta = text("""
        WITH dias AS (
            SELECT generate_series(CAST(:desde AS date), CAST(:hasta AS date), INTERVAL '1 day')::date AS fecha
        ),
        salidas_dia AS (
            SELECT s.fecha, SUM(ds.cantidad) AS cantidad
            FROM salidas s
            JOIN detalle_salidas ds ON ds.salida_id = s.id
            WHERE ds.insumo_id = :insumo_id
              AND s.fecha BETWEEN :desde AND :hasta
            GROUP BY s.fecha
        )
        SELECT d.fecha AS ds, COALESCE(sd.cantidad, 0) AS y
        FROM dias d
        LEFT JOIN salidas_dia sd ON sd.fecha = d.fecha
        ORDER BY d.fecha
    """)
    with engine.connect() as conn:
        df = pd.read_sql(consulta, conn, params={"insumo_id": insumo_id, "desde": desde, "hasta": hasta})
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = df["y"].astype(float)
    return df


def consumo_por_prendas(prendas: list[dict]) -> dict[int, float]:
    """
    Insumos que consumiria fabricar las prendas indicadas, segun la receta
    de cada uniforme (uniforme_insumos). Si la receta no tiene la talla
    pedida, se usa la talla UNICA.

    prendas: [{"uniforme_id": int, "talla": str | None, "cantidad": int}]
    Retorna: {insumo_id: cantidad_total}
    """
    if not prendas:
        return {}
    total: dict[int, float] = {}
    with engine.connect() as conn:
        for prenda in prendas:
            filas = conn.execute(text("""
                SELECT insumo_id, talla, cantidad_base
                FROM uniforme_insumos
                WHERE uniforme_id = :uniforme_id
            """), {"uniforme_id": prenda["uniforme_id"]}).mappings().all()
            talla = (prenda.get("talla") or "").upper()
            de_la_talla = [f for f in filas if (f["talla"] or "").upper() == talla]
            receta = de_la_talla or [f for f in filas if (f["talla"] or "").upper() == "UNICA"]
            if not receta and filas:
                # Sin la talla ni UNICA: se toma la primera talla registrada de la receta.
                primera = filas[0]["talla"]
                receta = [f for f in filas if f["talla"] == primera]
            for f in receta:
                total[f["insumo_id"]] = total.get(f["insumo_id"], 0.0) + float(f["cantidad_base"]) * prenda["cantidad"]
    return total

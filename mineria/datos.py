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


def movimientos_detalle(tipos: list[str]) -> pd.DataFrame:
    """
    Renglones de movimientos confirmados (salidas y/o entradas) con su insumo y
    el tercero: colegio en las salidas, proveedor en las entradas.
    Columnas: tipo, movimiento_id, renglon_id, fecha, insumo_id, insumo,
    unidad_medida, cantidad, tercero, descripcion.
    """
    partes = []
    if "SALIDA" in tipos:
        partes.append("""
            SELECT 'SALIDA' AS tipo, s.id AS movimiento_id, ds.id AS renglon_id, s.fecha,
                   ds.insumo_id, i.nombre AS insumo, i.unidad_medida, ds.cantidad,
                   c.nombre AS tercero, s.descripcion
            FROM detalle_salidas ds
            JOIN salidas s ON s.id = ds.salida_id
            JOIN insumos i ON i.id = ds.insumo_id
            LEFT JOIN colegios c ON c.id = s.colegio_id
            WHERE s.estado = 'CONFIRMADA'
        """)
    if "ENTRADA" in tipos:
        partes.append("""
            SELECT 'ENTRADA' AS tipo, e.id AS movimiento_id, de.id AS renglon_id, e.fecha,
                   de.insumo_id, i.nombre AS insumo, i.unidad_medida, de.cantidad,
                   p.nombre AS tercero, e.descripcion
            FROM detalle_entradas de
            JOIN entradas e ON e.id = de.entrada_id
            JOIN insumos i ON i.id = de.insumo_id
            LEFT JOIN proveedores p ON p.id = e.proveedor_id
            WHERE e.estado = 'CONFIRMADA'
        """)
    with engine.connect() as conn:
        df = pd.read_sql(text(" UNION ALL ".join(partes)), conn)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["cantidad"] = df["cantidad"].astype(float)
    # Sin colegio, proveedor o descripcion: None (no NaN) para la respuesta JSON.
    for columna in ("tercero", "descripcion"):
        df[columna] = df[columna].astype(object).where(df[columna].notna(), None)
    return df

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import pandas as pd
from datetime import date
from typing import Optional
import logging

from config import settings

logger = logging.getLogger(__name__)

engine = create_engine(
    settings.DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,        # reconecta si la conexion murio
    echo=settings.ENVIRONMENT == "development"
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Dependency para FastAPI."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def obtener_historial_insumo(insumo_id: int, dias: int = 365) -> pd.DataFrame:
    """
    Obtiene el historial completo de movimientos (entradas y salidas)
    de un insumo desde PostgreSQL.

    Retorna un DataFrame con columnas: fecha, entradas, salidas, neto
    ordenado por fecha ascendente.
    """
    # PostgreSQL no interpola parametros dentro de INTERVAL '...' string literal.
    # Se usa make_interval(days => :dias) que si acepta parametro numerico.
    query = text("""
        WITH fechas AS (
            SELECT generate_series(
                CURRENT_DATE - make_interval(days => :dias),
                CURRENT_DATE,
                INTERVAL '1 day'
            )::date AS fecha
        ),
        movimientos_entrada AS (
            SELECT
                e.fecha,
                COALESCE(SUM(de.cantidad), 0) AS entradas
            FROM entradas e
            JOIN detalle_entradas de ON de.entrada_id = e.id
            WHERE de.insumo_id = :insumo_id
            GROUP BY e.fecha
        ),
        movimientos_salida AS (
            SELECT
                s.fecha,
                COALESCE(SUM(ds.cantidad), 0) AS salidas
            FROM salidas s
            JOIN detalle_salidas ds ON ds.salida_id = s.id
            WHERE ds.insumo_id = :insumo_id
            GROUP BY s.fecha
        )
        SELECT
            f.fecha,
            COALESCE(me.entradas, 0) AS entradas,
            COALESCE(ms.salidas,  0) AS salidas,
            COALESCE(me.entradas, 0) - COALESCE(ms.salidas, 0) AS neto
        FROM fechas f
        LEFT JOIN movimientos_entrada me ON me.fecha = f.fecha
        LEFT JOIN movimientos_salida  ms ON ms.fecha = f.fecha
        ORDER BY f.fecha ASC
    """)

    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={
                "insumo_id": insumo_id,
                "dias": dias
            })
        df["fecha"] = pd.to_datetime(df["fecha"])
        return df
    except Exception as e:
        logger.error(f"Error obteniendo historial insumo {insumo_id}: {e}")
        return pd.DataFrame(columns=["fecha", "entradas", "salidas", "neto"])


def obtener_stock_actual(insumo_id: int) -> Optional[dict]:
    """
    Obtiene el stock actual y stockMinimo de un insumo.
    """
    query = text("""
        SELECT id, nombre, stock, stock_minimo, unidad_medida
        FROM insumos
        WHERE id = :insumo_id
    """)
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"insumo_id": insumo_id}).fetchone()
            if result:
                return {
                    "id":           result[0],
                    "nombre":       result[1],
                    "stock":        result[2],
                    "stock_minimo": result[3],
                    "unidad_medida":result[4],
                }
        return None
    except Exception as e:
        logger.error(f"Error obteniendo stock insumo {insumo_id}: {e}")
        return None


def obtener_todos_los_insumos() -> list[dict]:
    """
    Retorna todos los insumos activos para prediccion masiva.
    """
    query = text("""
        SELECT id, nombre, stock, stock_minimo, unidad_medida
        FROM insumos
        ORDER BY nombre
    """)
    try:
        with engine.connect() as conn:
            rows = conn.execute(query).fetchall()
            return [
                {
                    "id":           r[0],
                    "nombre":       r[1],
                    "stock":        r[2],
                    "stock_minimo": r[3],
                    "unidad_medida":r[4],
                }
                for r in rows
            ]
    except Exception as e:
        logger.error(f"Error obteniendo insumos: {e}")
        return []
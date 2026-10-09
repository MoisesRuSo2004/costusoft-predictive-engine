"""
Modulo de mineria de datos de CostuSoft.

  pronostico.py  Prediccion dinamica: el usuario elige insumos, horizonte,
                 agrupacion y un escenario "¿y si?" (Prophet + backtesting).
  anomalias.py   Deteccion de movimientos atipicos (z robusto por colas,
                 Isolation Forest, LOF) validada con anomalias simuladas.
  datos.py       Consultas de solo lectura a PostgreSQL para este modulo.
  esquemas.py    Contratos de entrada y salida (Pydantic).
  rutas.py       Endpoints /mineria/* (FastAPI).
"""

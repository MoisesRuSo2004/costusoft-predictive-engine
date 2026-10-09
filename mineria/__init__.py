"""
Modulo de mineria de datos de CostuSoft.

  pronostico.py  Prediccion dinamica: el usuario elige insumos, horizonte,
                 agrupacion y un escenario "¿y si?" (Prophet + backtesting).
  datos.py       Consultas de solo lectura a PostgreSQL para este modulo.
  esquemas.py    Contratos de entrada y salida (Pydantic).
  rutas.py       Endpoints /mineria/* (FastAPI).
"""

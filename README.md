# CostuSoft Predictive Engine 🧠📊

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688)
![Machine Learning](https://img.shields.io/badge/ML-Prophet%20%2B%20XGBoost-red)
![Status](https://img.shields.io/badge/Service-Internal-lightgrey)

[cite_start]Este microservicio es el componente especializado en **Machine Learning** de la plataforma CostuSoft[cite: 165, 168]. [cite_start]Su función principal es analizar el historial de movimientos de inventario para predecir cuándo se agotarán los insumos y categorizar su nivel de riesgo[cite: 168, 194].

## 🚀 Tecnologías Core

- [cite_start]**Framework:** FastAPI (Asíncrono)[cite: 171].
- [cite_start]**Modelado de Series de Tiempo:** Prophet (Meta) para estacionalidad escolar[cite: 171, 181].
- [cite_start]**Clasificación:** XGBoost para análisis de riesgo multivariable[cite: 171, 189].
- [cite_start]**Procesamiento de Datos:** Pandas & NumPy[cite: 177].
- [cite_start]**ORM:** SQLAlchemy con PostgreSQL[cite: 177].
- [cite_start]**Programación de Tareas:** APScheduler para reentrenamiento automático[cite: 177, 208].

## 🛠️ Modelos de Inteligencia Artificial

### 1. Predicción de Agotamiento (Prophet)

- [cite_start]**Entrada:** Historial de salidas de los últimos 365 días[cite: 183].
- [cite_start]**Lógica:** Detecta patrones estacionales (inicio/fin de año escolar) para proyectar el consumo a 90 días[cite: 182, 185].
- [cite_start]**Resultado:** Fecha estimada de stock cero y días restantes[cite: 186].

### 2. Clasificación de Riesgo (XGBoost)

- [cite_start]**Variables (12 features):** Incluye stock actual, ratio de consumo, tendencia y variabilidad[cite: 191].
- [cite_start]**Salida:** Niveles de riesgo: `BAJO`, `MEDIO`, `ALTO`, `CRÍTICO`[cite: 191].
- [cite_start]**Entrenamiento:** El modelo se reentrena automáticamente cada día a las 2:00 AM para mejorar su precisión con datos frescos[cite: 208].

## 🔐 Seguridad e Integración

- [cite_start]**Aislamiento:** El servicio no es accesible desde el frontend; solo responde a peticiones del backend Core[cite: 215, 216].
- [cite_start]**Autenticación:** Requiere un `X-API-Token` en cada solicitud[cite: 216].
- [cite_start]**Documentación:** Swagger UI disponible internamente en `/docs`[cite: 171, 255].

## 📦 Instalación y Configuración

1.  **Entorno Virtual:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # venv\Scripts\activate en Windows
    ```
2.  **Dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Variables de Entorno (.env):**
    - [cite_start]`DATABASE_URL`: Conexión a PostgreSQL Supabase[cite: 177, 252].
    - [cite_start]`API_SECRET_TOKEN`: Token compartido con el backend Core[cite: 217].
4.  **Ejecución:**
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8001
    ```

---

[cite_start]© 2026 CostuSoft - Módulo de Inteligencia Artificial Confidencial[cite: 172].

## 🧪 Pruebas

Pruebas del módulo de minería de datos (datos sintéticos, no usan la base de datos):

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

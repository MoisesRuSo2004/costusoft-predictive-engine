from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/inventario_db"
    PORT: int = 8001
    API_SECRET_TOKEN: str = "mi-token-secreto-interno-cambiar-en-produccion"
    MIN_DIAS_HISTORIAL: int = 30
    ENVIRONMENT: str = "development"
    # Vigencia de la prediccion masiva en cache (segundos). Pasado este tiempo
    # se sigue respondiendo con el ultimo calculo y se recalcula en segundo plano.
    CACHE_PREDICCIONES_SEGUNDOS: int = 600

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
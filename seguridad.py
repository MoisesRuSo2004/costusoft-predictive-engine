"""
Token interno: solo el backend Core puede llamar a este microservicio.
"""
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from config import settings

api_key_header = APIKeyHeader(name="X-API-Token", auto_error=False)


def verificar_token(token: str = Security(api_key_header)):
    if token != settings.API_SECRET_TOKEN:
        raise HTTPException(status_code=403, detail="Token invalido")
    return token

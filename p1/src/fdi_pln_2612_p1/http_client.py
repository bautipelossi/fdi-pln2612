"""Cliente HTTP robusto con retry y manejo de errores."""

from __future__ import annotations

import time
from typing import Any

import requests

from .config import BUTLER_URL, HEADERS, REQUEST_TIMEOUT


def _url(path: str) -> str:
    """Construye la URL completa para un endpoint."""
    return f"{BUTLER_URL.rstrip('/')}{path}"


def request_with_retry(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    payload: Any = None,
    max_retries: int = 3,
) -> requests.Response:
    """Realiza una petición HTTP con reintentos exponenciales."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return requests.request(
                method,
                _url(path),
                params=params,
                json=payload,
                timeout=REQUEST_TIMEOUT,
                headers=HEADERS,
            )
        except requests.exceptions.RequestException as e:
            last_exc = e
            time.sleep(0.25 * (2**attempt))
    raise last_exc  # type: ignore


def http_get(path: str, params: dict[str, Any] | None = None) -> requests.Response:
    """Petición GET."""
    return request_with_retry("GET", path, params=params)


def http_post(
    path: str, payload: Any, params: dict[str, Any] | None = None
) -> requests.Response:
    """Petición POST."""
    return request_with_retry("POST", path, params=params, payload=payload)


def http_delete(path: str, params: dict[str, Any] | None = None) -> requests.Response:
    """Petición DELETE."""
    return request_with_retry("DELETE", path, params=params)

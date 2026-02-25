"""Funciones para interactuar con la API de Butler."""

from __future__ import annotations

import time
from typing import Any

from .config import ALIAS, BAD_DEST_SECONDS, DEBUG, log
from .http_client import http_delete, http_get, http_post
from .models import InfoPuesto

# Estado mutable del módulo
_current_alias: str = ALIAS
BAD_DEST_UNTIL: dict[str, float] = {}


def get_alias() -> str:
    """Retorna el alias actual del agente."""
    return _current_alias


def set_alias(new_alias: str) -> None:
    """Establece el alias del agente."""
    global _current_alias
    _current_alias = new_alias


def set_alias_in_butler() -> str:
    """Registra el alias del agente en Butler.

    No incrementa el nombre si ya está tomado (probablemente
    de una sesión anterior nuestra).
    """
    global _current_alias

    r = http_post(f"/alias/{ALIAS}", payload={})
    if r.status_code == 200:
        log.info(f"Alias fijado en Butler: {ALIAS}")
        _current_alias = ALIAS
        return ALIAS

    if r.status_code == 403:
        # Alias ya tomado — probablemente nuestro de una sesión anterior
        try:
            r2 = http_get("/info", params={"agente": ALIAS})
            if r2.status_code == 200:
                log.info(f"Alias {ALIAS} ya registrado, reutilizando")
                _current_alias = ALIAS
                return ALIAS
        except Exception:
            pass

    log.warning(
        f"No se pudo registrar alias {ALIAS} ({r.status_code}), usando igualmente"
    )
    _current_alias = ALIAS
    return ALIAS


def get_info() -> InfoPuesto:
    """Obtiene el estado del puesto del agente."""
    r = http_get("/info", params={"agente": _current_alias})
    r.raise_for_status()
    data = r.json()
    if DEBUG:
        keys = list(data.keys()) if isinstance(data, dict) else type(data).__name__
        log.debug(f"[INFO] Claves respuesta: {keys}")
    return InfoPuesto.model_validate(data)


def get_gente() -> list[str]:
    """Obtiene la lista de agentes conectados."""
    r = http_get("/gente")
    r.raise_for_status()
    data = r.json()
    out = []
    for item in data:
        if isinstance(item, dict) and "alias" in item:
            out.append(item["alias"])
        elif isinstance(item, str):
            out.append(item)
    return out


def enviar_carta(dest: str, asunto: str, cuerpo: str) -> bool:
    """Envía una carta a otro agente."""
    payload = {"remi": _current_alias, "dest": dest, "asunto": asunto, "cuerpo": cuerpo}
    try:
        r = http_post("/carta", payload=payload)
    except Exception as e:
        log.warning(f"[CARTA] {dest} error: {e}")
        BAD_DEST_UNTIL[dest] = time.time() + BAD_DEST_SECONDS
        return False

    ok = r.status_code == 200
    if not ok:
        log.warning(f"[CARTA] {dest} falló ({r.status_code})")
        BAD_DEST_UNTIL[dest] = time.time() + BAD_DEST_SECONDS
    return ok


def borrar_mail(mail_id: str) -> None:
    """Borra un mail del buzón."""
    http_delete(f"/mail/{mail_id}", params={"agente": _current_alias})


def enviar_paquete(dest: str, paquete: dict[str, int]) -> bool:
    """Envía un paquete de recursos a otro agente."""
    r = http_post(
        f"/paquete/{dest}", payload=paquete, params={"agente": _current_alias}
    )
    ok = r.status_code == 200
    if not ok:
        log.warning(f"[PAQUETE] {dest} falló ({r.status_code})")
        BAD_DEST_UNTIL[dest] = time.time() + BAD_DEST_SECONDS
    return ok


def normalize_buzon(raw: Any) -> dict[str, dict[str, Any]]:
    """Normaliza datos del buzón a dict[mail_id, mail_data].

    Maneja tanto dict como list, y tanto claves en mayúscula como minúscula.
    """
    if not raw:
        return {}
    if isinstance(raw, dict):
        result: dict[str, dict[str, Any]] = {}
        for k, v in raw.items():
            if isinstance(v, dict):
                result[str(k)] = v
            else:
                result[str(k)] = {"cuerpo": str(v)}
        return result
    if isinstance(raw, list):
        result = {}
        for i, item in enumerate(raw):
            if isinstance(item, dict):
                mid = str(item.get("_id", item.get("id", f"mail_{i}")))
                result[mid] = item
            else:
                result[f"mail_{i}"] = {"cuerpo": str(item)}
        return result
    return {}

"""Modelos de datos del agente."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class InfoPuesto(BaseModel):
    """Estado del puesto del agente en Butler.

    Acepta campos en mayúscula y minúscula para
    compatibilidad con distintas versiones de Butler.
    """

    model_config = ConfigDict(populate_by_name=True)

    Alias: str = Field(default="", alias="alias")
    Recursos: dict[str, int] = Field(default_factory=dict, alias="recursos")
    Objetivo: dict[str, int] = Field(default_factory=dict, alias="objetivo")
    Buzon: dict[str, Any] | list[Any] | None = Field(default=None, alias="buzon")


class Decision(BaseModel):
    """Decisión tomada por el agente."""

    razonamiento: str = ""
    accion: dict[str, Any] = Field(default_factory=lambda: {"tipo": "esperar"})

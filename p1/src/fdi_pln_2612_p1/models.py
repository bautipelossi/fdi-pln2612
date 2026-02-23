"""Modelos de datos del agente."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class InfoPuesto(BaseModel):
    """Estado del puesto del agente en Butler."""

    Alias: str
    Recursos: dict[str, int] = Field(default_factory=dict)
    Objetivo: dict[str, int] = Field(default_factory=dict)
    Buzon: dict[str, dict[str, Any]] | None = None


class Decision(BaseModel):
    """Decisión tomada por el agente."""

    razonamiento: str = ""
    accion: dict[str, Any] = Field(default_factory=lambda: {"tipo": "esperar"})

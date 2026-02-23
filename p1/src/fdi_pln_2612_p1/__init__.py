"""Agente PLN - Práctica 1.

Módulos:
- config: Configuración y variables de entorno
- models: Modelos de datos (InfoPuesto, Decision)
- http_client: Cliente HTTP robusto
- butler_api: API de Butler
- protocol: Protocolo de mensajes
- strategy: Heurísticas de decisión
- llm: Integración con LLM (Ollama)
- main: Punto de entrada
"""

from .main import main

__all__ = ["main"]

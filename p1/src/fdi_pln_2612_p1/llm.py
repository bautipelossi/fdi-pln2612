"""Integración con LLM (Ollama) para decisiones inteligentes."""

from __future__ import annotations

import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any

from .butler_api import get_alias
from .config import LLM_MODEL, LLM_TIMEOUT, USE_LLM, log
from .models import Decision, InfoPuesto
from .protocol import parse_offer_from_text
from .strategy import can_give, can_send_offer_now, excedentes, faltantes

# Verificar disponibilidad de Ollama
try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


# =========================================================
# Prompts del sistema
# =========================================================

SYSTEM_PROMPT = """Eres un agente de comercio en una simulación multiagente.
Tu objetivo es conseguir los recursos que te faltan intercambiando con otros agentes.

REGLAS IMPORTANTES:
1. Si te faltan recursos Y tienes excedentes, DEBES hacer ofertas
2. Si hay ofertas que puedes aceptar (puedo_dar=True) y te dan algo que necesitas, ACEPTA
3. Solo espera si NO hay agentes disponibles o NO tienes recursos para ofrecer
4. Prioridad: aceptar ofertas útiles > hacer ofertas > esperar

Respuestas válidas (JSON exacto, sin explicaciones):
- Esperar: {"tipo": "esperar"}
- Ofertar: {"tipo": "ofertar", "dest": "ALIAS_DESTINO", "need_recurso": "X", "need_cantidad": 1, "offer_recurso": "Y", "offer_cantidad": 1}
- Aceptar: {"tipo": "aceptar", "mensaje_id": "ID_DEL_MENSAJE"}

Responde SOLO con el JSON."""


def build_user_prompt(
    estado: InfoPuesto, gente: list[str], mails_by_id: dict[str, dict[str, Any]]
) -> str:
    """Construye el prompt con el contexto actual de la simulación."""
    alias = get_alias()
    f = faltantes(estado)
    exc = excedentes(estado)
    otros = [a for a in gente if a != alias]

    # Formatear ofertas recibidas
    ofertas_recibidas = []
    for mid, mail in mails_by_id.items():
        remi = (mail.get("remi") or "").strip()
        cuerpo = (mail.get("cuerpo") or "").strip()
        if remi and remi != alias and remi.lower() != "sistema":
            parsed = parse_offer_from_text(cuerpo)
            if parsed:
                quiero, ofrezco = parsed
                ofertas_recibidas.append(
                    {
                        "id": mid,
                        "de": remi,
                        "pide": quiero,
                        "ofrece": ofrezco,
                        "puedo_dar": all(
                            can_give(estado, k, v) for k, v in quiero.items()
                        ),
                    }
                )

    prompt = f"""ESTADO ACTUAL:
- Mis recursos: {json.dumps(estado.Recursos, ensure_ascii=False)}
- Mi objetivo: {json.dumps(estado.Objetivo, ensure_ascii=False)}
- Me faltan: {json.dumps(f, ensure_ascii=False) if f else "nada (objetivo cumplido)"}
- Tengo de sobra (puedo ofrecer): {json.dumps(exc, ensure_ascii=False) if exc else "nada"}

AGENTES DISPONIBLES para ofertar: {json.dumps(otros, ensure_ascii=False) if otros else "ninguno"}

OFERTAS RECIBIDAS:
"""

    if ofertas_recibidas:
        for o in ofertas_recibidas:
            me_sirve = any(k in f for k in o["ofrece"].keys()) if f else False
            prompt += f"- ID={o['id']} de {o['de']}: pide {o['pide']}, ofrece {o['ofrece']}. ¿Puedo dar? {o['puedo_dar']}. ¿Me sirve? {me_sirve}\n"
    else:
        prompt += "- Ninguna\n"

    # Añadir sugerencia explícita con variación
    if f and exc and otros:
        need_list = list(f.keys())
        off_list = list(exc.keys())
        need_sug = random.choice(need_list)
        off_sug = random.choice(off_list)
        dest_sug = random.choice(otros)
        prompt += (
            f"\nRECURSOS QUE NECESITAS: {need_list}. Elige uno diferente cada vez."
        )
        prompt += f"\nRECURSOS QUE PUEDES OFRECER: {off_list}."
        prompt += f"\nEJEMPLO: Pide {need_sug} y ofrece {off_sug} a {dest_sug}."

    prompt += "\n\n¿Qué acción tomo? JSON:"
    return prompt


def decidir_con_llm(
    estado: InfoPuesto, gente: list[str], mails_by_id: dict[str, dict[str, Any]]
) -> Decision | None:
    """Usa el LLM para tomar una decisión inteligente."""
    if not OLLAMA_AVAILABLE or not USE_LLM:
        return None

    alias = get_alias()

    try:
        user_prompt = build_user_prompt(estado, gente, mails_by_id)

        # Ejecutar LLM con timeout real usando thread
        def _call_ollama():
            return ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": 0.3, "num_predict": 150},
            )

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call_ollama)
            try:
                response = future.result(timeout=LLM_TIMEOUT)
            except FuturesTimeout:
                log.warning(f"[LLM] Timeout ({LLM_TIMEOUT}s)")
                return None

        content = response["message"]["content"].strip()

        # Extracción robusta del JSON: buscar primer bloque {...}
        json_match = re.search(r"\{[^{}]*\}", content)
        if json_match:
            content = json_match.group(0)
        else:
            # Fallback: limpiar markdown
            content = re.sub(r"```(?:json)?\s*", "", content)
            content = content.replace("```", "").strip()

        accion = json.loads(content)

        # Validar la acción
        tipo = accion.get("tipo", "")
        if tipo not in ("esperar", "ofertar", "aceptar"):
            return None

        # Validaciones adicionales
        if tipo == "ofertar":
            dest = accion.get("dest", "")
            if not dest or dest == alias or not can_send_offer_now(dest):
                return None
            if not can_give(
                estado,
                accion.get("offer_recurso", ""),
                int(accion.get("offer_cantidad", 0)),
            ):
                return None

        if tipo == "aceptar":
            mid = accion.get("mensaje_id", "")
            if mid not in mails_by_id:
                return None

        return Decision(razonamiento=f"[LLM] Decidió: {tipo}", accion=accion)

    except Exception as e:
        if DEBUG:
            log.error(f"[LLM] Error: {e}")
        return None

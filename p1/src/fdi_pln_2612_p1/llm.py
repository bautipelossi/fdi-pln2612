"""Integración con LLM (Ollama) para decisiones inteligentes."""

from __future__ import annotations

import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any

from .butler_api import get_alias
from .config import (
    LLM_MODEL,
    LLM_NUM_PREDICT,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    USE_LLM,
    log,
)
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

SYSTEM_PROMPT = """Eres un agente de trueque. Respondes SOLO con JSON.

REGLA 1 (OBLIGATORIA): Si hay una oferta con puedo_dar=true y me_sirve=true → DEBES aceptarla. SIEMPRE.
REGLA 2: Si NO hay ofertas aceptables, haz una oferta pidiendo lo que necesitas.
REGLA 3: Espera solo si no puedes hacer nada.

IMPORTANTE: Aceptar ofertas es SIEMPRE prioritario sobre hacer nuevas ofertas.

Formatos JSON (responde SOLO uno, sin texto extra):
{"tipo":"aceptar","mensaje_id":"ID"}
{"tipo":"ofertar","dest":"NOMBRE","need_recurso":"X","need_cantidad":1,"offer_recurso":"Y","offer_cantidad":1}
{"tipo":"esperar"}"""


def _build_ofertas_section(
    estado: InfoPuesto,
    mails_by_id: dict[str, dict[str, Any]],
    f: dict[str, int],
) -> tuple[str, list[dict[str, Any]]]:
    """Construye la sección de ofertas recibidas del prompt.

    Retorna (texto_ofertas, lista_ofertas_aceptables).
    """
    alias = get_alias()
    ofertas_recibidas: list[dict[str, Any]] = []
    aceptables: list[dict[str, Any]] = []

    for mid, mail in mails_by_id.items():
        remi = (mail.get("remi") or "").strip()
        cuerpo = (mail.get("cuerpo") or "").strip()
        if not remi or remi == alias or remi.lower() == "sistema":
            continue

        parsed = parse_offer_from_text(cuerpo)
        if not parsed:
            continue

        quiero, ofrezco = parsed
        puedo_dar = all(can_give(estado, k, v) for k, v in quiero.items())
        me_sirve = any(k in f for k in ofrezco) if f else False
        entry = {
            "id": mid,
            "de": remi,
            "pide": quiero,
            "ofrece": ofrezco,
            "puedo_dar": puedo_dar,
            "me_sirve": me_sirve,
        }
        ofertas_recibidas.append(entry)
        if puedo_dar and me_sirve:
            aceptables.append(entry)

    if not ofertas_recibidas:
        return "Ninguna", aceptables

    lines = []
    for o in ofertas_recibidas:
        lines.append(
            f"ID={o['id']} de={o['de']} pide={o['pide']} ofrece={o['ofrece']} "
            f"puedo_dar={str(o['puedo_dar']).lower()} me_sirve={str(o['me_sirve']).lower()}"
        )
    return "\n".join(lines), aceptables


def build_user_prompt(
    estado: InfoPuesto,
    gente: list[str],
    mails_by_id: dict[str, dict[str, Any]],
) -> str:
    """Construye el prompt con el contexto actual de la simulación."""
    alias = get_alias()
    f = faltantes(estado)
    exc = excedentes(estado)
    otros = [a for a in gente if a != alias]

    ofertas_text, _ = _build_ofertas_section(estado, mails_by_id, f)

    # Construir prompt compacto para reducir latencia
    parts = [
        f"Recursos: {json.dumps(estado.Recursos, ensure_ascii=False)}",
        f"Objetivo: {json.dumps(estado.Objetivo, ensure_ascii=False)}",
        f"Faltan: {json.dumps(f, ensure_ascii=False) if f else 'nada'}",
        f"Sobran: {json.dumps(exc, ensure_ascii=False) if exc else 'nada'}",
        f"Agentes: {json.dumps(otros, ensure_ascii=False) if otros else 'ninguno'}",
        f"Ofertas:\n{ofertas_text}",
    ]

    # Sugerencia de acción concreta
    if f and exc and otros:
        need_sug = random.choice(list(f.keys()))
        off_sug = random.choice(list(exc.keys()))
        dest_sug = random.choice(otros)
        parts.append(f"Sugerencia: pide {need_sug} ofreciendo {off_sug} a {dest_sug}")

    parts.append("JSON:")
    return "\n".join(parts)


def _validate_and_fix_action(
    accion: dict[str, Any],
    estado: InfoPuesto,
    gente: list[str],
    mails_by_id: dict[str, dict[str, Any]],
    aceptables: list[dict[str, Any]],
) -> Decision | None:
    """Valida la acción del LLM y la corrige si es posible.

    Si la acción no es válida, intenta generar una acción correcta
    usando las ofertas aceptables disponibles.
    """
    alias = get_alias()
    tipo = accion.get("tipo", "")

    if tipo == "esperar":
        # Si hay ofertas aceptables, forzar aceptación en vez de esperar
        if aceptables:
            best = aceptables[0]
            log.info(f"[LLM] Quiso esperar pero hay oferta aceptable, forzando")
            return Decision(
                razonamiento=f"[LLM→fix] Acepto oferta de {best['de']}",
                accion={"tipo": "aceptar", "mensaje_id": best["id"]},
            )
        return Decision(razonamiento="[LLM] Esperar", accion={"tipo": "esperar"})

    if tipo == "aceptar":
        mid = accion.get("mensaje_id", "")
        if mid and mid in mails_by_id:
            return Decision(razonamiento=f"[LLM] Acepta mensaje {mid}", accion=accion)
        # El LLM dio un ID inválido → si hay aceptables, usar la primera
        if aceptables:
            best = aceptables[0]
            return Decision(
                razonamiento=f"[LLM→fix] ID inválido, acepto {best['id']} de {best['de']}",
                accion={"tipo": "aceptar", "mensaje_id": best["id"]},
            )
        return None

    if tipo == "ofertar":
        # Si hay ofertas aceptables, forzar aceptación en vez de ofertar
        if aceptables:
            best = aceptables[0]
            log.info(
                f"[LLM] Quiso ofertar pero hay oferta aceptable de {best['de']}, forzando aceptación"
            )
            return Decision(
                razonamiento=f"[LLM→fix] Acepto oferta de {best['de']} en vez de ofertar",
                accion={"tipo": "aceptar", "mensaje_id": best["id"]},
            )

        dest = accion.get("dest", "")
        otros = [a for a in gente if a != alias]

        # Validar destino
        if not dest or dest == alias or dest.lower() in ("alias", "null", "none"):
            if otros:
                dest = random.choice(otros)
                accion["dest"] = dest
                log.info(f"[LLM→fix] dest inválido, usando {dest}")
            else:
                return None

        # Validar que el destino existe
        if dest not in otros:
            if otros:
                dest = random.choice(otros)
                accion["dest"] = dest
            else:
                return None

        if not can_send_offer_now(dest):
            # Intentar otro destino con cooldown OK
            valid = [a for a in otros if can_send_offer_now(a)]
            if valid:
                accion["dest"] = random.choice(valid)
            else:
                return None

        offer_r = accion.get("offer_recurso", "")
        offer_c = int(accion.get("offer_cantidad", 0))
        if not offer_r or offer_c <= 0:
            return None
        if not can_give(estado, offer_r, offer_c):
            # Intentar ofrecer otro recurso excedente
            exc = excedentes(estado)
            if exc:
                offer_r = random.choice(list(exc.keys()))
                accion["offer_recurso"] = offer_r
                accion["offer_cantidad"] = 1
            else:
                return None

        need_r = accion.get("need_recurso", "")
        if not need_r:
            f = faltantes(estado)
            if f:
                accion["need_recurso"] = random.choice(list(f.keys()))
                accion["need_cantidad"] = 1
            else:
                return None

        return Decision(
            razonamiento=f"[LLM] Oferta a {accion['dest']}",
            accion=accion,
        )

    return None


def decidir_con_llm(
    estado: InfoPuesto, gente: list[str], mails_by_id: dict[str, dict[str, Any]]
) -> Decision | None:
    """Usa el LLM para tomar una decisión inteligente.

    Si el LLM no responde a tiempo o da una respuesta inválida,
    intenta corregir automáticamente basándose en las ofertas
    aceptables disponibles.
    """
    if not OLLAMA_AVAILABLE or not USE_LLM:
        return None

    alias = get_alias()
    f = faltantes(estado)

    # Pre-calcular ofertas aceptables (para usar como fallback inteligente)
    _, aceptables = _build_ofertas_section(estado, mails_by_id, f)

    try:
        user_prompt = build_user_prompt(estado, gente, mails_by_id)
        log.info(f"[LLM] Consultando {LLM_MODEL}...")

        def _call_ollama():
            return ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                options={
                    "temperature": LLM_TEMPERATURE,
                    "num_predict": LLM_NUM_PREDICT,
                },
            )

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call_ollama)
            try:
                response = future.result(timeout=LLM_TIMEOUT)
            except FuturesTimeout:
                log.warning(f"[LLM] Timeout ({LLM_TIMEOUT}s)")
                # Si hay ofertas aceptables, aceptar aunque el LLM haya fallado
                if aceptables:
                    best = aceptables[0]
                    log.info(
                        f"[LLM→timeout] Acepto oferta de {best['de']} (post-timeout)"
                    )
                    return Decision(
                        razonamiento=f"[LLM timeout] Acepto oferta disponible de {best['de']}",
                        accion={"tipo": "aceptar", "mensaje_id": best["id"]},
                    )
                return None

        content = response["message"]["content"].strip()
        log.info(f"[LLM] Respuesta: {content[:120]}")

        # Extracción robusta del JSON: buscar primer bloque {...}
        json_match = re.search(r"\{[^{}]*\}", content)
        if json_match:
            content = json_match.group(0)
        else:
            content = re.sub(r"```(?:json)?\s*", "", content)
            content = content.replace("```", "").strip()

        accion = json.loads(content)

        return _validate_and_fix_action(accion, estado, gente, mails_by_id, aceptables)

    except Exception as e:
        log.error(f"[LLM] Error: {e}")
        # Último recurso: si hay ofertas aceptables, aceptar
        if aceptables:
            best = aceptables[0]
            return Decision(
                razonamiento=f"[LLM error] Acepto oferta de {best['de']}",
                accion={"tipo": "aceptar", "mensaje_id": best["id"]},
            )
        return None

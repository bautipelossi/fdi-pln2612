from __future__ import annotations

from pydantic import BaseModel, Field
import requests
import json
import time
import os
import re
import random
from typing import Any

try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


# =========================================================
# CONFIG / ENTORNO
# =========================================================

BUTLER_URL = os.getenv("FDI_PLN__BUTLER_ADDRESS")
ALIAS = os.getenv("FDI_PLN__ALIAS", "fdi-pln-2612")

SLEEP_SECONDS = int(os.getenv("FDI_PLN__SLEEP_SECONDS", "6"))
REQUEST_TIMEOUT = float(os.getenv("FDI_PLN__TIMEOUT", "10"))

# Reducir carga
GENTE_EVERY = int(os.getenv("FDI_PLN__GENTE_EVERY", "3"))  # /gente cada N ciclos

# Anti-spam
OFFER_COOLDOWN_PER_DEST = int(os.getenv("FDI_PLN__OFFER_COOLDOWN", "25"))
OFFER_COOLDOWN_GLOBAL = int(os.getenv("FDI_PLN__OFFER_COOLDOWN_GLOBAL", "10"))
BAD_DEST_SECONDS = int(os.getenv("FDI_PLN__BAD_DEST_SECONDS", "60"))

# TEST MODE (casa): permitir trade aunque baje tu objetivo (si tenés el recurso)
ALLOW_BREAK_OBJECTIVE = os.getenv("FDI_PLN__ALLOW_BREAK_OBJECTIVE", "0") == "1"
# TEST MODE: aceptar aunque no te sirva (mientras puedas pagar)
ACCEPT_ANY = os.getenv("FDI_PLN__ACCEPT_ANY", "0") == "1"
# Limpiar buzón (para que no crezca infinito)
CLEAN_INBOX = os.getenv("FDI_PLN__CLEAN_INBOX", "1") == "1"
MAX_MAILS = int(os.getenv("FDI_PLN__MAX_MAILS", "20"))

DEBUG = os.getenv("FDI_PLN__DEBUG", "1") == "1"

# LLM
LLM_MODEL = os.getenv("FDI_PLN__LLM_MODEL", "llama3.2:3b")
USE_LLM = os.getenv("FDI_PLN__USE_LLM", "1") == "1"
LLM_TIMEOUT = float(os.getenv("FDI_PLN__LLM_TIMEOUT", "30"))

if not BUTLER_URL:
    raise RuntimeError(
        "FDI_PLN__BUTLER_ADDRESS no está definida (ej: http://127.0.0.1:8000)"
    )

HEADERS = {"Connection": "close"}  # evita keep-alive y baja 10053/10054

# Memoria local
LAST_OFFER_TS_DEST: dict[str, float] = {}
LAST_OFFER_TS_GLOBAL: float = 0.0
BAD_DEST_UNTIL: dict[str, float] = {}
CICLO = 0

# cache
gente_cache: list[str] = []
estado_cache: dict[str, Any] = {"recursos": {}}


# =========================================================
# MODELOS
# =========================================================


class InfoPuesto(BaseModel):
    Alias: str
    Recursos: dict[str, int] = Field(default_factory=dict)
    Objetivo: dict[str, int] = Field(default_factory=dict)
    Buzon: dict[str, dict[str, Any]] | None = None


class Decision(BaseModel):
    razonamiento: str = ""
    accion: dict[str, Any] = Field(default_factory=lambda: {"tipo": "esperar"})


# =========================================================
# HTTP robusto (retry + close)
# =========================================================


def _url(path: str) -> str:
    return f"{BUTLER_URL.rstrip('/')}{path}"


def request_with_retry(
    method: str, path: str, *, params: dict[str, Any] | None = None, payload: Any = None
) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(3):
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
    return request_with_retry("GET", path, params=params)


def http_post(
    path: str, payload: Any, params: dict[str, Any] | None = None
) -> requests.Response:
    return request_with_retry("POST", path, params=params, payload=payload)


def http_delete(path: str, params: dict[str, Any] | None = None) -> requests.Response:
    return request_with_retry("DELETE", path, params=params)


# =========================================================
# Butler API
# =========================================================


def set_alias_in_butler() -> str:
    base = ALIAS
    for attempt in range(1, 6):
        candidate = base if attempt == 1 else f"{base}-{attempt}"
        r = http_post(f"/alias/{candidate}", payload={})
        if r.status_code == 200:
            print(f"[OK] Alias fijado en Butler: {candidate}")
            return candidate
        if r.status_code == 403:
            continue
        if DEBUG:
            print(f"[WARN] No pude setear alias {candidate}: {r.status_code} {r.text}")

    candidate = f"{base}-{random.randint(1000, 9999)}"
    r = http_post(f"/alias/{candidate}", payload={})
    if r.status_code == 200:
        print(f"[OK] Alias fijado en Butler: {candidate}")
        return candidate

    print("[WARN] No pude setear alias final, sigo con ALIAS base.")
    return base


def get_info() -> InfoPuesto:
    r = http_get("/info", params={"agente": ALIAS})
    r.raise_for_status()
    return InfoPuesto.model_validate(r.json())


def get_gente() -> list[str]:
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
    payload = {"remi": ALIAS, "dest": dest, "asunto": asunto, "cuerpo": cuerpo}
    try:
        r = http_post("/carta", payload=payload)
    except Exception as e:
        if DEBUG:
            print(f"[CARTA] -> {dest} EXC:", e)
        BAD_DEST_UNTIL[dest] = time.time() + BAD_DEST_SECONDS
        return False

    ok = r.status_code == 200
    rid = None
    try:
        rid = r.json().get("id")
    except Exception:
        pass
    if DEBUG:
        print(f"[CARTA] -> {dest} asunto='{asunto}' status={r.status_code} id={rid}")
        if not ok:
            print("[CARTA] body:", r.text)

    if not ok:
        BAD_DEST_UNTIL[dest] = time.time() + BAD_DEST_SECONDS
    return ok


def borrar_mail(mail_id: str) -> None:
    r = http_delete(f"/mail/{mail_id}", params={"agente": ALIAS})
    if DEBUG and r.status_code not in (200, 404):
        print(f"[WARN] borrar_mail fallo: {r.status_code} {r.text}")


def enviar_paquete(dest: str, paquete: dict[str, int]) -> bool:
    r = http_post(f"/paquete/{dest}", payload=paquete, params={"agente": ALIAS})
    ok = r.status_code == 200
    if DEBUG:
        print(f"[PAQUETE] -> {dest} {paquete} status={r.status_code}")
        if not ok:
            print("[PAQUETE] body:", r.text)
    if not ok:
        BAD_DEST_UNTIL[dest] = time.time() + BAD_DEST_SECONDS
    return ok


# =========================================================
# Protocolo parseable
# =========================================================

TAG_OFFER_RE = re.compile(
    r"\[OFERTA_V1\]\s*quiero=(\{.*?\})\s*ofrezco=(\{.*?\})",
    re.IGNORECASE,
)
TAG_ACCEPT_RE = re.compile(
    r"\[ACEPTO_V1\]\s*te_envio=(\{.*?\})\s*espero=(\{.*?\})",
    re.IGNORECASE,
)

OFFER_RE = re.compile(
    r"necesit[oa]\s+(\d+)\s+([a-zA-ZáéíóúñÑ]+).*?ofrezc[oa]\s+(\d+)\s+([a-zA-ZáéíóúñÑ]+)",
    re.IGNORECASE | re.DOTALL,
)


def build_offer_body(
    need_item: str, need_qty: int, offer_item: str, offer_qty: int
) -> str:
    line1 = f"[OFERTA_V1] quiero={json.dumps({need_item: need_qty}, ensure_ascii=False)} ofrezco={json.dumps({offer_item: offer_qty}, ensure_ascii=False)}"
    line2 = f"Necesito {need_qty} {need_item} y ofrezco {offer_qty} {offer_item}."
    return line1 + "\n" + line2


def _extract_first_item(d: dict[str, int]) -> tuple[str, int] | None:
    """Extrae el primer item de un dict. Retorna None si está vacío o tiene >1 item."""
    items = list(d.items())
    if len(items) != 1:
        if DEBUG and len(items) > 1:
            print(f"[WARN] Oferta con múltiples items ({len(items)}), ignorando.")
        return None
    return items[0]


def build_accept_body(
    give_to_other: dict[str, int], expect_from_other: dict[str, int]
) -> str:
    line1 = f"[ACEPTO_V1] te_envio={json.dumps(give_to_other, ensure_ascii=False)} espero={json.dumps(expect_from_other, ensure_ascii=False)}"
    # Tomar primer item de cada dict para el texto legible
    give_items = list(give_to_other.items())
    expect_items = list(expect_from_other.items())
    gk, gv = give_items[0] if give_items else ("", 0)
    ek, ev = expect_items[0] if expect_items else ("", 0)
    line2 = f"Acepto tu oferta. Te envié {gv} {gk}. Enviame {ev} {ek}."
    return line1 + "\n" + line2


def parse_offer_from_text(texto: str) -> tuple[dict[str, int], dict[str, int]] | None:
    t = texto or ""
    m = TAG_OFFER_RE.search(t)
    if m:
        try:
            quiero = json.loads(m.group(1))
            ofrezco = json.loads(m.group(2))
            if isinstance(quiero, dict) and isinstance(ofrezco, dict):
                return quiero, ofrezco
        except Exception:
            pass

    m2 = OFFER_RE.search(t)
    if m2:
        need_qty = int(m2.group(1))
        need_item = m2.group(2).lower()
        offer_qty = int(m2.group(3))
        offer_item = m2.group(4).lower()
        return {need_item: need_qty}, {offer_item: offer_qty}

    return None


def parse_accept_from_text(texto: str) -> tuple[dict[str, int], dict[str, int]] | None:
    t = texto or ""
    m = TAG_ACCEPT_RE.search(t)
    if not m:
        return None
    try:
        te_envio = json.loads(m.group(1))
        espero = json.loads(m.group(2))
        if isinstance(te_envio, dict) and isinstance(espero, dict):
            return te_envio, espero
    except Exception:
        return None
    return None


# =========================================================
# Heurísticas
# =========================================================


def faltantes(estado: InfoPuesto) -> dict[str, int]:
    f = {}
    for r, obj in (estado.Objetivo or {}).items():
        cur = (estado.Recursos or {}).get(r, 0)
        if cur < obj:
            f[r] = obj - cur
    return f


def excedentes(estado: InfoPuesto) -> dict[str, int]:
    exc = {}
    for r, qty in (estado.Recursos or {}).items():
        if r == "oro":
            continue
        obj = (estado.Objetivo or {}).get(r, 0)
        if qty > obj:
            exc[r] = qty - obj
        if r not in (estado.Objetivo or {}) and qty > 0:
            exc[r] = max(exc.get(r, 0), qty)
    return exc


def can_give(estado: InfoPuesto, item: str, qty: int) -> bool:
    have = estado.Recursos.get(item, 0)
    if have < qty:
        return False
    if ALLOW_BREAK_OBJECTIVE:
        return True
    # modo clase: no romper objetivo
    obj = estado.Objetivo.get(item, 0)
    return (have - obj) >= qty


# =========================================================
# Automatismos: responder ACEPTO_V1
# =========================================================


def procesar_mails_automaticos(mails_by_id: dict[str, dict[str, Any]]) -> None:
    for mid, mail in list(mails_by_id.items()):
        remi = (mail.get("remi") or "").strip()
        asunto = (mail.get("asunto") or "").strip()
        cuerpo = (mail.get("cuerpo") or "").strip()

        if remi.lower() == "sistema":
            if CLEAN_INBOX:
                borrar_mail(mid)
                mails_by_id.pop(mid, None)
            continue

        acc = parse_accept_from_text(cuerpo)
        if acc and remi:
            _, espero = acc
            ok_send = True
            for k, v in espero.items():
                if estado_cache["recursos"].get(k, 0) < int(v):
                    ok_send = False
                    break

            if ok_send:
                enviar_paquete(remi, {k: int(v) for k, v in espero.items()})
                enviar_carta(remi, "Envío", f"Listo. Te envié {espero}.")
            else:
                enviar_carta(
                    remi, "No puedo", f"No tengo suficiente para enviarte {espero}."
                )

            if CLEAN_INBOX:
                borrar_mail(mid)
                mails_by_id.pop(mid, None)


# =========================================================
# Anti-spam
# =========================================================


def can_send_offer_now(dest: str) -> bool:
    global LAST_OFFER_TS_GLOBAL
    now = time.time()
    if dest in BAD_DEST_UNTIL and now < BAD_DEST_UNTIL[dest]:
        return False
    if now - LAST_OFFER_TS_GLOBAL < OFFER_COOLDOWN_GLOBAL:
        return False
    last_d = LAST_OFFER_TS_DEST.get(dest, 0.0)
    if now - last_d < OFFER_COOLDOWN_PER_DEST:
        return False
    return True


def mark_offer_sent(dest: str) -> None:
    global LAST_OFFER_TS_GLOBAL
    now = time.time()
    LAST_OFFER_TS_GLOBAL = now
    LAST_OFFER_TS_DEST[dest] = now


# =========================================================
# LLM - Decisiones Inteligentes
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
    f = faltantes(estado)
    exc = excedentes(estado)
    otros = [a for a in gente if a != ALIAS]

    # Formatear ofertas recibidas
    ofertas_recibidas = []
    for mid, mail in mails_by_id.items():
        remi = (mail.get("remi") or "").strip()
        cuerpo = (mail.get("cuerpo") or "").strip()
        if remi and remi != ALIAS and remi.lower() != "sistema":
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
        import random as _rnd

        need_list = list(f.keys())
        off_list = list(exc.keys())
        need_sug = _rnd.choice(need_list)
        off_sug = _rnd.choice(off_list)
        dest_sug = _rnd.choice(otros)
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

    try:
        user_prompt = build_user_prompt(estado, gente, mails_by_id)

        if DEBUG:
            print("[LLM] Consultando modelo...")

        # Ejecutar LLM con timeout real usando thread
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

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
                if DEBUG:
                    print(f"[LLM] Timeout después de {LLM_TIMEOUT}s")
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

        if DEBUG:
            print(f"[LLM] Respuesta: {content}")

        accion = json.loads(content)

        # Validar la acción
        tipo = accion.get("tipo", "")
        if tipo not in ("esperar", "ofertar", "aceptar"):
            return None

        # Validaciones adicionales
        if tipo == "ofertar":
            dest = accion.get("dest", "")
            if not dest or dest == ALIAS or not can_send_offer_now(dest):
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
            print(f"[LLM] Error: {e}")
        return None


# =========================================================
# Decidir + Ejecutar
# =========================================================


def decidir_fallback(
    estado: InfoPuesto, gente: list[str], mails_by_id: dict[str, dict[str, Any]]
) -> Decision:
    """Heurísticas de respaldo cuando el LLM no está disponible."""
    f = faltantes(estado)
    exc = excedentes(estado)

    # limitar crecimiento de buzón
    if CLEAN_INBOX and len(mails_by_id) > MAX_MAILS:
        # borro extras (arbitrario) para que no explote
        for mid in list(mails_by_id.keys())[MAX_MAILS:]:
            borrar_mail(mid)
            mails_by_id.pop(mid, None)

    # 1) aceptar ofertas entrantes
    for mid, mail in list(mails_by_id.items()):
        remi = (mail.get("remi") or "").strip()
        cuerpo = (mail.get("cuerpo") or "").strip()
        if not remi or remi == ALIAS:
            continue

        parsed = parse_offer_from_text(cuerpo)
        if not parsed:
            continue

        quiero, ofrezco = parsed
        # Validar que sea exactamente 1 item por lado
        need_pair = _extract_first_item(quiero)
        offer_pair = _extract_first_item(ofrezco)
        if not need_pair or not offer_pair:
            if CLEAN_INBOX:
                borrar_mail(mid)
                mails_by_id.pop(mid, None)
            continue
        need_item, need_qty = need_pair
        offer_item, offer_qty = offer_pair
        need_qty = int(need_qty)
        offer_qty = int(offer_qty)

        if not can_give(estado, need_item, need_qty):
            if DEBUG:
                print(
                    f"[SKIP] No puedo dar {need_qty} {need_item} (ALLOW_BREAK_OBJECTIVE={ALLOW_BREAK_OBJECTIVE})."
                )
            if CLEAN_INBOX:
                borrar_mail(mid)
                mails_by_id.pop(mid, None)
            continue

        if ACCEPT_ANY or f.get(offer_item, 0) > 0:
            return Decision(
                razonamiento=f"Acepto: puedo dar {need_qty} {need_item}; recibiría {offer_qty} {offer_item}.",
                accion={"tipo": "aceptar", "mensaje_id": mid},
            )

        # si no me sirve, lo limpio igual (test)
        if CLEAN_INBOX:
            borrar_mail(mid)
            mails_by_id.pop(mid, None)

    # 2) ofertar proactivo
    otros = [a for a in gente if a != ALIAS]
    if not otros:
        return Decision(
            razonamiento="No hay otros agentes conectados.", accion={"tipo": "esperar"}
        )

    random.shuffle(otros)
    dest = None
    for cand in otros:
        if can_send_offer_now(cand):
            dest = cand
            break
    if dest is None:
        return Decision(
            razonamiento="Cooldown/global o destinos malos -> espero.",
            accion={"tipo": "esperar"},
        )

    # elegir qué pedir y qué ofrecer (ofrecer excedente si existe; si no, algo que pueda dar)
    if f:
        need_item = max(f, key=lambda k: f[k])
    else:
        need_item = random.choice(list(estado.Recursos.keys()) or ["madera"])

    offer_item = None
    if exc:
        offer_item = max(exc, key=lambda k: exc[k])
    else:
        # cualquier recurso que tenga >=1 y que pueda dar
        cand_items = [
            k for k, v in estado.Recursos.items() if v >= 1 and can_give(estado, k, 1)
        ]
        if cand_items:
            offer_item = random.choice(cand_items)

    if not offer_item or offer_item == need_item:
        return Decision(
            razonamiento="No tengo buen recurso para ofrecer ahora.",
            accion={"tipo": "esperar"},
        )

    return Decision(
        razonamiento=f"Oferto a {dest}: pido {need_item} y ofrezco {offer_item}.",
        accion={
            "tipo": "ofertar",
            "dest": dest,
            "need_recurso": need_item,
            "need_cantidad": 1,
            "offer_recurso": offer_item,
            "offer_cantidad": 1,
        },
    )


def ejecutar_decision(
    dec: Decision, estado: InfoPuesto, mails_by_id: dict[str, dict[str, Any]]
) -> None:
    accion = dec.accion or {}
    tipo = accion.get("tipo", "esperar")

    if DEBUG:
        print("\n==============================")
        print("MI ALIAS:", ALIAS)
        print("Gente:", [x for x in gente_cache if x != ALIAS])
        print("Mails:", len(mails_by_id))
        print("Recursos:", estado.Recursos)
        print("Objetivo:", estado.Objetivo)
        print("RAZONAMIENTO:", dec.razonamiento)
        print("ACCION:", accion)
        print("==============================")

    if tipo == "esperar":
        return

    if tipo == "ofertar":
        dest = accion.get("dest")
        need_r = accion.get("need_recurso")
        need_c = int(accion.get("need_cantidad", 1))
        off_r = accion.get("offer_recurso")
        off_c = int(accion.get("offer_cantidad", 1))
        if not all([dest, need_r, off_r]) or need_c <= 0 or off_c <= 0:
            return
        if not can_send_offer_now(dest):
            return

        cuerpo = build_offer_body(need_r, need_c, off_r, off_c)
        ok = enviar_carta(dest, "Oferta", cuerpo)
        if ok:
            mark_offer_sent(dest)
        return

    if tipo == "aceptar":
        mid = accion.get("mensaje_id")
        if not mid or mid not in mails_by_id:
            return

        mail = mails_by_id[mid]
        remi = (mail.get("remi") or "").strip()
        cuerpo = (mail.get("cuerpo") or "").strip()
        parsed = parse_offer_from_text(cuerpo)
        if not remi or not parsed:
            return

        quiero, ofrezco = parsed
        # Validar que sea exactamente 1 item por lado
        need_pair = _extract_first_item(quiero)
        offer_pair = _extract_first_item(ofrezco)
        if not need_pair or not offer_pair:
            if CLEAN_INBOX:
                borrar_mail(mid)
            return
        need_item, need_qty = need_pair
        offer_item, offer_qty = offer_pair
        need_qty = int(need_qty)
        offer_qty = int(offer_qty)

        if not can_give(estado, need_item, need_qty):
            if CLEAN_INBOX:
                borrar_mail(mid)
            return

        ok = enviar_paquete(remi, {need_item: need_qty})
        if ok:
            cuerpo_acc = build_accept_body(
                give_to_other={need_item: need_qty},
                expect_from_other={offer_item: offer_qty},
            )
            enviar_carta(remi, "Acepto", cuerpo_acc)
            if CLEAN_INBOX:
                borrar_mail(mid)


def ciclo_autonomo() -> None:
    global ALIAS, CICLO, gente_cache

    print("AGENTE INICIADO:", ALIAS)
    print("BUTLER_URL:", BUTLER_URL)
    print("ACCEPT_ANY:", ACCEPT_ANY)
    print("ALLOW_BREAK_OBJECTIVE:", ALLOW_BREAK_OBJECTIVE)

    ALIAS = set_alias_in_butler()

    while True:
        try:
            estado = get_info()
            estado_cache["recursos"] = dict(estado.Recursos)

            if CICLO % GENTE_EVERY == 0:
                gente_cache = get_gente()

            buzon_raw = estado.Buzon or {}
            mails_by_id: dict[str, dict[str, Any]] = {}
            for mid, m in buzon_raw.items():
                mails_by_id[mid] = (
                    dict(m) if isinstance(m, dict) else {"cuerpo": str(m)}
                )

            procesar_mails_automaticos(mails_by_id)

            # Primero intentar con LLM, si falla usar heurísticas
            dec = decidir_con_llm(estado, gente_cache, mails_by_id)
            if dec is None:
                dec = decidir_fallback(estado, gente_cache, mails_by_id)

            ejecutar_decision(dec, estado, mails_by_id)

        except Exception as e:
            print("[ERROR] EN CICLO:", e)

        CICLO += 1
        time.sleep(SLEEP_SECONDS)


def main() -> None:
    ciclo_autonomo()


if __name__ == "__main__":
    main()

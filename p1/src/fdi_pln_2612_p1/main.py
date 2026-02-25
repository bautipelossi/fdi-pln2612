"""Agente autónomo de negociación de recursos.

Este módulo es el punto de entrada principal del agente.
"""

from __future__ import annotations

import signal
import time
from typing import Any

from .butler_api import (
    borrar_mail,
    enviar_carta,
    enviar_paquete,
    get_alias,
    get_gente,
    get_info,
    normalize_buzon,
    set_alias_in_butler,
)
from .config import (
    CLEAN_INBOX,
    GENTE_EVERY,
    SLEEP_SECONDS,
    log,
    validate_config,
)
from .llm import decidir_con_llm
from .models import Decision, InfoPuesto
from .protocol import (
    build_accept_body,
    build_offer_body,
    extract_first_item,
    parse_offer_from_text,
)
from .strategy import (
    can_give,
    can_send_offer_now,
    mark_offer_sent,
    objetivo_cumplido,
    procesar_mails_automaticos,
)

# =========================================================
# Estado global del agente
# =========================================================

CICLO = 0
gente_cache: list[str] = []
estado_cache: dict[str, Any] = {"recursos": {}}
intercambios_realizados: list[dict[str, Any]] = []
_shutdown_requested = False


def _signal_handler(signum, frame):
    """Manejador de señales para shutdown gracioso."""
    global _shutdown_requested
    log.info("Señal de terminación recibida, cerrando...")
    _shutdown_requested = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# =========================================================
# Funciones auxiliares
# =========================================================


def registrar_intercambio(
    tipo: str, con: str, di: dict, recibido: dict | None = None
) -> None:
    """Registra un intercambio en el historial."""
    intercambios_realizados.append(
        {
            "ciclo": CICLO,
            "tipo": tipo,
            "con": con,
            "di": di,
            "recibido": recibido,
            "ts": time.time(),
        }
    )
    log.info(
        f"INTERCAMBIO #{len(intercambios_realizados)}: {tipo} con {con} - di={di} recibido={recibido}"
    )


def ejecutar_decision(
    dec: Decision, estado: InfoPuesto, mails_by_id: dict[str, dict[str, Any]]
) -> None:
    """Ejecuta la decisión tomada por el agente."""
    accion = dec.accion or {}
    tipo = accion.get("tipo", "esperar")

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
            log.info(f"📤 OFERTA -> {dest}: doy {off_c} {off_r} por {need_c} {need_r}")
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
        need_pair = extract_first_item(quiero)
        offer_pair = extract_first_item(ofrezco)
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
            log.info(
                f"✅ ACEPTO <- {remi}: doy {need_qty} {need_item}, recibo {offer_qty} {offer_item}"
            )
            registrar_intercambio(
                "aceptar", remi, {need_item: need_qty}, {offer_item: offer_qty}
            )
            if CLEAN_INBOX:
                borrar_mail(mid)


# =========================================================
# Ciclo principal
# =========================================================


def ciclo_autonomo() -> None:
    """Ciclo principal del agente autónomo."""
    global CICLO, gente_cache

    validate_config()

    log.info("AGENTE INICIANDO...")

    set_alias_in_butler()
    alias = get_alias()
    log.info(f"ALIAS FINAL: {alias}")

    while not _shutdown_requested:
        try:
            estado = get_info()
            estado_cache["recursos"] = dict(estado.Recursos)

            # Mostrar estado visual
            print(f"\n{'─' * 50}")
            print(f"  CICLO {CICLO}")
            print(f"{'─' * 50}")

            # Progreso por recurso objetivo
            all_done = True
            for recurso, necesito in estado.Objetivo.items():
                tengo = estado.Recursos.get(recurso, 0)
                pct = min(100, int(tengo / necesito * 100)) if necesito > 0 else 100
                bar_len = 20
                filled = int(bar_len * pct / 100)
                bar = "█" * filled + "░" * (bar_len - filled)
                check = "✅" if tengo >= necesito else "  "
                print(f"  {check} {recurso:12} [{bar}] {tengo}/{necesito}")
                if tengo < necesito:
                    all_done = False

            # Recursos extra (no en objetivo)
            extras = {
                k: v
                for k, v in estado.Recursos.items()
                if k not in estado.Objetivo and v > 0
            }
            if extras:
                extra_str = ", ".join(f"{k}:{v}" for k, v in extras.items())
                print(f"  💰 Extras: {extra_str}")

            print(f"{'─' * 50}")

            # Verificar si ya ganamos
            if objetivo_cumplido(estado):
                log.info("🎉 ¡OBJETIVO CUMPLIDO! Recursos finales: %s", estado.Recursos)
                log.info(
                    f"Total de intercambios realizados: {len(intercambios_realizados)}"
                )
                if CICLO % 10 == 0:
                    log.info("Objetivo cumplido, esperando...")
                CICLO += 1
                time.sleep(SLEEP_SECONDS)
                continue

            if CICLO % GENTE_EVERY == 0:
                gente_cache = get_gente()

            buzon_raw = estado.Buzon or {}
            mails_by_id = normalize_buzon(buzon_raw)
            if mails_by_id:
                log.info(f"📬 Buzón: {len(mails_by_id)} mensaje(s)")
                for mid, m in list(mails_by_id.items())[:5]:
                    remi = (m.get("remi") or "?")[:20]
                    asunto = (m.get("asunto") or "?")[:20]
                    log.info(f"  └ [{mid[:8]}] de={remi} asunto={asunto}")

            procesar_mails_automaticos(mails_by_id, estado_cache["recursos"])

            # Decisión con LLM (único motor de decisión)
            dec = decidir_con_llm(estado, gente_cache, mails_by_id)
            if dec is None:
                log.warning("LLM no disponible, esperando...")
                dec = Decision(
                    razonamiento="LLM no disponible",
                    accion={"tipo": "esperar"},
                )

            ejecutar_decision(dec, estado, mails_by_id)

        except Exception as e:
            log.error(f"EN CICLO: {e}")

        CICLO += 1
        time.sleep(SLEEP_SECONDS)

    # Shutdown limpio
    log.info("Agente terminado. Intercambios totales: %d", len(intercambios_realizados))


def main() -> None:
    """Punto de entrada principal."""
    ciclo_autonomo()


if __name__ == "__main__":
    main()

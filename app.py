from fastapi import FastAPI
from pydantic import BaseModel
import requests
import json
import time
import threading
import ollama
import os

# =========================================================
# CARGA DE ENTORNO
# =========================================================

BUTLER_URL = os.getenv("FDI_PLN__BUTLER_ADDRESS")
MODEL = os.getenv("FDI_PLN__MODEL", "mistral")
ALIAS = os.getenv("FDI_PLN__ALIAS", "fdi-pln-2612")
SLEEP_SECONDS = int(os.getenv("FDI_PLN__SLEEP_SECONDS", 15))

# =========================================================
# APP
# =========================================================

app = FastAPI(title="Agente Autónomo PLN")

# =========================================================
# MODELO DE ESTADO
# =========================================================

class ButlerState(BaseModel):
    Alias: list[str] | None = None
    Recursos: dict
    Objetivo: dict
    Buzon: dict | None = None

# =========================================================
# PROMPT PARA NEGOCIAR
# =========================================================

def construir_prompt(estado: dict) -> str:
    return f"""
Eres un agente autónomo en un sistema de intercambio de recursos.

OBJETIVO:
Cumplir tu objetivo intercambiando recursos con otros agentes.

REGLAS ESTRICTAS:
- RESPONDE SOLO CON JSON VÁLIDO
- NO INCLUYAS TEXTO FUERA DEL JSON
- NO INCLUYAS CÓDIGO
- NO USES BLOQUES ````
- USA SOLO LOS TIPOS DE ACCIÓN DEFINIDOS

ACCIONES POSIBLES:

1) esperar
2) pedir
3) aceptar
4) contraofertar

FORMATO OBLIGATORIO:

{{
  "razonamiento": "<breve>",
  "accion": {{
    "tipo": "esperar"
  }}
}}

{{
  "razonamiento": "<breve>",
  "accion": {{
    "tipo": "pedir",
    "recurso": "<string>",
    "cantidad": <int>
  }}
}}

{{
  "razonamiento": "<breve>",
  "accion": {{
    "tipo": "aceptar",
    "mensaje_id": "<string>"
  }}
}}

{{
  "razonamiento": "<breve>",
  "accion": {{
    "tipo": "contraofertar",
    "mensaje_id": "<string>",
    "oferta": "<string>"
  }}
}}

NO INVENTES ACCIONES.
USA SOLO ESTOS CAMPOS.

ESTADO ACTUAL:
{json.dumps(estado, ensure_ascii=False)}
"""

# =========================================================
# CONSULTA A OLLAMA
# =========================================================

def consultar_ollama(prompt: str) -> dict:
    try:
        response = ollama.generate(
            model=MODEL,
            prompt=prompt,
            format="json",
            stream=False
        )
        texto = response["response"].strip()
        return json.loads(texto)
    except Exception as e:
        print("ERROR OLLAMA:", e)
        return {
            "razonamiento": "Error del modelo",
            "accion": {"tipo": "esperar"}
        }

# =========================================================
# ENVÍO DE CARTAS
# =========================================================

def enviar_carta(mensaje: str):
    r = requests.post(
        f"{BUTLER_URL}/carta",
        json={
            "alias": ALIAS,
            "mensaje": mensaje
        }
    )
    print("CARTA ENVIADA:", mensaje)
    print("RESPUESTA BUTLER:", r.status_code)

# =========================================================
# EJECUTAR DECISIÓN
# =========================================================

def ejecutar_decision(respuesta: dict, estado: dict):
    accion = respuesta.get("accion", {})
    tipo = accion.get("tipo", "esperar")

    print("RAZONAMIENTO:", respuesta.get("razonamiento"))
    print("ACCION:", accion)

    if tipo == "esperar":
        return

    if tipo == "pedir":
        recurso = accion.get("recurso")
        cantidad = accion.get("cantidad")
        if recurso and cantidad:
            enviar_carta(
                f"Necesito {cantidad} unidades de {recurso}. ¿Qué ofrecés a cambio?"
            )

    if tipo == "aceptar":
        mensaje_id = accion.get("mensaje_id")
        if mensaje_id:
            enviar_carta(f"Acepto la oferta del mensaje {mensaje_id}")

    if tipo == "contraofertar":
        mensaje_id = accion.get("mensaje_id")
        oferta = accion.get("oferta")
        if mensaje_id and oferta:
            enviar_carta(
                f"En respuesta al mensaje {mensaje_id}, propongo: {oferta}"
            )

# =========================================================
# While de automatización
# =========================================================

def ciclo_autonomo():
    print("AGENTE INICIADO:", ALIAS)

    while True:
        try:
            estado = requests.get(f"{BUTLER_URL}/info", timeout=10).json()
            prompt = construir_prompt(estado)
            respuesta = consultar_ollama(prompt)
            ejecutar_decision(respuesta, estado)
        except Exception as e:
            print("ERROR EN CICLO:", e)

        time.sleep(SLEEP_SECONDS)

# =========================================================
# FASTAPI
# =========================================================

@app.get("/")
def healthcheck():
    return {
        "status": "activo",
        "alias": ALIAS
    }

@app.on_event("startup")
def iniciar_agente():
    threading.Thread(target=ciclo_autonomo, daemon=True).start()

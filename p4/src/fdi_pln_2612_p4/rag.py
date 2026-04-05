from __future__ import annotations

import os
import importlib
from typing import Any

from fdi_pln_2612_p4.embeddings import construir_resultados_desde_coincidencias
from fdi_pln_2612_p4.modelos import (
    CoincidenciaParrafo,
    CorpusQuijote,
    ResultadosBusqueda,
)
from fdi_pln_2612_p4.nlp_utils import (
    factor_calidad_texto,
    fragmentar_en_frases,
    normalizar_espacios,
    procesar_texto_spacy,
    resumir_parte,
)


MAX_RESULTADOS_RAG = 8
MAX_EVIDENCIAS_RAG = 4
MAX_CARACTERES_FRAGMENTO = 240
MODELO_OLLAMA_POR_DEFECTO = "llama3.2:3b"


def _clave_parrafo(coincidencia: CoincidenciaParrafo) -> tuple[int, int]:
    return coincidencia.parrafo.indice_seccion, coincidencia.parrafo.posicion_en_seccion


def _compactar_texto(
    texto: str, *, max_caracteres: int = MAX_CARACTERES_FRAGMENTO
) -> str:
    texto_limpio = normalizar_espacios(texto)
    if len(texto_limpio) <= max_caracteres:
        return texto_limpio
    return texto_limpio[: max_caracteres - 4].rstrip() + " ..."


def _combinar_coincidencias(
    resultados_clasicos: ResultadosBusqueda,
    resultados_semanticos: ResultadosBusqueda | None = None,
) -> list[CoincidenciaParrafo]:
    motores: list[tuple[tuple[CoincidenciaParrafo, ...], float]] = []
    if resultados_clasicos.coincidencias:
        motores.append((resultados_clasicos.coincidencias, 0.55))
    if resultados_semanticos and resultados_semanticos.coincidencias:
        motores.append((resultados_semanticos.coincidencias, 0.45))

    if not motores:
        return []

    peso_total = sum(peso for _, peso in motores)
    combinadas: dict[tuple[int, int], CoincidenciaParrafo] = {}
    scores_combinados: dict[tuple[int, int], float] = {}

    for coincidencias, peso in motores:
        mejor_score = coincidencias[0].score or 1.0
        peso_normalizado = peso / peso_total

        for posicion, coincidencia in enumerate(
            coincidencias[:MAX_RESULTADOS_RAG], start=1
        ):
            clave = _clave_parrafo(coincidencia)
            score_reescalado = coincidencia.score / mejor_score
            bonus_ranking = 1.0 / (posicion + 1)
            factor_calidad = factor_calidad_texto(coincidencia.parrafo.texto)
            score_final = (
                peso_normalizado
                * (0.85 * score_reescalado + 0.15 * bonus_ranking)
                * factor_calidad
            )

            combinadas.setdefault(clave, coincidencia)
            scores_combinados[clave] = scores_combinados.get(clave, 0.0) + score_final

    coincidencias_ordenadas = sorted(
        (
            CoincidenciaParrafo(
                parrafo=combinadas[clave].parrafo,
                spans=combinadas[clave].spans,
                score=score,
            )
            for clave, score in scores_combinados.items()
        ),
        key=lambda item: item.score,
        reverse=True,
    )
    return coincidencias_ordenadas[:MAX_RESULTADOS_RAG]


def preparar_contexto_rag(
    corpus: CorpusQuijote,
    resultados_clasicos: ResultadosBusqueda,
    resultados_semanticos: ResultadosBusqueda | None = None,
) -> ResultadosBusqueda:
    coincidencias = _combinar_coincidencias(resultados_clasicos, resultados_semanticos)
    return construir_resultados_desde_coincidencias(
        corpus,
        resultados_clasicos.consulta,
        resultados_clasicos.consulta_normalizada,
        coincidencias,
        ignorar_tildes=resultados_clasicos.ignorar_tildes,
    )


def _score_frase(frase: str, consulta_tokens: set[str]) -> float:
    tokens_frase = procesar_texto_spacy(frase)
    if not tokens_frase:
        return 0.0

    if not consulta_tokens:
        return 1 / len(tokens_frase)

    solapamiento = len(consulta_tokens.intersection(tokens_frase))
    if solapamiento == 0:
        return 0.0

    cobertura = solapamiento / len(consulta_tokens)
    densidad = solapamiento / len(tokens_frase)
    return cobertura + densidad


def _mejor_fragmento(texto: str, consulta_tokens: set[str]) -> str:
    mejor_frase = ""
    mejor_score = 0.0

    for inicio, fin in fragmentar_en_frases(texto):
        frase = normalizar_espacios(texto[inicio:fin])
        if not frase:
            continue

        score = _score_frase(frase, consulta_tokens)
        if score > mejor_score:
            mejor_score = score
            mejor_frase = frase

    if mejor_frase:
        return _compactar_texto(mejor_frase)

    return _compactar_texto(texto)


def _construir_bloque_evidencias(
    resultados_recuperacion: ResultadosBusqueda,
    consulta_tokens: set[str],
) -> list[tuple[str, CoincidenciaParrafo, str, str]]:
    evidencias: list[tuple[str, CoincidenciaParrafo, str, str]] = []
    for indice, coincidencia in enumerate(
        resultados_recuperacion.coincidencias[:MAX_EVIDENCIAS_RAG],
        start=1,
    ):
        referencia = f"E{indice}"
        encabezado = (
            f"{resumir_parte(coincidencia.parrafo.titulo_parte)} | "
            f"{coincidencia.parrafo.titulo_seccion}"
        )
        fragmento = _mejor_fragmento(coincidencia.parrafo.texto, consulta_tokens)
        evidencias.append((referencia, coincidencia, encabezado, fragmento))
    return evidencias


def _prompt_rag(
    consulta: str, evidencias: list[tuple[str, CoincidenciaParrafo, str, str]]
) -> str:
    lineas = [
        "Responde en espanol usando solamente la informacion de las evidencias.",
        "Si no hay informacion suficiente, dilo explicitamente.",
        "Incluye referencias en formato [E1], [E2], etc. junto a cada afirmacion relevante.",
        "No inventes datos fuera de las evidencias.",
        "",
        f"Consulta del usuario: {consulta}",
        "",
        "Evidencias:",
    ]
    for referencia, _, encabezado, fragmento in evidencias:
        lineas.append(f"[{referencia}] {encabezado}: {fragmento}")

    return "\n".join(lineas)


def _generar_respuesta_ollama(prompt: str) -> str | None:
    try:
        ollama = importlib.import_module("ollama")
    except ModuleNotFoundError:
        return None

    modelo = os.getenv("RAG_OLLAMA_MODEL", MODELO_OLLAMA_POR_DEFECTO)
    try:
        respuesta: Any = ollama.chat(
            model=modelo,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un asistente de QA sobre El Quijote. "
                        "Responde solo con evidencias proporcionadas y cita referencias [E#]."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.1},
        )
    except Exception:
        return None

    if hasattr(respuesta, "model_dump"):
        try:
            respuesta = respuesta.model_dump()
        except Exception:
            pass

    mensaje: Any = None
    if isinstance(respuesta, dict):
        mensaje = respuesta.get("message")
    elif hasattr(respuesta, "message"):
        mensaje = getattr(respuesta, "message")

    contenido: Any = None
    if isinstance(mensaje, dict):
        contenido = mensaje.get("content")
    elif hasattr(mensaje, "content"):
        contenido = getattr(mensaje, "content")

    if not isinstance(contenido, str):
        return None
    contenido_limpio = contenido.strip()
    return contenido_limpio or None


def _responder_extractivo(
    resultados_recuperacion: ResultadosBusqueda,
    evidencias: list[tuple[str, CoincidenciaParrafo, str, str]],
) -> str:
    secciones_principales = ", ".join(
        list(dict.fromkeys(encabezado for _, _, encabezado, _ in evidencias))[:3]
    )
    sintesis = " ".join(fragmento for _, _, _, fragmento in evidencias[:3])

    lineas = [
        "Respuesta basada solo en los pasajes recuperados:",
        f"He reunido {len(resultados_recuperacion.coincidencias)} pasajes de apoyo.",
    ]
    if secciones_principales:
        lineas.append(f"Secciones principales: {secciones_principales}.")
    if sintesis:
        lineas.append(f"Sintesis: {sintesis}")
    return "\n".join(lineas)


def _formatear_referencias(
    evidencias: list[tuple[str, CoincidenciaParrafo, str, str]],
) -> str:
    lineas = ["", "Evidencias:"]
    for referencia, _, encabezado, fragmento in evidencias:
        lineas.append(f"- [{referencia}] {encabezado}: {fragmento}")
    return "\n".join(lineas)


def responder_rag(
    consulta: str,
    resultados_recuperacion: ResultadosBusqueda,
) -> str:
    if not resultados_recuperacion.coincidencias:
        return "No he recuperado contexto suficiente para responder con fiabilidad a esa consulta."

    consulta_tokens = set(procesar_texto_spacy(consulta))
    evidencias = _construir_bloque_evidencias(resultados_recuperacion, consulta_tokens)
    if not evidencias:
        return "No he recuperado contexto suficiente para responder con fiabilidad a esa consulta."

    prompt = _prompt_rag(consulta, evidencias)
    respuesta_llm = _generar_respuesta_ollama(prompt)
    if respuesta_llm:
        return respuesta_llm + _formatear_referencias(evidencias)

    respuesta_extractiva = _responder_extractivo(resultados_recuperacion, evidencias)
    aviso = (
        "\n\nAviso: no fue posible usar Ollama en este entorno. "
        "Se devuelve una respuesta extractiva local como fallback."
    )
    return respuesta_extractiva + aviso + _formatear_referencias(evidencias)

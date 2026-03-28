from __future__ import annotations

from fdi_pln_2612_p4.embeddings import construir_resultados_desde_coincidencias
from fdi_pln_2612_p4.modelos import CoincidenciaParrafo, CorpusQuijote, ResultadosBusqueda
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


def _clave_parrafo(coincidencia: CoincidenciaParrafo) -> tuple[int, int]:
    return coincidencia.parrafo.indice_seccion, coincidencia.parrafo.posicion_en_seccion


def _compactar_texto(texto: str, *, max_caracteres: int = MAX_CARACTERES_FRAGMENTO) -> str:
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

        for posicion, coincidencia in enumerate(coincidencias[:MAX_RESULTADOS_RAG], start=1):
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


def responder_rag(
    consulta: str,
    resultados_recuperacion: ResultadosBusqueda,
) -> str:
    if not resultados_recuperacion.coincidencias:
        return (
            "No he recuperado contexto suficiente para responder con fiabilidad a esa consulta."
        )

    consulta_tokens = set(procesar_texto_spacy(consulta))
    evidencias = [
        (
            coincidencia,
            _mejor_fragmento(coincidencia.parrafo.texto, consulta_tokens),
        )
        for coincidencia in resultados_recuperacion.coincidencias[:MAX_EVIDENCIAS_RAG]
    ]

    secciones_principales = ", ".join(
        list(
            dict.fromkeys(
                f"{resumir_parte(coincidencia.parrafo.titulo_parte)} | "
                f"{coincidencia.parrafo.titulo_seccion}"
                for coincidencia, _ in evidencias
            )
        )[:3]
    )
    sintesis = " ".join(fragmento for _, fragmento in evidencias[:3])

    lineas = [
        "Respuesta basada solo en los pasajes recuperados:",
        f"He reunido {len(resultados_recuperacion.coincidencias)} pasajes de apoyo.",
    ]
    if secciones_principales:
        lineas.append(f"Secciones principales: {secciones_principales}.")
    if sintesis:
        lineas.append(f"Síntesis: {sintesis}")

    lineas.append("")
    lineas.append("Evidencias:")
    for coincidencia, fragmento in evidencias:
        encabezado = (
            f"{resumir_parte(coincidencia.parrafo.titulo_parte)} | "
            f"{coincidencia.parrafo.titulo_seccion}"
        )
        lineas.append(f"- {encabezado}: {fragmento}")

    return "\n".join(lineas)

from __future__ import annotations

import math
from collections import Counter

from fdi_pln_2612_p4.ir_clasico import (
    similitud_coseno,
    vector_consulta as vector_consulta_tfidf,
)
from fdi_pln_2612_p4.modelos import (
    ChunkTexto,
    CoincidenciaParrafo,
    CorpusQuijote,
    Parrafo,
    ResultadosBusqueda,
    ResumenSeccion,
    Seccion,
)
from fdi_pln_2612_p4.nlp_utils import (
    factor_calidad_texto,
    normalizar_espacios,
    obtener_nlp,
    procesar_consulta_spacy,
)


TAMANO_LOTE_EMBEDDINGS = 64
MAX_RESULTADOS_SEMANTICOS = 50
MIN_RESULTADOS_SEMANTICOS = 5
UMBRAL_ABSOLUTO_SEMANTICO = 0.45
FACTOR_UMBRAL_SEMANTICO = 0.72
PESO_SEMANTICO_DENSO = 0.75
PESO_SEMANTICO_TFIDF = 0.25


def _iterar_unidades_semanticas(corpus: CorpusQuijote) -> list[ChunkTexto | Parrafo]:
    if corpus.chunks:
        return list(corpus.chunks)
    return [parrafo for seccion in corpus.secciones for parrafo in seccion.parrafos]


def _parrafo_representativo(
    corpus: CorpusQuijote, unidad: ChunkTexto | Parrafo
) -> Parrafo:
    if isinstance(unidad, Parrafo):
        return unidad

    seccion = corpus.secciones[unidad.indice_seccion]
    if not seccion.parrafos:
        raise RuntimeError(
            "Sección sin párrafos al buscar párrafo representativo de chunk."
        )

    inicio = min(max(unidad.parrafo_inicio, 0), len(seccion.parrafos) - 1)
    return seccion.parrafos[inicio]


def _texto_para_embedding_parrafo(parrafo: Parrafo) -> str:
    return " ".join(parrafo.lemas_normalizados) or normalizar_espacios(parrafo.texto)


def _texto_para_embedding_chunk(chunk: ChunkTexto) -> str:
    return " ".join(chunk.lemas_normalizados) or normalizar_espacios(chunk.texto)


def _texto_para_embedding_consulta(
    consulta_limpia: str, consulta_tokens: list[str]
) -> str:
    return " ".join(consulta_tokens) or consulta_limpia


def _vector_de_doc(doc) -> tuple[float, ...]:
    vector = tuple(float(valor) for valor in doc.vector)
    if vector:
        return vector

    if (
        getattr(doc, "tensor", None) is not None
        and getattr(doc.tensor, "shape", ())[:1]
    ):
        return tuple(float(valor) for valor in doc.tensor.mean(axis=0).tolist())

    raise RuntimeError(
        "El modelo de spaCy disponible no ofrece vectores densos utilizables "
        "para el modo semántico."
    )


def _vectorizar_texto(texto: str) -> tuple[float, ...]:
    return _vector_de_doc(obtener_nlp()(texto))


def similitud_coseno_densa(vec1: tuple[float, ...], vec2: tuple[float, ...]) -> float:
    if not vec1 or not vec2:
        return 0.0

    producto_punto = sum(valor_1 * valor_2 for valor_1, valor_2 in zip(vec1, vec2))
    if producto_punto == 0.0:
        return 0.0

    norma_1 = math.sqrt(sum(valor * valor for valor in vec1))
    norma_2 = math.sqrt(sum(valor * valor for valor in vec2))
    if norma_1 == 0.0 or norma_2 == 0.0:
        return 0.0

    return producto_punto / (norma_1 * norma_2)


def construir_resultados_desde_coincidencias(
    corpus: CorpusQuijote,
    consulta: str,
    consulta_normalizada: str,
    coincidencias: list[CoincidenciaParrafo],
    *,
    ignorar_tildes: bool,
) -> ResultadosBusqueda:
    coincidencias.sort(key=lambda item: item.score, reverse=True)

    apariciones_por_seccion: Counter[int] = Counter()
    parrafos_por_seccion: Counter[int] = Counter()
    for coincidencia in coincidencias:
        indice_seccion = coincidencia.parrafo.indice_seccion
        apariciones_por_seccion[indice_seccion] += 1
        parrafos_por_seccion[indice_seccion] += 1

    resumenes: list[ResumenSeccion] = []
    for indice_seccion, seccion in enumerate(corpus.secciones):
        apariciones = apariciones_por_seccion.get(indice_seccion, 0)
        if not apariciones:
            continue
        resumenes.append(
            ResumenSeccion(
                titulo=seccion.titulo,
                titulo_parte=seccion.titulo_parte,
                apariciones=apariciones,
                parrafos_con_coincidencias=parrafos_por_seccion[indice_seccion],
            )
        )

    return ResultadosBusqueda(
        consulta=consulta,
        consulta_normalizada=consulta_normalizada,
        ignorar_tildes=ignorar_tildes,
        total_apariciones=len(coincidencias),
        coincidencias=tuple(coincidencias),
        resumen_secciones=tuple(resumenes),
    )


def precalcular_embeddings(corpus: CorpusQuijote) -> CorpusQuijote:
    unidades = _iterar_unidades_semanticas(corpus)
    textos: list[str] = []
    for unidad in unidades:
        if isinstance(unidad, ChunkTexto):
            textos.append(_texto_para_embedding_chunk(unidad))
        else:
            textos.append(_texto_para_embedding_parrafo(unidad))

    if not textos:
        return corpus

    vectores = [
        _vector_de_doc(doc)
        for doc in obtener_nlp().pipe(textos, batch_size=TAMANO_LOTE_EMBEDDINGS)
    ]
    iterador_vectores = iter(vectores)

    secciones_actualizadas: list[Seccion] = []
    chunks_actualizados: list[ChunkTexto] = []
    for seccion in corpus.secciones:
        parrafos_actualizados: list[Parrafo] = []
        for parrafo in seccion.parrafos:
            vector_semantico = parrafo.vector_semantico
            if not corpus.chunks:
                vector_semantico = next(iterador_vectores)
            parrafos_actualizados.append(
                Parrafo(
                    texto=parrafo.texto,
                    titulo_seccion=parrafo.titulo_seccion,
                    titulo_parte=parrafo.titulo_parte,
                    indice_seccion=parrafo.indice_seccion,
                    posicion_en_seccion=parrafo.posicion_en_seccion,
                    indice_simple=parrafo.indice_simple,
                    indice_sin_tildes=parrafo.indice_sin_tildes,
                    lemas_normalizados=parrafo.lemas_normalizados,
                    vector_tfidf=parrafo.vector_tfidf,
                    vector_semantico=vector_semantico,
                )
            )
        secciones_actualizadas.append(
            Seccion(
                titulo=seccion.titulo,
                titulo_parte=seccion.titulo_parte,
                parrafos=tuple(parrafos_actualizados),
            )
        )

    for chunk in corpus.chunks:
        chunks_actualizados.append(
            ChunkTexto(
                id_chunk=chunk.id_chunk,
                texto=chunk.texto,
                lemas_normalizados=chunk.lemas_normalizados,
                indice_seccion=chunk.indice_seccion,
                titulo_seccion=chunk.titulo_seccion,
                titulo_parte=chunk.titulo_parte,
                parrafo_inicio=chunk.parrafo_inicio,
                parrafo_fin=chunk.parrafo_fin,
                vector_tfidf=chunk.vector_tfidf,
                vector_semantico=next(iterador_vectores),
            )
        )

    return CorpusQuijote(
        ruta_fuente=corpus.ruta_fuente,
        secciones=tuple(secciones_actualizadas),
        chunks=tuple(chunks_actualizados),
        vocabulario=corpus.vocabulario,
        idf=corpus.idf,
    )


def buscar_en_corpus_semantico(
    corpus: CorpusQuijote,
    consulta: str,
    *,
    ignorar_tildes: bool = True,
) -> ResultadosBusqueda:
    consulta_limpia = normalizar_espacios(consulta)
    if not consulta_limpia:
        raise ValueError("La consulta no puede estar vacía.")

    consulta_tokens = procesar_consulta_spacy(
        consulta_limpia,
        ignorar_tildes=ignorar_tildes,
        idf=corpus.idf,
    )
    if not consulta_tokens:
        raise ValueError("La consulta se ha quedado vacía tras eliminar stopwords.")

    unidades = _iterar_unidades_semanticas(corpus)
    if not unidades:
        return ResultadosBusqueda(
            consulta=consulta_limpia,
            consulta_normalizada=" ".join(consulta_tokens),
            ignorar_tildes=ignorar_tildes,
            total_apariciones=0,
            coincidencias=(),
            resumen_secciones=(),
        )

    if not unidades[0].vector_semantico:
        raise RuntimeError(
            "El corpus no tiene embeddings precalculados. "
            "Ejecuta primero la preparación semántica del corpus."
        )

    consulta_normalizada = " ".join(consulta_tokens)
    vector_consulta_semantico = _vectorizar_texto(
        _texto_para_embedding_consulta(consulta_limpia, consulta_tokens)
    )
    vector_consulta_clasico = vector_consulta_tfidf(
        consulta_limpia,
        corpus.idf,
        ignorar_tildes=ignorar_tildes,
    )

    puntuaciones: list[tuple[ChunkTexto | Parrafo, float, float, float]] = []
    for unidad in unidades:
        score_denso = similitud_coseno_densa(
            vector_consulta_semantico, unidad.vector_semantico
        )
        score_tfidf = similitud_coseno(vector_consulta_clasico, unidad.vector_tfidf)
        factor_calidad = factor_calidad_texto(unidad.texto)
        if score_denso <= 0.0 and score_tfidf <= 0.0:
            continue
        if score_tfidf <= 0.0 and factor_calidad < 0.7:
            continue
        puntuaciones.append((unidad, score_denso, score_tfidf, factor_calidad))

    if not puntuaciones:
        return ResultadosBusqueda(
            consulta=consulta_limpia,
            consulta_normalizada=consulta_normalizada,
            ignorar_tildes=ignorar_tildes,
            total_apariciones=0,
            coincidencias=(),
            resumen_secciones=(),
        )

    mejor_score_denso = max(score_denso for _, score_denso, _, _ in puntuaciones)
    mejor_score_tfidf = max(score_tfidf for _, _, score_tfidf, _ in puntuaciones)

    mejores_por_parrafo: dict[tuple[int, int], CoincidenciaParrafo] = {}
    for unidad, score_denso, score_tfidf, factor_calidad in puntuaciones:
        score_hibrido = (
            PESO_SEMANTICO_DENSO
            * (score_denso / mejor_score_denso if mejor_score_denso else 0.0)
            + PESO_SEMANTICO_TFIDF
            * (score_tfidf / mejor_score_tfidf if mejor_score_tfidf else 0.0)
        ) * factor_calidad

        parrafo_ref = _parrafo_representativo(corpus, unidad)
        clave = (parrafo_ref.indice_seccion, parrafo_ref.posicion_en_seccion)
        anterior = mejores_por_parrafo.get(clave)
        if anterior is None or score_hibrido > anterior.score:
            mejores_por_parrafo[clave] = CoincidenciaParrafo(
                parrafo=parrafo_ref,
                spans=(),
                score=score_hibrido,
            )

    coincidencias = list(mejores_por_parrafo.values())
    coincidencias.sort(key=lambda item: item.score, reverse=True)
    mejor_score = coincidencias[0].score
    umbral = max(UMBRAL_ABSOLUTO_SEMANTICO, mejor_score * FACTOR_UMBRAL_SEMANTICO)
    relevantes = [item for item in coincidencias if item.score >= umbral][
        :MAX_RESULTADOS_SEMANTICOS
    ]

    if len(relevantes) < MIN_RESULTADOS_SEMANTICOS:
        relevantes = coincidencias[: min(MIN_RESULTADOS_SEMANTICOS, len(coincidencias))]

    return construir_resultados_desde_coincidencias(
        corpus,
        consulta_limpia,
        consulta_normalizada,
        relevantes,
        ignorar_tildes=ignorar_tildes,
    )

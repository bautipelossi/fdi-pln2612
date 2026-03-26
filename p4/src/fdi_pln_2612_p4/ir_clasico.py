from __future__ import annotations

from collections import Counter
import math

from fdi_pln_2612_p4.modelos import CoincidenciaParrafo, CorpusQuijote, Parrafo, ResultadosBusqueda, ResumenSeccion, Seccion
from fdi_pln_2612_p4.nlp_utils import normalizar_espacios, procesar_texto_spacy


def _terminos_con_n_gramas(lemas: tuple[str, ...] | list[str]) -> Counter[str]:
    contador: Counter[str] = Counter(lemas)

    for indice in range(len(lemas) - 1):
        bigrama = f"{lemas[indice]}_{lemas[indice + 1]}"
        contador[bigrama] += 2

    for indice in range(len(lemas) - 2):
        trigrama = f"{lemas[indice]}_{lemas[indice + 1]}_{lemas[indice + 2]}"
        contador[trigrama] += 3

    return contador


def construir_vocabulario(corpus: CorpusQuijote) -> tuple[str, ...]:
    vocabulario: set[str] = set()
    for seccion in corpus.secciones:
        for parrafo in seccion.parrafos:
            vocabulario.update(_terminos_con_n_gramas(parrafo.lemas_normalizados).keys())
    return tuple(sorted(vocabulario))


def calcular_tf(parrafo: Parrafo) -> dict[str, float]:
    contador = _terminos_con_n_gramas(parrafo.lemas_normalizados)
    total = sum(contador.values())
    if total == 0:
        return {}
    return {termino: frecuencia / total for termino, frecuencia in contador.items()}


def calcular_idf(corpus: CorpusQuijote) -> dict[str, float]:
    total_documentos = corpus.total_parrafos
    if total_documentos == 0:
        return {}

    frecuencia_documental: Counter[str] = Counter()
    for seccion in corpus.secciones:
        for parrafo in seccion.parrafos:
            frecuencia_documental.update(set(_terminos_con_n_gramas(parrafo.lemas_normalizados).keys()))

    return {
        termino: math.log(total_documentos / documentos_con_termino)
        for termino, documentos_con_termino in frecuencia_documental.items()
    }


def construir_vector_tfidf(parrafo: Parrafo, idf: dict[str, float]) -> dict[str, float]:
    tf = calcular_tf(parrafo)
    return {
        termino: frecuencia * idf[termino]
        for termino, frecuencia in tf.items()
        if termino in idf
    }


def vector_consulta(
    consulta: str,
    idf: dict[str, float],
    *,
    ignorar_tildes: bool = True,
) -> dict[str, float]:
    lemas = tuple(procesar_texto_spacy(consulta, ignorar_tildes=ignorar_tildes))
    if not lemas:
        return {}

    contador = _terminos_con_n_gramas(lemas)
    total = sum(contador.values())
    tf = {termino: frecuencia / total for termino, frecuencia in contador.items()}
    return {termino: frecuencia * idf[termino] for termino, frecuencia in tf.items() if termino in idf}


def similitud_coseno(vec1: dict[str, float], vec2: dict[str, float]) -> float:
    if not vec1 or not vec2:
        return 0.0

    if len(vec1) > len(vec2):
        vec1, vec2 = vec2, vec1

    producto_punto = sum(valor * vec2.get(termino, 0.0) for termino, valor in vec1.items())
    if producto_punto == 0.0:
        return 0.0

    norma_1 = math.sqrt(sum(valor * valor for valor in vec1.values()))
    norma_2 = math.sqrt(sum(valor * valor for valor in vec2.values()))
    if norma_1 == 0.0 or norma_2 == 0.0:
        return 0.0

    return producto_punto / (norma_1 * norma_2)


def precalcular_tfidf(corpus: CorpusQuijote) -> CorpusQuijote:
    idf = calcular_idf(corpus)
    vocabulario = construir_vocabulario(corpus)
    secciones_actualizadas = []

    for seccion in corpus.secciones:
        parrafos_actualizados = []
        for parrafo in seccion.parrafos:
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
                    vector_tfidf=construir_vector_tfidf(parrafo, idf),
                )
            )
        secciones_actualizadas.append(
            Seccion(
                titulo=seccion.titulo,
                titulo_parte=seccion.titulo_parte,
                parrafos=tuple(parrafos_actualizados),
            )
        )

    return CorpusQuijote(
        ruta_fuente=corpus.ruta_fuente,
        secciones=tuple(secciones_actualizadas),
        vocabulario=vocabulario,
        idf=idf,
    )


def buscar_en_corpus(
    corpus: CorpusQuijote,
    consulta: str,
    *,
    ignorar_tildes: bool = True,
) -> ResultadosBusqueda:
    consulta_limpia = normalizar_espacios(consulta)
    if not consulta_limpia:
        raise ValueError("La consulta no puede estar vacía.")

    consulta_tokens = procesar_texto_spacy(consulta_limpia, ignorar_tildes=ignorar_tildes)
    if not consulta_tokens:
        raise ValueError("La consulta se ha quedado vacía tras eliminar stopwords.")

    consulta_normalizada = " ".join(consulta_tokens)
    vec_consulta = vector_consulta(consulta_limpia, corpus.idf, ignorar_tildes=ignorar_tildes)

    coincidencias: list[CoincidenciaParrafo] = []
    resumenes: list[ResumenSeccion] = []
    total_apariciones = 0

    if not vec_consulta:
        return ResultadosBusqueda(
            consulta=consulta_limpia,
            consulta_normalizada=consulta_normalizada,
            ignorar_tildes=ignorar_tildes,
            total_apariciones=0,
            coincidencias=(),
            resumen_secciones=(),
        )

    for seccion in corpus.secciones:
        apariciones_seccion = 0
        parrafos_con_coincidencias = 0

        for parrafo in seccion.parrafos:
            score = similitud_coseno(vec_consulta, parrafo.vector_tfidf)
            if score <= 0:
                continue

            apariciones_seccion += 1
            total_apariciones += 1
            parrafos_con_coincidencias += 1
            coincidencias.append(CoincidenciaParrafo(parrafo=parrafo, spans=(), score=score))

        if apariciones_seccion:
            resumenes.append(
                ResumenSeccion(
                    titulo=seccion.titulo,
                    titulo_parte=seccion.titulo_parte,
                    apariciones=apariciones_seccion,
                    parrafos_con_coincidencias=parrafos_con_coincidencias,
                )
            )

    coincidencias.sort(key=lambda item: item.score, reverse=True)

    return ResultadosBusqueda(
        consulta=consulta_limpia,
        consulta_normalizada=consulta_normalizada,
        ignorar_tildes=ignorar_tildes,
        total_apariciones=total_apariciones,
        coincidencias=tuple(coincidencias),
        resumen_secciones=tuple(resumenes),
    )

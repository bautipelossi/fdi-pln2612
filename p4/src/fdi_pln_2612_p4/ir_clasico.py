from __future__ import annotations

from collections import Counter
import math

from fdi_pln_2612_p4.modelos import (
    ChunkTexto,
    CoincidenciaParrafo,
    CorpusQuijote,
    Parrafo,
    ResultadosBusqueda,
    ResumenSeccion,
    Seccion,
)
from fdi_pln_2612_p4.nlp_utils import normalizar_espacios, procesar_consulta_spacy


def _terminos_con_n_gramas(lemas: tuple[str, ...] | list[str]) -> Counter[str]:
    contador: Counter[str] = Counter(lemas)

    for indice in range(len(lemas) - 1):
        bigrama = f"{lemas[indice]}_{lemas[indice + 1]}"
        contador[bigrama] += 2

    for indice in range(len(lemas) - 2):
        trigrama = f"{lemas[indice]}_{lemas[indice + 1]}_{lemas[indice + 2]}"
        contador[trigrama] += 3

    return contador


def _iterar_unidades(corpus: CorpusQuijote) -> list[ChunkTexto | Parrafo]:
    if corpus.chunks:
        return list(corpus.chunks)
    return [parrafo for seccion in corpus.secciones for parrafo in seccion.parrafos]


def _parrafo_representativo(
    corpus: CorpusQuijote,
    unidad: ChunkTexto | Parrafo,
    vec_consulta: dict[str, float],
) -> Parrafo:
    if isinstance(unidad, Parrafo):
        return unidad

    seccion = corpus.secciones[unidad.indice_seccion]
    if not seccion.parrafos:
        raise RuntimeError(
            "Sección sin párrafos al buscar párrafo representativo de chunk."
        )

    inicio = min(max(unidad.parrafo_inicio, 0), len(seccion.parrafos) - 1)
    fin = min(max(unidad.parrafo_fin + 1, inicio + 1), len(seccion.parrafos))

    mejor_parrafo = seccion.parrafos[inicio]
    mejor_score = similitud_coseno(vec_consulta, mejor_parrafo.vector_tfidf)
    for parrafo in seccion.parrafos[inicio:fin]:
        score = similitud_coseno(vec_consulta, parrafo.vector_tfidf)
        if score > mejor_score:
            mejor_parrafo = parrafo
            mejor_score = score

    return mejor_parrafo


def construir_vocabulario(corpus: CorpusQuijote) -> tuple[str, ...]:
    vocabulario: set[str] = set()
    for unidad in _iterar_unidades(corpus):
        vocabulario.update(_terminos_con_n_gramas(unidad.lemas_normalizados).keys())
    return tuple(sorted(vocabulario))


def calcular_tf(lemas_normalizados: tuple[str, ...] | list[str]) -> dict[str, float]:
    contador = _terminos_con_n_gramas(lemas_normalizados)
    total = sum(contador.values())
    if total == 0:
        return {}
    return {termino: frecuencia / total for termino, frecuencia in contador.items()}


def calcular_idf(corpus: CorpusQuijote) -> dict[str, float]:
    unidades = _iterar_unidades(corpus)
    total_documentos = len(unidades)
    if total_documentos == 0:
        return {}

    frecuencia_documental: Counter[str] = Counter()
    for unidad in unidades:
        frecuencia_documental.update(
            set(_terminos_con_n_gramas(unidad.lemas_normalizados).keys())
        )

    return {
        termino: math.log(total_documentos / documentos_con_termino)
        for termino, documentos_con_termino in frecuencia_documental.items()
    }


def construir_vector_tfidf(
    lemas_normalizados: tuple[str, ...], idf: dict[str, float]
) -> dict[str, float]:
    tf = calcular_tf(lemas_normalizados)
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
    lemas = tuple(
        procesar_consulta_spacy(
            consulta,
            ignorar_tildes=ignorar_tildes,
            idf=idf,
        )
    )
    if not lemas:
        return {}

    contador = _terminos_con_n_gramas(lemas)
    total = sum(contador.values())
    tf = {termino: frecuencia / total for termino, frecuencia in contador.items()}
    return {
        termino: frecuencia * idf[termino]
        for termino, frecuencia in tf.items()
        if termino in idf
    }


def similitud_coseno(vec1: dict[str, float], vec2: dict[str, float]) -> float:
    if not vec1 or not vec2:
        return 0.0

    if len(vec1) > len(vec2):
        vec1, vec2 = vec2, vec1

    producto_punto = sum(
        valor * vec2.get(termino, 0.0) for termino, valor in vec1.items()
    )
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
                    vector_tfidf=construir_vector_tfidf(
                        parrafo.lemas_normalizados, idf
                    ),
                    vector_semantico=parrafo.vector_semantico,
                )
            )
        secciones_actualizadas.append(
            Seccion(
                titulo=seccion.titulo,
                titulo_parte=seccion.titulo_parte,
                parrafos=tuple(parrafos_actualizados),
            )
        )

    chunks_actualizados: list[ChunkTexto] = []
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
                vector_tfidf=construir_vector_tfidf(chunk.lemas_normalizados, idf),
                vector_semantico=chunk.vector_semantico,
            )
        )

    return CorpusQuijote(
        ruta_fuente=corpus.ruta_fuente,
        secciones=tuple(secciones_actualizadas),
        chunks=tuple(chunks_actualizados),
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

    consulta_tokens = procesar_consulta_spacy(
        consulta_limpia,
        ignorar_tildes=ignorar_tildes,
        idf=corpus.idf,
    )
    if not consulta_tokens:
        raise ValueError("La consulta se ha quedado vacía tras eliminar stopwords.")

    consulta_normalizada = " ".join(consulta_tokens)
    vec_consulta = vector_consulta(
        consulta_limpia, corpus.idf, ignorar_tildes=ignorar_tildes
    )

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

    mejores_por_parrafo: dict[tuple[int, int], CoincidenciaParrafo] = {}
    for unidad in _iterar_unidades(corpus):
        score_unidad = similitud_coseno(vec_consulta, unidad.vector_tfidf)
        if score_unidad <= 0:
            continue

        parrafo_ref = _parrafo_representativo(corpus, unidad, vec_consulta)
        score_parrafo = similitud_coseno(vec_consulta, parrafo_ref.vector_tfidf)
        score_final = score_parrafo if score_parrafo > 0 else score_unidad
        clave = (parrafo_ref.indice_seccion, parrafo_ref.posicion_en_seccion)
        anterior = mejores_por_parrafo.get(clave)
        if anterior is None or score_final > anterior.score:
            mejores_por_parrafo[clave] = CoincidenciaParrafo(
                parrafo=parrafo_ref,
                spans=(),
                score=score_final,
            )

    coincidencias = list(mejores_por_parrafo.values())
    total_apariciones = len(coincidencias)

    for indice_seccion, seccion in enumerate(corpus.secciones):
        coincidencias_seccion = [
            item
            for item in coincidencias
            if item.parrafo.indice_seccion == indice_seccion
        ]
        if not coincidencias_seccion:
            continue
        resumenes.append(
            ResumenSeccion(
                titulo=seccion.titulo,
                titulo_parte=seccion.titulo_parte,
                apariciones=len(coincidencias_seccion),
                parrafos_con_coincidencias=len(coincidencias_seccion),
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

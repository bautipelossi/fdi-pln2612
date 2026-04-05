from __future__ import annotations

from html import unescape
from pathlib import Path
from xml.etree import ElementTree as ET

from fdi_pln_2612_p4.modelos import ChunkTexto, CorpusQuijote, Parrafo, Seccion
from fdi_pln_2612_p4.nlp_utils import (
    MARCAS_PARTE,
    construir_indice,
    normalizar_espacios,
    procesar_texto_spacy,
)


TAMANO_CHUNK_TOKENS = 140
OVERLAP_CHUNK_TOKENS = 40


def _es_marca_de_parte(texto: str) -> bool:
    return any(texto.startswith(marca) for marca in MARCAS_PARTE)


def _texto_de_elemento(elemento: ET.Element) -> str:
    return normalizar_espacios("".join(elemento.itertext()))


def construir_chunks_con_overlap(
    secciones: tuple[Seccion, ...],
    *,
    tamano_tokens: int = TAMANO_CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_CHUNK_TOKENS,
) -> tuple[ChunkTexto, ...]:
    if tamano_tokens <= 0:
        raise ValueError("El tamaño de chunk debe ser mayor que cero.")
    if overlap_tokens < 0:
        raise ValueError("El overlap no puede ser negativo.")
    if overlap_tokens >= tamano_tokens:
        raise ValueError("El overlap debe ser menor que el tamaño de chunk.")

    chunks: list[ChunkTexto] = []
    siguiente_id = 1

    for seccion in secciones:
        if not seccion.parrafos:
            continue

        inicio = 0
        while inicio < len(seccion.parrafos):
            lemas_chunk: list[str] = []
            textos_chunk: list[str] = []
            fin = inicio

            while fin < len(seccion.parrafos):
                parrafo = seccion.parrafos[fin]
                lemas_parrafo = list(parrafo.lemas_normalizados)

                if (
                    lemas_chunk
                    and len(lemas_chunk) + len(lemas_parrafo) > tamano_tokens
                ):
                    break

                if not lemas_chunk and len(lemas_parrafo) > tamano_tokens:
                    lemas_chunk = lemas_parrafo[:tamano_tokens]
                    textos_chunk.append(parrafo.texto)
                    fin += 1
                    break

                lemas_chunk.extend(lemas_parrafo)
                textos_chunk.append(parrafo.texto)
                fin += 1

                if len(lemas_chunk) >= tamano_tokens:
                    break

            if not textos_chunk:
                break

            texto_chunk = normalizar_espacios(" ".join(textos_chunk))
            chunks.append(
                ChunkTexto(
                    id_chunk=siguiente_id,
                    texto=texto_chunk,
                    lemas_normalizados=tuple(lemas_chunk),
                    indice_seccion=seccion.parrafos[inicio].indice_seccion,
                    titulo_seccion=seccion.titulo,
                    titulo_parte=seccion.titulo_parte,
                    parrafo_inicio=seccion.parrafos[inicio].posicion_en_seccion,
                    parrafo_fin=seccion.parrafos[fin - 1].posicion_en_seccion,
                )
            )
            siguiente_id += 1

            if fin >= len(seccion.parrafos):
                break

            tokens_a_conservar = overlap_tokens
            nuevo_inicio = fin - 1
            while nuevo_inicio > inicio and tokens_a_conservar > 0:
                tokens_a_conservar -= len(
                    seccion.parrafos[nuevo_inicio].lemas_normalizados
                )
                nuevo_inicio -= 1

            inicio = max(inicio + 1, nuevo_inicio + 1)

    return tuple(chunks)


def cargar_corpus_html(ruta_html: str | Path) -> CorpusQuijote:
    ruta = Path(ruta_html)
    contenido = unescape(ruta.read_text(encoding="utf-8"))
    raiz = ET.fromstring(contenido)
    espacio_nombres = {"xhtml": "http://www.w3.org/1999/xhtml"}
    cuerpo = raiz.find("xhtml:body", espacio_nombres)

    if cuerpo is None:
        raise ValueError("No se ha encontrado el cuerpo del documento HTML.")

    secciones_crudas: list[tuple[str, str | None, list[str]]] = []
    titulo_parte_actual: str | None = None
    titulo_parte_seccion: str | None = None
    titulo_seccion_actual: str | None = None
    parrafos_actuales: list[str] = []

    def cerrar_seccion() -> None:
        nonlocal titulo_parte_seccion, titulo_seccion_actual, parrafos_actuales
        if titulo_seccion_actual and parrafos_actuales:
            secciones_crudas.append(
                (titulo_seccion_actual, titulo_parte_seccion, parrafos_actuales[:])
            )
        titulo_seccion_actual = None
        titulo_parte_seccion = None
        parrafos_actuales = []

    for elemento in cuerpo.iter():
        etiqueta = elemento.tag.split("}")[-1]
        if etiqueta not in {"h3", "p"}:
            continue

        texto = _texto_de_elemento(elemento)
        if not texto:
            continue

        if etiqueta == "p" and _es_marca_de_parte(texto):
            cerrar_seccion()
            titulo_parte_actual = texto
            continue

        if etiqueta == "h3":
            cerrar_seccion()
            titulo_seccion_actual = texto
            titulo_parte_seccion = titulo_parte_actual
            continue

        if etiqueta == "p" and titulo_seccion_actual:
            parrafos_actuales.append(texto)

    cerrar_seccion()

    secciones: list[Seccion] = []
    for indice_seccion, (titulo, titulo_parte, textos_parrafos) in enumerate(
        secciones_crudas
    ):
        parrafos: list[Parrafo] = []
        for posicion_en_seccion, texto_parrafo in enumerate(textos_parrafos):
            lemas = tuple(procesar_texto_spacy(texto_parrafo))
            parrafos.append(
                Parrafo(
                    texto=texto_parrafo,
                    titulo_seccion=titulo,
                    titulo_parte=titulo_parte,
                    indice_seccion=indice_seccion,
                    posicion_en_seccion=posicion_en_seccion,
                    indice_simple=construir_indice(texto_parrafo, quitar=False),
                    indice_sin_tildes=construir_indice(texto_parrafo, quitar=True),
                    lemas_normalizados=lemas,
                    vector_tfidf={},
                )
            )
        secciones.append(
            Seccion(
                titulo=titulo,
                titulo_parte=titulo_parte,
                parrafos=tuple(parrafos),
            )
        )

    secciones_finales = tuple(secciones)
    chunks = construir_chunks_con_overlap(secciones_finales)
    return CorpusQuijote(ruta_fuente=ruta, secciones=secciones_finales, chunks=chunks)

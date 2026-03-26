from __future__ import annotations

from html import unescape
from pathlib import Path
from xml.etree import ElementTree as ET

from fdi_pln_2612_p4.modelos import CorpusQuijote, Parrafo, Seccion
from fdi_pln_2612_p4.nlp_utils import MARCAS_PARTE, construir_indice, normalizar_espacios, procesar_texto_spacy


def _es_marca_de_parte(texto: str) -> bool:
    return any(texto.startswith(marca) for marca in MARCAS_PARTE)


def _texto_de_elemento(elemento: ET.Element) -> str:
    return normalizar_espacios("".join(elemento.itertext()))


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
    for indice_seccion, (titulo, titulo_parte, textos_parrafos) in enumerate(secciones_crudas):
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

    return CorpusQuijote(ruta_fuente=ruta, secciones=tuple(secciones))

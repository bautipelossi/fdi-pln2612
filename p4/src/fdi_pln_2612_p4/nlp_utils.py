from __future__ import annotations

import importlib
import re
import unicodedata

from fdi_pln_2612_p4.modelos import IndiceTexto


ESPACIOS_RE = re.compile(r"\s+")
PALABRA_RE = re.compile(r"\w", re.UNICODE)
FRASE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÜÑ¿¡\"«—])")
MARCAS_PARTE = (
    "Primera parte del ingenioso hidalgo don Quijote de la Mancha",
    "Segunda parte del ingenioso hidalgo don Quijote de la Mancha",
    "Segunda parte del ingenioso caballero don Quijote de la Mancha",
)
MODO_ETIQUETAS = {
    "frase": "Coincidencias por frase",
    "parrafo": "Coincidencias por párrafo",
    "contexto": "Párrafo con contexto",
    "conteo": "Solo número total de apariciones",
    "seccion": "Resumen por capítulo o sección",
}
PATRONES_CONSULTA = (
    re.compile(r"^\s*[¿?]?\s*(?:donde|dónde)\s+aparece(?:n)?\s+", re.IGNORECASE),
    re.compile(r"^\s*(?:buscar|busca|búscame)\s+", re.IGNORECASE),
    re.compile(r"^\s*(?:apariciones?|ocurrencias?)\s+de\s+", re.IGNORECASE),
    re.compile(r"^\s*(?:quiero\s+buscar|mu[eé]strame)\s+", re.IGNORECASE),
)

_NLP = None


def normalizar_espacios(texto: str) -> str:
    return ESPACIOS_RE.sub(" ", texto).strip()


def quitar_tildes(texto: str) -> str:
    return "".join(
        signo
        for signo in unicodedata.normalize("NFD", texto)
        if unicodedata.category(signo) != "Mn"
    )


def obtener_nlp():
    global _NLP
    if _NLP is None:
        try:
            spacy = importlib.import_module("spacy")
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "spaCy no está disponible en este entorno. "
                "Instala las dependencias declaradas en pyproject.toml usando uv sync."
            ) from error

        try:
            _NLP = spacy.load("es_core_news_sm", disable=["parser", "ner", "textcat"])
        except OSError as error:
            raise RuntimeError(
                "No se pudo cargar el modelo 'es_core_news_sm'. "
                "El modelo debe estar declarado en pyproject.toml e instalado con uv sync."
            ) from error
    return _NLP


def procesar_texto_spacy(texto: str, *, ignorar_tildes: bool = True) -> list[str]:
    doc = obtener_nlp()(texto)
    lemas: list[str] = []

    for token in doc:
        if token.is_space or token.is_punct or token.is_stop:
            continue

        lema = token.lemma_.lower().strip()
        if not lema or lema == "-pron-":
            lema = token.lower_
        if ignorar_tildes:
            lema = quitar_tildes(lema)
        if not PALABRA_RE.search(lema):
            continue
        lemas.append(lema)

    return lemas


def construir_indice(texto: str, *, quitar: bool) -> IndiceTexto:
    salida: list[str] = []
    mapa: list[int] = []

    for posicion, caracter in enumerate(texto):
        fragmento = caracter.lower()
        if quitar:
            fragmento = quitar_tildes(fragmento)

        for signo in fragmento:
            salida.append(signo)
            mapa.append(posicion)

    return IndiceTexto("".join(salida), tuple(mapa))


def extraer_consulta(texto: str) -> str:
    consulta = " ".join(texto.strip().split())
    while True:
        original = consulta
        for patron in PATRONES_CONSULTA:
            consulta = patron.sub("", consulta)
        if consulta == original:
            break
    return consulta.strip(" \"'«»?!.")


def fragmentar_en_frases(texto: str) -> list[tuple[int, int]]:
    if not texto:
        return []

    spans: list[tuple[int, int]] = []
    inicio = 0

    for coincidencia in FRASE_RE.finditer(texto):
        fin = coincidencia.start()
        if texto[inicio:fin].strip():
            spans.append((inicio, fin))
        inicio = coincidencia.end()

    if texto[inicio:].strip():
        spans.append((inicio, len(texto)))

    return spans


def resumir_parte(titulo_parte: str | None) -> str:
    if not titulo_parte:
        return "Preliminares"
    if titulo_parte.lower().startswith("primera parte"):
        return "Primera parte"
    if titulo_parte.lower().startswith("segunda parte"):
        return "Segunda parte"
    return titulo_parte

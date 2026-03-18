from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from html import unescape
import importlib
import math
from pathlib import Path
import re
import shutil
import textwrap
import unicodedata
from xml.etree import ElementTree as ET


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
RUTA_BASE = Path(__file__).resolve().parent
RUTA_HTML = RUTA_BASE / "2000-h.htm"
_RICH_CONSOLE = None


@dataclass(frozen=True)
class IndiceTexto:
    clave: str
    mapa: tuple[int, ...]


@dataclass(frozen=True)
class Parrafo:
    texto: str
    titulo_seccion: str
    titulo_parte: str | None
    indice_seccion: int
    posicion_en_seccion: int
    indice_simple: IndiceTexto
    indice_sin_tildes: IndiceTexto
    lemas_normalizados: tuple[str, ...]
    vector_tfidf: dict[str, float]


@dataclass(frozen=True)
class Seccion:
    titulo: str
    titulo_parte: str | None
    parrafos: tuple[Parrafo, ...]


@dataclass(frozen=True)
class CorpusQuijote:
    ruta_fuente: Path
    secciones: tuple[Seccion, ...]
    vocabulario: tuple[str, ...] = ()
    idf: dict[str, float] = field(default_factory=dict)

    @property
    def total_parrafos(self) -> int:
        return sum(len(seccion.parrafos) for seccion in self.secciones)


@dataclass(frozen=True)
class CoincidenciaParrafo:
    parrafo: Parrafo
    spans: tuple[tuple[int, int], ...]
    score: float


@dataclass(frozen=True)
class ResumenSeccion:
    titulo: str
    titulo_parte: str | None
    apariciones: int
    parrafos_con_coincidencias: int


@dataclass(frozen=True)
class ResultadosBusqueda:
    consulta: str
    consulta_normalizada: str
    ignorar_tildes: bool
    total_apariciones: int
    coincidencias: tuple[CoincidenciaParrafo, ...]
    resumen_secciones: tuple[ResumenSeccion, ...]


@dataclass
class ConfiguracionConsola:
    modo_salida: str = "contexto"
    ignorar_tildes: bool = True
    limite_resultados: int = 5
    contexto_parrafos: int = 1


def normalizar_espacios(texto: str) -> str:
    return ESPACIOS_RE.sub(" ", texto).strip()


def _obtener_consola_rich():
    global _RICH_CONSOLE
    if _RICH_CONSOLE is not None:
        return _RICH_CONSOLE

    try:
        rich_console = importlib.import_module("rich.console")
    except ModuleNotFoundError:
        return None

    _RICH_CONSOLE = rich_console.Console()
    return _RICH_CONSOLE


def ui_print(texto: str = "", *, style: str | None = None) -> None:
    consola = _obtener_consola_rich()
    if consola is None:
        print(texto)
        return
    if style:
        consola.print(texto, style=style)
    else:
        consola.print(texto)


def ui_panel(texto: str, *, titulo: str, style: str = "cyan") -> None:
    consola = _obtener_consola_rich()
    if consola is None:
        print()
        print(titulo)
        print("=" * min(len(titulo), ancho_terminal()))
        print(texto)
        return

    rich_panel = importlib.import_module("rich.panel")
    panel = rich_panel.Panel.fit(texto, title=titulo, border_style=style)
    consola.print(panel)


def ui_tabla_resultados(resultados: ResultadosBusqueda, *, limite: int) -> None:
    if not resultados.coincidencias:
        return

    consola = _obtener_consola_rich()
    if consola is None:
        return

    rich_table = importlib.import_module("rich.table")
    tabla = rich_table.Table(title="Top resultados", show_lines=False)
    tabla.add_column("#", justify="right", style="bold")
    tabla.add_column("Score", justify="right", style="green")
    tabla.add_column("Sección", style="cyan")
    tabla.add_column("Fragmento", style="white")

    for indice, coincidencia in enumerate(resultados.coincidencias[:limite], start=1):
        fragmento, _ = recortar_fragmento(coincidencia.parrafo.texto, max_caracteres=120)
        tabla.add_row(
            str(indice),
            f"{coincidencia.score:.4f}",
            encabezado_seccion(coincidencia.parrafo),
            fragmento,
        )

    consola.print(tabla)


def _quitar_tildes(texto: str) -> str:
    return "".join(
        signo
        for signo in unicodedata.normalize("NFD", texto)
        if unicodedata.category(signo) != "Mn"
    )


_NLP = None


def _obtener_nlp():
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
    doc = _obtener_nlp()(texto)
    lemas: list[str] = []

    for token in doc:
        if token.is_space or token.is_punct or token.is_stop:
            continue

        lema = token.lemma_.lower().strip()
        if not lema or lema == "-pron-":
            lema = token.lower_
        if ignorar_tildes:
            lema = _quitar_tildes(lema)
        if not PALABRA_RE.search(lema):
            continue
        lemas.append(lema)

    return lemas


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
    secciones_actualizadas: list[Seccion] = []

    for seccion in corpus.secciones:
        parrafos_actualizados: list[Parrafo] = []
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


def construir_indice(texto: str, *, quitar_tildes: bool) -> IndiceTexto:
    salida: list[str] = []
    mapa: list[int] = []

    for posicion, caracter in enumerate(texto):
        fragmento = caracter.lower()
        if quitar_tildes:
            fragmento = _quitar_tildes(fragmento)

        for signo in fragmento:
            salida.append(signo)
            mapa.append(posicion)

    return IndiceTexto("".join(salida), tuple(mapa))


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
                    indice_simple=construir_indice(texto_parrafo, quitar_tildes=False),
                    indice_sin_tildes=construir_indice(texto_parrafo, quitar_tildes=True),
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


def _crear_patron(consulta_normalizada: str) -> re.Pattern[str]:
    expresion = re.escape(consulta_normalizada)
    if PALABRA_RE.search(consulta_normalizada):
        return re.compile(rf"(?<!\w){expresion}(?!\w)")
    return re.compile(expresion)


def _buscar_spans(indice: IndiceTexto, patron: re.Pattern[str]) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []

    for coincidencia in patron.finditer(indice.clave):
        inicio_normalizado, fin_normalizado = coincidencia.span()
        inicio_original = indice.mapa[inicio_normalizado]
        fin_original = indice.mapa[fin_normalizado - 1] + 1
        spans.append((inicio_original, fin_original))

    return tuple(spans)


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


def extraer_consulta(texto: str) -> str:
    consulta = " ".join(texto.strip().split())
    while True:
        original = consulta
        for patron in PATRONES_CONSULTA:
            consulta = patron.sub("", consulta)
        if consulta == original:
            break
    return consulta.strip(" \"'«»?!.")


def resumir_parte(titulo_parte: str | None) -> str:
    if not titulo_parte:
        return "Preliminares"
    if titulo_parte.lower().startswith("primera parte"):
        return "Primera parte"
    if titulo_parte.lower().startswith("segunda parte"):
        return "Segunda parte"
    return titulo_parte


def ancho_terminal() -> int:
    return min(100, shutil.get_terminal_size((100, 24)).columns)


def imprimir_titulo(texto: str) -> None:
    ui_print()
    ui_print(texto, style="bold cyan")
    ui_print("=" * min(len(texto), ancho_terminal()), style="cyan")


def envolver(texto: str, *, sangria: str = "", sangria_siguiente: str | None = None) -> str:
    siguiente = sangria if sangria_siguiente is None else sangria_siguiente
    return textwrap.fill(
        texto,
        width=ancho_terminal(),
        initial_indent=sangria,
        subsequent_indent=siguiente,
    )


def recortar_fragmento(
    texto: str,
    spans: tuple[tuple[int, int], ...] = (),
    *,
    max_caracteres: int,
) -> tuple[str, tuple[tuple[int, int], ...]]:
    if len(texto) <= max_caracteres:
        return texto, spans

    if not spans:
        fragmento = texto[: max_caracteres - 4].rstrip() + " ..."
        return fragmento, ()

    inicio_referencia = spans[0][0]
    inicio = max(0, inicio_referencia - max_caracteres // 3)
    fin = min(len(texto), inicio + max_caracteres)
    if fin == len(texto):
        inicio = max(0, len(texto) - max_caracteres)

    desplazamiento = 0
    fragmento = texto[inicio:fin].strip()
    spans_ajustados: list[tuple[int, int]] = []

    for span_inicio, span_fin in spans:
        if span_fin <= inicio or span_inicio >= fin:
            continue
        spans_ajustados.append(
            (max(span_inicio, inicio) - inicio, min(span_fin, fin) - inicio)
        )

    if inicio > 0:
        fragmento = "... " + fragmento
        desplazamiento += 4
    if fin < len(texto):
        fragmento = fragmento + " ..."

    if desplazamiento:
        spans_ajustados = [
            (span_inicio + desplazamiento, span_fin + desplazamiento)
            for span_inicio, span_fin in spans_ajustados
        ]

    return fragmento, tuple(spans_ajustados)


def resaltar_texto(texto: str, spans: tuple[tuple[int, int], ...]) -> str:
    if not spans:
        return texto

    piezas: list[str] = []
    cursor = 0

    for inicio, fin in spans:
        piezas.append(texto[cursor:inicio])
        piezas.append(f"[[{texto[inicio:fin]}]]")
        cursor = fin

    piezas.append(texto[cursor:])
    return "".join(piezas)


def encabezado_seccion(parrafo: Parrafo) -> str:
    return f"{resumir_parte(parrafo.titulo_parte)} | {parrafo.titulo_seccion}"


def mostrar_menu(configuracion: ConfiguracionConsola) -> None:
    estado = (
        f"Modo: {MODO_ETIQUETAS[configuracion.modo_salida]}\n"
        f"Tildes ignoradas: {'sí' if configuracion.ignorar_tildes else 'no'}\n"
        f"Límite de resultados: {configuracion.limite_resultados}\n"
        f"Párrafos de contexto: {configuracion.contexto_parrafos}\n\n"
        "1) Buscar\n"
        "2) Cambiar modo de salida\n"
        "3) Ajustes\n"
        "4) Repetir última búsqueda\n"
        "5) Ayuda\n"
        "0) Salir\n\n"
        'Tip: también puedes escribir directamente la consulta, por ejemplo "dónde aparece dulcinea".'
    )
    ui_panel(estado, titulo="Menú principal", style="blue")


def mostrar_ayuda() -> None:
    texto = (
        "La búsqueda no distingue mayúsculas y minúsculas.\n"
        "Por defecto ignora tildes (configurable en ajustes).\n"
        "Consultas válidas: dulcinea, molinos, dónde aparece Sancho Panza.\n"
        "Motor: lematización + TF-IDF + similitud coseno.\n"
        "Modos: frase, párrafo, contexto, conteo y resumen por sección."
    )
    ui_panel(texto, titulo="Ayuda rápida", style="magenta")


def seleccionar_modo(configuracion: ConfiguracionConsola) -> None:
    imprimir_titulo("Modo de salida")
    for indice, clave in enumerate(MODO_ETIQUETAS, start=1):
        print(f"{indice}. {MODO_ETIQUETAS[clave]}")

    eleccion = input("Elige un modo: ").strip().lower()
    opciones = {
        "1": "frase",
        "2": "parrafo",
        "3": "contexto",
        "4": "conteo",
        "5": "seccion",
        "frase": "frase",
        "parrafo": "parrafo",
        "contexto": "contexto",
        "conteo": "conteo",
        "seccion": "seccion",
    }
    modo = opciones.get(eleccion)
    if modo:
        configuracion.modo_salida = modo
        ui_print(f"Modo actualizado a: {MODO_ETIQUETAS[modo]}", style="green")
    else:
        ui_print("No he reconocido ese modo.", style="yellow")


def menu_ajustes(configuracion: ConfiguracionConsola) -> None:
    while True:
        imprimir_titulo("Ajustes")
        print(
            f"1. Alternar normalización de tildes "
            f"({'activada' if configuracion.ignorar_tildes else 'desactivada'})"
        )
        print(f"2. Cambiar límite de resultados ({configuracion.limite_resultados})")
        print(f"3. Cambiar párrafos de contexto ({configuracion.contexto_parrafos})")
        print("0. Volver")

        eleccion = input("ajustes> ").strip().lower()
        if eleccion == "0":
            return
        if eleccion == "1":
            configuracion.ignorar_tildes = not configuracion.ignorar_tildes
            continue
        if eleccion == "2":
            configuracion.limite_resultados = pedir_entero(
                "Nuevo límite de resultados: ",
                minimo=1,
                valor_actual=configuracion.limite_resultados,
            )
            continue
        if eleccion == "3":
            configuracion.contexto_parrafos = pedir_entero(
                "Número de párrafos de contexto: ",
                minimo=0,
                valor_actual=configuracion.contexto_parrafos,
            )
            continue
        ui_print("Opción no válida.", style="yellow")


def pedir_entero(mensaje: str, *, minimo: int, valor_actual: int) -> int:
    valor = input(mensaje).strip()
    if not valor:
        return valor_actual
    if valor.isdigit() and int(valor) >= minimo:
        return int(valor)
    ui_print("Valor no válido; mantengo el anterior.", style="yellow")
    return valor_actual


def mostrar_resumen_general(resultados: ResultadosBusqueda) -> None:
    ui_print()
    ui_print(f"Consulta interpretada: {resultados.consulta}", style="bold")
    ui_print(
        f"Resultados relevantes: {resultados.total_apariciones} "
        f"en {len(resultados.coincidencias)} párrafos y "
        f"{len(resultados.resumen_secciones)} secciones.",
        style="green",
    )

    if resultados.resumen_secciones:
        primeras = resultados.resumen_secciones[:3]
        resumen = ", ".join(
            f"{item.titulo} ({item.apariciones})" for item in primeras
        )
        ui_print(f"Primeras secciones con coincidencias: {resumen}", style="cyan")

    ui_tabla_resultados(resultados, limite=min(5, len(resultados.coincidencias)))


def mostrar_por_seccion(
    resultados: ResultadosBusqueda,
    *,
    limite: int,
) -> None:
    imprimir_titulo("Resumen por sección")
    for indice, resumen in enumerate(resultados.resumen_secciones[:limite], start=1):
        ui_print(
            f"{indice}. {resumir_parte(resumen.titulo_parte)} | {resumen.titulo} "
            f"-> {resumen.apariciones} apariciones en "
            f"{resumen.parrafos_con_coincidencias} párrafos"
        )

    if len(resultados.resumen_secciones) > limite:
        ui_print(f"... hay {len(resultados.resumen_secciones) - limite} secciones más.", style="yellow")


def mostrar_por_parrafo(
    resultados: ResultadosBusqueda,
    *,
    limite: int,
) -> None:
    imprimir_titulo("Coincidencias por párrafo")
    for indice, coincidencia in enumerate(resultados.coincidencias[:limite], start=1):
        ui_print(f"[{indice}] Score: {coincidencia.score:.4f}", style="green")
        ui_print(encabezado_seccion(coincidencia.parrafo), style="cyan")
        fragmento, spans = recortar_fragmento(
            coincidencia.parrafo.texto,
            coincidencia.spans,
            max_caracteres=430,
        )
        ui_print(envolver(resaltar_texto(fragmento, spans), sangria="  "))
        ui_print()

    if len(resultados.coincidencias) > limite:
        ui_print(f"... hay {len(resultados.coincidencias) - limite} párrafos más.", style="yellow")


def _spans_relativos(
    spans: tuple[tuple[int, int], ...],
    inicio: int,
    fin: int,
) -> tuple[tuple[int, int], ...]:
    relativos: list[tuple[int, int]] = []
    for span_inicio, span_fin in spans:
        if span_fin <= inicio or span_inicio >= fin:
            continue
        relativos.append((max(span_inicio, inicio) - inicio, min(span_fin, fin) - inicio))
    return tuple(relativos)


def mostrar_por_frase(
    resultados: ResultadosBusqueda,
    *,
    limite: int,
) -> None:
    imprimir_titulo("Coincidencias por frase")
    mostradas = 0

    for coincidencia in resultados.coincidencias:
        frase_mostrada = False
        for inicio, fin in fragmentar_en_frases(coincidencia.parrafo.texto):
            spans = _spans_relativos(coincidencia.spans, inicio, fin)
            frase = coincidencia.parrafo.texto[inicio:fin].strip()
            if not frase:
                continue
            frase, spans = recortar_fragmento(frase, spans, max_caracteres=320)
            mostradas += 1
            ui_print(f"[{mostradas}] Score: {coincidencia.score:.4f}", style="green")
            ui_print(f"[{mostradas}] {encabezado_seccion(coincidencia.parrafo)}", style="cyan")
            ui_print(envolver(resaltar_texto(frase, spans), sangria="  "))
            ui_print()
            frase_mostrada = True
            break
        if not frase_mostrada:
            frase, _ = recortar_fragmento(coincidencia.parrafo.texto, max_caracteres=320)
            mostradas += 1
            ui_print(f"[{mostradas}] Score: {coincidencia.score:.4f}", style="green")
            ui_print(f"[{mostradas}] {encabezado_seccion(coincidencia.parrafo)}", style="cyan")
            ui_print(envolver(frase, sangria="  "))
            ui_print()
        if mostradas >= limite:
            break

    restantes = max(0, len(resultados.coincidencias) - mostradas)
    if restantes:
        ui_print(f"... quedan {restantes} resultados sin mostrar.", style="yellow")


def mostrar_por_contexto(
    corpus: CorpusQuijote,
    resultados: ResultadosBusqueda,
    *,
    limite: int,
    contexto: int,
) -> None:
    imprimir_titulo("Coincidencias con contexto")
    for indice, coincidencia in enumerate(resultados.coincidencias[:limite], start=1):
        parrafo = coincidencia.parrafo
        seccion = corpus.secciones[parrafo.indice_seccion]
        inicio = max(0, parrafo.posicion_en_seccion - contexto)
        fin = min(len(seccion.parrafos), parrafo.posicion_en_seccion + contexto + 1)

        ui_print(f"[{indice}] Score: {coincidencia.score:.4f}", style="green")
        ui_print(encabezado_seccion(parrafo), style="cyan")
        for posicion in range(inicio, fin):
            texto = seccion.parrafos[posicion].texto
            if posicion == parrafo.posicion_en_seccion:
                fragmento, spans = recortar_fragmento(
                    texto,
                    coincidencia.spans,
                    max_caracteres=360,
                )
                etiqueta = "  > "
                contenido = resaltar_texto(fragmento, spans)
            else:
                fragmento, _ = recortar_fragmento(texto, max_caracteres=220)
                etiqueta = "    "
                contenido = fragmento

            ui_print(envolver(contenido, sangria=etiqueta))
        ui_print()

    if len(resultados.coincidencias) > limite:
        ui_print(f"... hay {len(resultados.coincidencias) - limite} párrafos más.", style="yellow")


def mostrar_resultados(
    corpus: CorpusQuijote,
    resultados: ResultadosBusqueda,
    configuracion: ConfiguracionConsola,
) -> None:
    mostrar_resumen_general(resultados)
    if resultados.total_apariciones == 0:
        return

    if configuracion.modo_salida == "conteo":
        return
    if configuracion.modo_salida == "seccion":
        mostrar_por_seccion(resultados, limite=configuracion.limite_resultados)
        return
    if configuracion.modo_salida == "parrafo":
        mostrar_por_parrafo(resultados, limite=configuracion.limite_resultados)
        return
    if configuracion.modo_salida == "frase":
        mostrar_por_frase(resultados, limite=configuracion.limite_resultados)
        return

    mostrar_por_contexto(
        corpus,
        resultados,
        limite=configuracion.limite_resultados,
        contexto=configuracion.contexto_parrafos,
    )


def ejecutar_busqueda(
    corpus: CorpusQuijote,
    configuracion: ConfiguracionConsola,
    consulta_bruta: str,
) -> str | None:
    consulta = extraer_consulta(consulta_bruta)
    if not consulta:
        ui_print("Necesito una palabra o expresión para buscar.", style="yellow")
        return None

    try:
        resultados = buscar_en_corpus(
            corpus,
            consulta,
            ignorar_tildes=configuracion.ignorar_tildes,
        )
    except ValueError as error:
        ui_print(str(error), style="yellow")
        return None

    mostrar_resultados(corpus, resultados, configuracion)
    return consulta


def cargar_corpus() -> CorpusQuijote:
    if not RUTA_HTML.exists():
        raise FileNotFoundError(f"No encuentro el archivo {RUTA_HTML.name}.")
    return precalcular_tfidf(cargar_corpus_html(RUTA_HTML))


def bienvenida(corpus: CorpusQuijote, configuracion: ConfiguracionConsola) -> None:
    texto = (
        f"Fuente: {corpus.ruta_fuente.name} | "
        f"secciones: {len(corpus.secciones)} | "
        f"párrafos: {corpus.total_parrafos}\n"
        f"Modo inicial: {MODO_ETIQUETAS[configuracion.modo_salida]} | "
        f"tildes: {'ignoradas' if configuracion.ignorar_tildes else 'respetadas'}"
    )
    ui_panel(texto, titulo="Buscador IR de El Quijote", style="green")


def main() -> None:
    corpus = cargar_corpus()
    configuracion = ConfiguracionConsola()
    ultima_consulta: str | None = None

    bienvenida(corpus, configuracion)

    while True:
        mostrar_menu(configuracion)
        entrada = input("quijote> ").strip()
        if not entrada:
            continue

        entrada_minuscula = entrada.casefold()

        if entrada_minuscula in {"0", "salir", "exit", "q"}:
            ui_print("Hasta pronto.", style="bold green")
            return

        if entrada_minuscula in {"1", "buscar"}:
            consulta = input("Consulta: ").strip()
            ultima_consulta = ejecutar_busqueda(corpus, configuracion, consulta) or ultima_consulta
            continue

        if entrada_minuscula.startswith("buscar "):
            ultima_consulta = (
                ejecutar_busqueda(corpus, configuracion, entrada[7:]) or ultima_consulta
            )
            continue

        if entrada_minuscula in {"2", "modo"}:
            seleccionar_modo(configuracion)
            continue

        if entrada_minuscula in {"3", "ajustes"}:
            menu_ajustes(configuracion)
            continue

        if entrada_minuscula in {"4", "repetir", "ultima", "última"}:
            if not ultima_consulta:
                ui_print("Todavía no hay ninguna búsqueda previa.", style="yellow")
                continue
            ejecutar_busqueda(corpus, configuracion, ultima_consulta)
            continue

        if entrada_minuscula in {"5", "ayuda", "help"}:
            mostrar_ayuda()
            continue

        ultima_consulta = ejecutar_busqueda(corpus, configuracion, entrada) or ultima_consulta


if __name__ == "__main__":
    main()

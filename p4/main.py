from __future__ import annotations

from dataclasses import dataclass
from html import unescape
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


@dataclass(frozen=True)
class Seccion:
    titulo: str
    titulo_parte: str | None
    parrafos: tuple[Parrafo, ...]


@dataclass(frozen=True)
class CorpusQuijote:
    ruta_fuente: Path
    secciones: tuple[Seccion, ...]

    @property
    def total_parrafos(self) -> int:
        return sum(len(seccion.parrafos) for seccion in self.secciones)


@dataclass(frozen=True)
class CoincidenciaParrafo:
    parrafo: Parrafo
    spans: tuple[tuple[int, int], ...]


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


def construir_indice(texto: str, *, quitar_tildes: bool) -> IndiceTexto:
    salida: list[str] = []
    mapa: list[int] = []

    for posicion, caracter in enumerate(texto):
        fragmento = caracter.lower()
        if quitar_tildes:
            fragmento = "".join(
                signo
                for signo in unicodedata.normalize("NFD", fragmento)
                if unicodedata.category(signo) != "Mn"
            )

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
            parrafos.append(
                Parrafo(
                    texto=texto_parrafo,
                    titulo_seccion=titulo,
                    titulo_parte=titulo_parte,
                    indice_seccion=indice_seccion,
                    posicion_en_seccion=posicion_en_seccion,
                    indice_simple=construir_indice(texto_parrafo, quitar_tildes=False),
                    indice_sin_tildes=construir_indice(texto_parrafo, quitar_tildes=True),
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

    indice_consulta = construir_indice(consulta_limpia, quitar_tildes=ignorar_tildes)
    patron = _crear_patron(indice_consulta.clave)

    coincidencias: list[CoincidenciaParrafo] = []
    resumenes: list[ResumenSeccion] = []
    total_apariciones = 0

    for seccion in corpus.secciones:
        apariciones_seccion = 0
        parrafos_con_coincidencias = 0

        for parrafo in seccion.parrafos:
            indice = parrafo.indice_sin_tildes if ignorar_tildes else parrafo.indice_simple
            spans = _buscar_spans(indice, patron)
            if not spans:
                continue

            apariciones_seccion += len(spans)
            total_apariciones += len(spans)
            parrafos_con_coincidencias += 1
            coincidencias.append(CoincidenciaParrafo(parrafo=parrafo, spans=spans))

        if apariciones_seccion:
            resumenes.append(
                ResumenSeccion(
                    titulo=seccion.titulo,
                    titulo_parte=seccion.titulo_parte,
                    apariciones=apariciones_seccion,
                    parrafos_con_coincidencias=parrafos_con_coincidencias,
                )
            )

    return ResultadosBusqueda(
        consulta=consulta_limpia,
        consulta_normalizada=indice_consulta.clave,
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
    print()
    print(texto)
    print("=" * min(len(texto), ancho_terminal()))


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
    print()
    print("1. Buscar")
    print(f"2. Cambiar modo de salida (actual: {MODO_ETIQUETAS[configuracion.modo_salida]})")
    print(
        f"3. Ajustes (tildes: {'sí' if configuracion.ignorar_tildes else 'no'}, "
        f"límite: {configuracion.limite_resultados}, "
        f"contexto: {configuracion.contexto_parrafos})"
    )
    print("4. Repetir última búsqueda")
    print("5. Ayuda")
    print("0. Salir")
    print('También puedes escribir directamente una consulta como "dónde aparece dulcinea".')


def mostrar_ayuda() -> None:
    imprimir_titulo("Ayuda rápida")
    print("La búsqueda no distingue mayúsculas y minúsculas.")
    print("Por defecto también ignora tildes, pero puedes desactivarlo en ajustes.")
    print("Consultas válidas: dulcinea, molinos, dónde aparece Sancho Panza.")
    print("Modos disponibles: frase, párrafo, contexto, conteo y resumen por sección.")


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
        print(f"Modo actualizado a: {MODO_ETIQUETAS[modo]}")
    else:
        print("No he reconocido ese modo.")


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
        print("Opción no válida.")


def pedir_entero(mensaje: str, *, minimo: int, valor_actual: int) -> int:
    valor = input(mensaje).strip()
    if not valor:
        return valor_actual
    if valor.isdigit() and int(valor) >= minimo:
        return int(valor)
    print("Valor no válido; mantengo el anterior.")
    return valor_actual


def mostrar_resumen_general(resultados: ResultadosBusqueda) -> None:
    print()
    print(f"Consulta interpretada: {resultados.consulta}")
    print(
        f"Apariciones totales: {resultados.total_apariciones} "
        f"en {len(resultados.coincidencias)} párrafos y "
        f"{len(resultados.resumen_secciones)} secciones."
    )

    if resultados.resumen_secciones:
        primeras = resultados.resumen_secciones[:3]
        resumen = ", ".join(
            f"{item.titulo} ({item.apariciones})" for item in primeras
        )
        print(f"Primeras secciones con coincidencias: {resumen}")


def mostrar_por_seccion(
    resultados: ResultadosBusqueda,
    *,
    limite: int,
) -> None:
    imprimir_titulo("Resumen por sección")
    for indice, resumen in enumerate(resultados.resumen_secciones[:limite], start=1):
        print(
            f"{indice}. {resumir_parte(resumen.titulo_parte)} | {resumen.titulo} "
            f"-> {resumen.apariciones} apariciones en "
            f"{resumen.parrafos_con_coincidencias} párrafos"
        )

    if len(resultados.resumen_secciones) > limite:
        print(f"... hay {len(resultados.resumen_secciones) - limite} secciones más.")


def mostrar_por_parrafo(
    resultados: ResultadosBusqueda,
    *,
    limite: int,
) -> None:
    imprimir_titulo("Coincidencias por párrafo")
    for indice, coincidencia in enumerate(resultados.coincidencias[:limite], start=1):
        print(f"[{indice}] {encabezado_seccion(coincidencia.parrafo)}")
        fragmento, spans = recortar_fragmento(
            coincidencia.parrafo.texto,
            coincidencia.spans,
            max_caracteres=430,
        )
        print(envolver(resaltar_texto(fragmento, spans), sangria="  "))
        print()

    if len(resultados.coincidencias) > limite:
        print(f"... hay {len(resultados.coincidencias) - limite} párrafos más.")


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
        for inicio, fin in fragmentar_en_frases(coincidencia.parrafo.texto):
            spans = _spans_relativos(coincidencia.spans, inicio, fin)
            if not spans:
                continue

            frase = coincidencia.parrafo.texto[inicio:fin].strip()
            frase, spans = recortar_fragmento(frase, spans, max_caracteres=320)
            mostradas += 1
            print(f"[{mostradas}] {encabezado_seccion(coincidencia.parrafo)}")
            print(envolver(resaltar_texto(frase, spans), sangria="  "))
            print()

            if mostradas >= limite:
                break
        if mostradas >= limite:
            break

    restantes = max(0, resultados.total_apariciones - mostradas)
    if restantes:
        print(f"... quedan al menos {restantes} apariciones sin mostrar.")


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

        print(f"[{indice}] {encabezado_seccion(parrafo)}")
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

            print(envolver(contenido, sangria=etiqueta))
        print()

    if len(resultados.coincidencias) > limite:
        print(f"... hay {len(resultados.coincidencias) - limite} párrafos más.")


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
        print("Necesito una palabra o expresión para buscar.")
        return None

    resultados = buscar_en_corpus(
        corpus,
        consulta,
        ignorar_tildes=configuracion.ignorar_tildes,
    )
    mostrar_resultados(corpus, resultados, configuracion)
    return consulta


def cargar_corpus() -> CorpusQuijote:
    if not RUTA_HTML.exists():
        raise FileNotFoundError(f"No encuentro el archivo {RUTA_HTML.name}.")
    return cargar_corpus_html(RUTA_HTML)


def bienvenida(corpus: CorpusQuijote, configuracion: ConfiguracionConsola) -> None:
    imprimir_titulo("Buscador interactivo de El Quijote")
    print(
        f"Fuente: {corpus.ruta_fuente.name} | "
        f"secciones: {len(corpus.secciones)} | "
        f"párrafos: {corpus.total_parrafos}"
    )
    print(
        f"Modo inicial: {MODO_ETIQUETAS[configuracion.modo_salida]} | "
        f"tildes: {'ignoradas' if configuracion.ignorar_tildes else 'respetadas'}"
    )


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
            print("Hasta pronto.")
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
                print("Todavía no hay ninguna búsqueda previa.")
                continue
            ejecutar_busqueda(corpus, configuracion, ultima_consulta)
            continue

        if entrada_minuscula in {"5", "ayuda", "help"}:
            mostrar_ayuda()
            continue

        ultima_consulta = ejecutar_busqueda(corpus, configuracion, entrada) or ultima_consulta


if __name__ == "__main__":
    main()

from __future__ import annotations

import importlib
import shutil
import textwrap

from fdi_pln_2612_p4.modelos import ConfiguracionConsola, CorpusQuijote, Parrafo, ResultadosBusqueda
from fdi_pln_2612_p4.nlp_utils import MODO_ETIQUETAS, fragmentar_en_frases, resumir_parte


_RICH_CONSOLE = None
BUSQUEDA_ETIQUETAS = {
    "clasico": "Búsqueda clásica (lemas + TF-IDF)",
    "semantico": "Búsqueda semántica (embeddings)",
    "rag": "RAG (respuesta con contexto)",
}


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


def ancho_terminal() -> int:
    return min(100, shutil.get_terminal_size((100, 24)).columns)


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


def mostrar_menu(configuracion: ConfiguracionConsola) -> None:
    estado = (
        f"Motor: {BUSQUEDA_ETIQUETAS[configuracion.modo_busqueda]}\n"
        f"Modo: {MODO_ETIQUETAS[configuracion.modo_salida]}\n"
        f"Tildes ignoradas: {'sí' if configuracion.ignorar_tildes else 'no'}\n"
        f"Límite de resultados: {configuracion.limite_resultados}\n"
        f"Párrafos de contexto: {configuracion.contexto_parrafos}\n\n"
        "1) Buscar\n"
        "2) Cambiar motor de búsqueda\n"
        "3) Cambiar modo de salida (presentación)\n"
        "4) Ajustes\n"
        "5) Repetir última búsqueda\n"
        "6) Ayuda\n"
        "0) Salir\n\n"
        'Tip: también puedes escribir directamente la consulta, por ejemplo "dónde aparece dulcinea".'
    )
    ui_panel(estado, titulo="Menú principal", style="blue")


def mostrar_ayuda() -> None:
    texto = (
        "La búsqueda no distingue mayúsculas y minúsculas.\n"
        "Por defecto ignora tildes (configurable en ajustes).\n"
        "Consultas válidas: dulcinea, molinos, dónde aparece Sancho Panza.\n"
        "Motores: clásica, semántica y RAG (según implementación).\n"
        "Presentación: frase, párrafo, contexto, conteo y resumen por sección."
    )
    ui_panel(texto, titulo="Ayuda rápida", style="magenta")


def seleccionar_modo_busqueda(configuracion: ConfiguracionConsola) -> None:
    imprimir_titulo("Motor de búsqueda")
    for indice, clave in enumerate(BUSQUEDA_ETIQUETAS, start=1):
        ui_print(f"{indice}. {BUSQUEDA_ETIQUETAS[clave]}")

    eleccion = input("Elige un motor: ").strip().lower()
    opciones = {
        "1": "clasico",
        "2": "semantico",
        "3": "rag",
        "clasico": "clasico",
        "clásico": "clasico",
        "semantico": "semantico",
        "semántico": "semantico",
        "rag": "rag",
    }
    modo = opciones.get(eleccion)
    if modo:
        configuracion.modo_busqueda = modo
        ui_print(f"Motor actualizado a: {BUSQUEDA_ETIQUETAS[modo]}", style="green")
    else:
        ui_print("No he reconocido ese motor.", style="yellow")


def seleccionar_modo(configuracion: ConfiguracionConsola) -> None:
    imprimir_titulo("Modo de salida")
    for indice, clave in enumerate(MODO_ETIQUETAS, start=1):
        ui_print(f"{indice}. {MODO_ETIQUETAS[clave]}")

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


def pedir_entero(mensaje: str, *, minimo: int, valor_actual: int) -> int:
    valor = input(mensaje).strip()
    if not valor:
        return valor_actual
    if valor.isdigit() and int(valor) >= minimo:
        return int(valor)
    ui_print("Valor no válido; mantengo el anterior.", style="yellow")
    return valor_actual


def menu_ajustes(configuracion: ConfiguracionConsola) -> None:
    while True:
        imprimir_titulo("Ajustes")
        ui_print(
            f"1. Alternar normalización de tildes "
            f"({'activada' if configuracion.ignorar_tildes else 'desactivada'})"
        )
        ui_print(f"2. Cambiar límite de resultados ({configuracion.limite_resultados})")
        ui_print(f"3. Cambiar párrafos de contexto ({configuracion.contexto_parrafos})")
        ui_print("0. Volver")

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


def mostrar_por_seccion(resultados: ResultadosBusqueda, *, limite: int) -> None:
    imprimir_titulo("Resumen por sección")
    for indice, resumen in enumerate(resultados.resumen_secciones[:limite], start=1):
        ui_print(
            f"{indice}. {resumir_parte(resumen.titulo_parte)} | {resumen.titulo} "
            f"-> {resumen.apariciones} apariciones en "
            f"{resumen.parrafos_con_coincidencias} párrafos"
        )

    if len(resultados.resumen_secciones) > limite:
        ui_print(f"... hay {len(resultados.resumen_secciones) - limite} secciones más.", style="yellow")


def mostrar_por_parrafo(resultados: ResultadosBusqueda, *, limite: int) -> None:
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


def mostrar_por_frase(resultados: ResultadosBusqueda, *, limite: int) -> None:
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


def bienvenida(corpus: CorpusQuijote, configuracion: ConfiguracionConsola) -> None:
    texto = (
        f"Fuente: {corpus.ruta_fuente.name} | "
        f"secciones: {len(corpus.secciones)} | "
        f"párrafos: {corpus.total_parrafos}\n"
        f"Motor inicial: {BUSQUEDA_ETIQUETAS[configuracion.modo_busqueda]} | "
        f"Modo inicial: {MODO_ETIQUETAS[configuracion.modo_salida]} | "
        f"tildes: {'ignoradas' if configuracion.ignorar_tildes else 'respetadas'}"
    )
    ui_panel(texto, titulo="Buscador IR de El Quijote", style="green")

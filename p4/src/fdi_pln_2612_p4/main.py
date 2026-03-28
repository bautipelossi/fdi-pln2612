from __future__ import annotations

from pathlib import Path

from fdi_pln_2612_p4.corpus_loader import cargar_corpus_html
from fdi_pln_2612_p4.embeddings import buscar_en_corpus_semantico, precalcular_embeddings
from fdi_pln_2612_p4.ir_clasico import buscar_en_corpus, precalcular_tfidf
from fdi_pln_2612_p4.modelos import ConfiguracionConsola, CorpusQuijote
from fdi_pln_2612_p4.nlp_utils import extraer_consulta
from fdi_pln_2612_p4.rag import preparar_contexto_rag, responder_rag
from fdi_pln_2612_p4.ui_terminal import (
    bienvenida,
    menu_ajustes,
    mostrar_ayuda,
    mostrar_menu,
    mostrar_resultados,
    seleccionar_modo_busqueda,
    seleccionar_modo,
    ui_print,
)

RUTA_BASE = Path(__file__).resolve().parents[2]
RUTA_HTML = RUTA_BASE / "2000-h.htm"


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
        if configuracion.modo_busqueda == "clasico":
            resultados = buscar_en_corpus(
                corpus,
                consulta,
                ignorar_tildes=configuracion.ignorar_tildes,
            )
            mostrar_resultados(corpus, resultados, configuracion)
        elif configuracion.modo_busqueda == "semantico":
            resultados = buscar_en_corpus_semantico(
                corpus,
                consulta,
                ignorar_tildes=configuracion.ignorar_tildes,
            )
            mostrar_resultados(corpus, resultados, configuracion)
        elif configuracion.modo_busqueda == "rag":
            resultados_clasicos = buscar_en_corpus(
                corpus,
                consulta,
                ignorar_tildes=configuracion.ignorar_tildes,
            )
            resultados_semanticos = buscar_en_corpus_semantico(
                corpus,
                consulta,
                ignorar_tildes=configuracion.ignorar_tildes,
            )
            resultados_rag = preparar_contexto_rag(
                corpus,
                resultados_clasicos,
                resultados_semanticos,
            )
            mostrar_resultados(corpus, resultados_rag, configuracion)
            respuesta = responder_rag(consulta, resultados_rag)
            ui_print()
            ui_print("Respuesta RAG", style="bold magenta")
            ui_print(respuesta)
        else:
            ui_print("Motor de búsqueda no reconocido.", style="yellow")
            return None
    except (ValueError, NotImplementedError, RuntimeError) as error:
        ui_print(str(error), style="yellow")
        return None

    return consulta


def cargar_corpus() -> CorpusQuijote:
    if not RUTA_HTML.exists():
        raise FileNotFoundError(f"No encuentro el archivo {RUTA_HTML.name}.")
    corpus = precalcular_tfidf(cargar_corpus_html(RUTA_HTML))
    return precalcular_embeddings(corpus)


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

        if entrada_minuscula in {"2", "motor", "busqueda", "búsqueda"}:
            seleccionar_modo_busqueda(configuracion)
            continue

        if entrada_minuscula in {"3", "modo"}:
            seleccionar_modo(configuracion)
            continue

        if entrada_minuscula in {"4", "ajustes"}:
            menu_ajustes(configuracion)
            continue

        if entrada_minuscula in {"5", "repetir", "ultima", "última"}:
            if not ultima_consulta:
                ui_print("Todavía no hay ninguna búsqueda previa.", style="yellow")
                continue
            ejecutar_busqueda(corpus, configuracion, ultima_consulta)
            continue

        if entrada_minuscula in {"6", "ayuda", "help"}:
            mostrar_ayuda()
            continue

        ultima_consulta = ejecutar_busqueda(corpus, configuracion, entrada) or ultima_consulta


if __name__ == "__main__":
    main()

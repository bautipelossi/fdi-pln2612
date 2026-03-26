from __future__ import annotations

from fdi_pln_2612_p4.modelos import ResultadosBusqueda


def responder_rag(
    consulta: str,
    resultados_clasicos: ResultadosBusqueda,
    resultados_semanticos: ResultadosBusqueda | None = None,
) -> str:
    raise NotImplementedError(
        "RAG aún no implementado. El módulo está preparado para integrar LLM en la siguiente iteración."
    )

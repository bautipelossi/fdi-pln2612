from __future__ import annotations

from fdi_pln_2612_p4.modelos import CorpusQuijote, ResultadosBusqueda


def buscar_en_corpus_semantico(
    corpus: CorpusQuijote,
    consulta: str,
    *,
    ignorar_tildes: bool = True,
) -> ResultadosBusqueda:
    raise NotImplementedError(
        "Búsqueda semántica aún no implementada. "
        "El módulo está preparado para integrar embeddings en la siguiente iteración."
    )

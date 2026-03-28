from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


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
    vector_semantico: tuple[float, ...] = ()


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
    modo_busqueda: str = "clasico"
    ignorar_tildes: bool = True
    limite_resultados: int = 5
    contexto_parrafos: int = 1

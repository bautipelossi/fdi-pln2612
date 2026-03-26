# Práctica 4 — Buscador IR de El Quijote

**Grupo 12 · Procesamiento del Lenguaje Natural (2612)**
Bautista Pelossi Schweizer · Ignacio Ramírez Suarez

---

## Descripción

Aplicación de terminal para recuperar información sobre el corpus de *Don Quijote* con una interfaz interactiva en consola.

La aplicación ofrece tres motores seleccionables por el usuario:

1. **Búsqueda clásica**: lematización + eliminación de stopwords + ranking TF-IDF.
2. **Búsqueda semántica**: estructura preparada para embeddings (pendiente de implementación).
3. **RAG**: estructura preparada para respuesta aumentada con recuperación (pendiente de implementación).

El flujo principal actual en modo clásico es:

1. Cargar y parsear el HTML del Quijote.
2. Dividir en secciones y párrafos.
3. Preprocesar párrafos con spaCy (lemas y stopwords).
4. Precalcular estructura TF-IDF una sola vez.
5. Procesar consulta y devolver resultados ordenados por relevancia.

---

## Arquitectura

```text
src/fdi_pln_2612_p4/
├── main.py          # Punto de entrada y orquestación de la app
├── modelos.py       # Dataclasses de dominio (corpus, párrafos, resultados, config)
├── corpus_loader.py # Parseo del HTML y construcción de CorpusQuijote
├── nlp_utils.py     # Preprocesado NLP (spaCy), normalización y utilidades de texto
├── ir_clasico.py    # Motor clásico: TF-IDF + similitud coseno + ranking
├── embeddings.py    # Interfaz del motor semántico (placeholder)
├── rag.py           # Interfaz de RAG (placeholder)
└── ui_terminal.py   # TUI en terminal (menús, paneles, tablas y resultados)
```

| Módulo | Responsabilidad |
|---|---|
| `main.py` | Gestiona menú, selección de motor y ciclo interactivo. |
| `modelos.py` | Define estructuras tipadas para corpus, resultados y configuración. |
| `corpus_loader.py` | Convierte el HTML del Quijote en secciones y párrafos estructurados. |
| `nlp_utils.py` | Lematiza, elimina stopwords y normaliza consulta/documentos con spaCy. |
| `ir_clasico.py` | Calcula TF, IDF, vectores TF-IDF y ranking por similitud coseno. |
| `ui_terminal.py` | Presenta TUI con menú, opciones de motor/modo y resultados con score. |
| `embeddings.py` | Punto de extensión para búsqueda por embeddings. |
| `rag.py` | Punto de extensión para respuesta con contexto recuperado. |

---

## Requisitos

- **Python** >= 3.12
- **[uv](https://docs.astral.sh/uv/)**
- Dependencias permitidas por la consigna:
	- `spacy`
	- `es-core-news-sm`
	- `rich`

---

## Instalación

```bash
cd p4
uv sync
```

No se requiere crear entornos manualmente ni descargar modelos con comandos separados: el modelo está declarado en `pyproject.toml`.

---

## Uso

### Ejecución

```bash
uv run fdi-pln-2612-p4
```

### Opciones del menú

1. Buscar
2. Cambiar motor de búsqueda
3. Cambiar modo de salida
4. Ajustes
5. Repetir última búsqueda
6. Ayuda
0. Salir

### Ejemplos de consultas

- `molinos de viento`
- `dulcinea`
- `dónde aparece Sancho Panza`

---

## Implementado

### 1. Reprocesado

- Preprocesado con spaCy:
    - tokenización
    - lematización
    - eliminación de stopwords
    - normalización para consulta y corpus
- Segmentación del corpus en unidades recuperables:
    - división por secciones y párrafos del texto fuente
    - cada párrafo funciona como documento del índice clásico
- Normalización lingüística alineada con IR clásico:
    - minúsculas para reducir variación superficial
    - manejo de tildes para mejorar recall en español
    - consulta y documentos pasan por el mismo pipeline


### 2. IR Clásico (TF-IDF)
- Búsqueda clásica con ranking TF-IDF y similitud coseno.
- TUI interactiva en terminal con resultados ponderados por score.
- Arquitectura modular en `src/`.
- Representación vectorial de documentos:
    - construcción de vocabulario sobre lemas del corpus
    - cálculo de frecuencia de término (TF) por documento
    - cálculo de frecuencia documental (DF)
    - cálculo de peso inverso (IDF = log(N / DF))
    - vector TF-IDF para cada párrafo
- Representación vectorial de consulta:
    - misma normalización que el corpus
    - construcción del vector TF-IDF de consulta en el mismo espacio
- Función de similitud y ranking:
    - similitud coseno entre vector de consulta y documentos
    - ordenación descendente por score de relevancia
    - devolución de pasajes con contexto y sección para interpretación.
---

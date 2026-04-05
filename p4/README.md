# Práctica 4 — Buscador IR de El Quijote

**Grupo 12 · Procesamiento del Lenguaje Natural (2612)**
Bautista Pelossi Schweizer · Ignacio Ramírez Suarez

---

## Descripción

Aplicación de terminal para recuperar información sobre el corpus de *Don Quijote* con una interfaz interactiva en consola.

La aplicación ofrece tres motores seleccionables por el usuario:

1. **Búsqueda clásica**: lematización + eliminación de stopwords + ranking TF-IDF.
2. **Búsqueda semántica**: embeddings densos con `spaCy` (`tok2vec`) y ranking por similitud coseno.
3. **RAG**: recuperación híbrida (clásica + semántica) y respuesta generada con LLM local (`ollama`) basada en evidencias recuperadas.

El flujo principal actual en modo clásico es:

1. Cargar y parsear el HTML del Quijote.
2. Dividir en secciones, párrafos y chunks con overlap.
3. Preprocesar unidades recuperables con spaCy (lemas y stopwords).
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
├── embeddings.py    # Precálculo de embeddings y búsqueda semántica
├── rag.py           # Recuperación híbrida y síntesis RAG
└── ui_terminal.py   # TUI en terminal (menús, paneles, tablas y resultados)
```

| Módulo | Responsabilidad |
|---|---|
| `main.py` | Gestiona menú, selección de motor y ciclo interactivo. |
| `modelos.py` | Define estructuras tipadas para corpus, resultados y configuración. |
| `corpus_loader.py` | Convierte el HTML del Quijote en secciones, párrafos y chunks con overlap. |
| `nlp_utils.py` | Lematiza, elimina stopwords y normaliza consulta/documentos con spaCy. |
| `ir_clasico.py` | Calcula TF, IDF, vectores TF-IDF y ranking por similitud coseno sobre chunks. |
| `ui_terminal.py` | Presenta TUI con menú, opciones de motor/modo y resultados con score. |
| `embeddings.py` | Precalcula vectores densos por chunk y ejecuta búsqueda semántica híbrida. |
| `rag.py` | Fusiona recuperación clásica y semántica y construye una respuesta basada en evidencias. |

---

## Requisitos

- **Python** >= 3.12
- **[uv](https://docs.astral.sh/uv/)**
- Dependencias permitidas por la consigna:
	- `spacy`
	- `es-core-news-sm`
	- `rich`
    - `ollama`

---

## Instalación

```bash
cd p4
uv sync
```

No se requiere crear entornos manualmente ni descargar modelos con comandos separados: el modelo está declarado en `pyproject.toml`.

Para usar RAG con LLM local, debes tener el servicio de Ollama activo y al menos un modelo instalado.

### Preparación de Ollama para RAG

1. Instalar Ollama en el sistema.
2. Verificar que el servicio esté activo.
3. Descargar al menos un modelo local, por ejemplo:

```bash
ollama pull llama3.2:3b
```

4. (Opcional) Seleccionar modelo con variable de entorno `RAG_OLLAMA_MODEL`.

Si Ollama no está disponible, el modo RAG mantiene el funcionamiento con una respuesta extractiva local de respaldo.

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

### Probar cada motor

1. Ejecuta `uv run fdi-pln-2612-p4`.
2. En el menú principal, elige `2` para cambiar el motor.
3. Selecciona:
   - `1` para búsqueda clásica
   - `2` para búsqueda semántica
   - `3` para RAG
4. Lanza la consulta con `1` o escribiéndola directamente.

### Ejemplos de consultas

- `molinos de viento`
- `dulcinea`
- `dónde aparece Sancho Panza`

### Configuración opcional de modelo RAG

- Variable de entorno: `RAG_OLLAMA_MODEL`
- Valor por defecto: `llama3.2:3b`

Ejemplo en PowerShell:

```powershell
$env:RAG_OLLAMA_MODEL = "llama3.2:3b"
uv run fdi-pln-2612-p4
```

### Comprobación rápida de RAG con LLM

1. Ejecutar la app.
2. Cambiar motor a RAG.
3. Consultar, por ejemplo: `dulcinea`.
4. Confirmar salida:
    - Si el LLM responde correctamente, no aparece el aviso de fallback.
    - Si aparece el aviso de fallback, revisar estado del servicio/modelo de Ollama.

---

## Checklist de entrega

### Repositorio y documentación

- README en la raíz con integrantes y guía de uso.
- Historial de commits del equipo verificable.

### Formato de código

```bash
uv format --check
```

Debe finalizar sin errores.

### Ejecución

```bash
uv run fdi-pln-2612-p4
```

Debe iniciar correctamente en entorno local y en Linux de laboratorio.

### Wheel

```bash
uv build
```

El artefacto queda en `dist/` y debe incluir ejecutable `fdi-pln-2612-p4`.

### Datos y modelos

- El repositorio incluye los datos necesarios del corpus.
- No se incluyen modelos de IA pesados dentro del repositorio.
- El preprocesado y construcción de índices/embeddings se puede regenerar bajo demanda por código.

### Dependencias permitidas usadas en esta práctica

- `spacy`
- `es-core-news-sm`
- `rich`
- `ollama`

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
    - construcción de chunks con ventana deslizante y overlap configurable
    - cada chunk funciona como documento recuperable para IR clásico y semántico
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
    - vector TF-IDF para cada chunk
- Representación vectorial de consulta:
    - misma normalización que el corpus
    - construcción del vector TF-IDF de consulta en el mismo espacio
- Función de similitud y ranking:
    - similitud coseno entre vector de consulta y documentos
    - ordenación descendente por score de relevancia
    - devolución de pasajes con contexto y sección para interpretación.
    - deduplicación por párrafo representativo cuando múltiples chunks coinciden.

### 3. Búsqueda semántica

- Precálculo de embeddings densos por chunk usando el pipeline `tok2vec` ya disponible en `spaCy`.
- Reutilización del mismo corpus segmentado en chunks y del mismo flujo de `ResultadosBusqueda` que usa el modo clásico.
- Ranking por similitud coseno sobre vectores densos y filtrado adaptativo para evitar devolver todo el corpus.
- Score híbrido en semántico: combinación de similitud densa y señal TF-IDF del chunk.

### 4. RAG

- Recuperación híbrida combinando los mejores resultados clásicos y semánticos.
- Construcción de contexto con evidencias recuperadas (fragmentos con referencia `[E1]`, `[E2]`, ...).
- Generación de respuesta con LLM local vía `ollama`, restringida al contexto recuperado.
- Fallback extractivo local si `ollama` no está disponible en ejecución.

### 5. Cumplimiento operativo

- Formato de código validado con `uv format --check`.
- Ejecución principal por script de proyecto (`uv run fdi-pln-2612-p4`).
- Arquitectura modular y preprocesado reproducible desde código.
---

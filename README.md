# Procesamiento del Lenguaje Natural (PLN)

**Grupo 12**  
**Bautista Pelossi Schweizer · Ignacio Ramirez Suarez**  
Facultad de Informatica | Universidad Complutense de Madrid

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Repo](https://img.shields.io/badge/github-bautipelossi%2Ffdi--pln2612-black.svg)](https://github.com/bautipelossi/fdi-pln2612)

> Repositorio con las actividades prácticas del curso **Procesamiento de Lenguaje Natural (PLN)**.  
> Organización modular por práctica

## Indice

1. [Descripcion](#descripcion)
2. [Estructura actual del repositorio](#estructura-actual-del-repositorio)
3. [Guia rapida por practica](#guia-rapida-por-practica)
4. [Como empezar](#como-empezar)
5. [Autores](#autores)

## Descripcion

Este repositorio contiene las practicas de la materia, cada una en su carpeta:

- **P1**: agente autonomo para mercado de trueques (Butler) con toma de decisiones asistida por LLM local.
- **P2**: trabajo de fonetica/sintesis por concatenacion de pangramas, con audios originales y sinteticos.
- **P3**: script para codificar/decodificar ficheros entre UTF-8 y formato binario PLNCG26 utilizado en la asignatura.
- **P4**: buscador IR sobre el corpus del Quijote con tres modos: clasico (TF/IDF), semantico (mediante **embeddings**) y **RAG**.
- **P5**: LLM causal pequeno con Transformer, generacion de texto y NER.

## Estructura actual del repositorio

```text
fdi-pln2612/
├── butler_local/                    # Entorno y utilidades locales para trabajo con Butler
├── p1/                              # Practica 1: Agente de trueque Butler
│   ├── README.md
│   ├── pyproject.toml
│   ├── uv.lock
│   └── src/
│       └── fdi_pln_2612_p1/
│           ├── __init__.py
│           ├── main.py
│           ├── strategy.py
│           ├── llm.py
│           ├── protocol.py
│           ├── models.py
│           ├── http_client.py
│           ├── butler_api.py
│           └── config.py
├── p2/                              # Practica 2: Audios y pangramas
│   ├── README.md
│   ├── originales/
│   └── sinteticos/
├── p3/                              # Practica 3: Criptoglifos PLNCG26
│   └── fdi-pln-2612-p3.py
├── p4/                              # Practica 4: IR sobre El Quijote
│   ├── README.md
│   ├── pyproject.toml
│   ├── uv.lock
│   └── src/
│       └── fdi_pln_2612_p4/
│           ├── __init__.py
│           ├── data/
│           │   ├── __init__.py
│           │   ├── 2000-h.htm
│           │   └── 2000-desde-PROLOGO.txt
│           ├── main.py
│           ├── modelos.py
│           ├── corpus_loader.py
│           ├── nlp_utils.py
│           ├── ir_clasico.py
│           ├── embeddings.py
│           ├── rag.py
│           └── ui_terminal.py
├── p5/                              # Practica 5: LLM causal con Transformer
│   ├── README.md
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── corpus/
│   ├── data_ner/
│   ├── tests/
│   └── src/
└── README.md
```

## Guia rapida por practica

### P1 - Agente Butler

- Documentacion: `p1/README.md`
- Requisitos: Python >= 3.11, `uv`, Ollama y acceso al Butler de clase.
- Ejecucion rapida:

```bash
cd p1
uv sync
uv run fdi-pln-2612-p1
```

### P2 - Audio y sintesis por concatenacion

- Documentacion: `p2/README.md`
- Contenido principal:
	- `p2/originales/` (grabaciones fuente)
	- `p2/sinteticos/` (resultados finales)

### P3 - Criptoglifos PLNCG26

- Script principal: `p3/fdi-pln-2612-p3.py`
- Dependencia: `typer` (declarada en cabecera del script).
- Ejemplos de uso:

```bash
python p3/fdi-pln-2612-p3.py decode archivo.plncg26 > salida.txt
python p3/fdi-pln-2612-p3.py encode entrada.txt > salida.plncg26
python p3/fdi-pln-2612-p3.py detect archivo.bin
```

### P4 - Buscador IR del Quijote

- Documentacion: `p4/README.md`
- Requisitos: Python >= 3.12, `uv`.
- Ejecucion rapida:

```bash
cd p4
uv sync
uv run fdi-pln-2612-p4
```

### P5 - LLM causal con Transformer

- Documentacion: `p5/README.md`
- Requisitos: Python >= 3.11, `uv`.
- Ejecucion rapida:

```bash
cd p5
uv sync
uv run fdi-pln-2612-p5 --help
```
## Instalación

### A) Clonar el repositorio
```bash
git clone https://github.com/bautipelossi/fdi-pln2612.git
cd fdi-pln2612
```
### B) Instalar dependencias de cada práctica
Buscar las dependencias correspondientes a cada practica en su README.me


## Autores

- Bautista Pelossi Schweizer
- Ignacio Ramirez Suarez

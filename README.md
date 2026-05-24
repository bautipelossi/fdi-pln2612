# Procesamiento del Lenguaje Natural (PLN)

**Grupo 12** · Bautista Pelossi Schweizer · Ignacio Ramirez Suarez  
Ingenieria de Datos e Inteligencia Artifical | Facultad de Informática | Universidad Complutense de Madrid

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Repo](https://img.shields.io/badge/github-bautipelossi%2Ffdi--pln2612-black.svg)](https://github.com/bautipelossi/fdi-pln2612)

> Repositorio con las actividades prácticas del curso **Procesamiento de Lenguaje Natural (PLN)**.
> 
> Organización modular por práctica.

---

## Índice

1. [Descripción](#descripción)
2. [Estructura del repositorio](#estructura-del-repositorio)
3. [Guía rápida por práctica](#guía-rápida-por-práctica)
4. [Instalación](#instalación)
5. [Autores](#autores)

---

## Descripción

| Práctica | Tema 
|---|---
| **P1** | Agente autónomo de trueque (Butler) con toma de decisiones asistida por LLM local 
| **P2** | Fonética y síntesis por concatenación de pangramas 
| **P3** | Codificador/decodificador del formato binario PLNCG26 de la asignatura 
| **P4** | Buscador IR sobre el corpus del Quijote: clásico (TF-IDF), semántico (embeddings) y RAG 
| **P5** | LLM causal con Transformer entrenado sobre *Alice in Wonderland*, con generación de texto y NER 

---

## Estructura del repositorio

```text
fdi-pln2612/
├── p1/                              # Agente de trueque Butler
│   ├── README.md
│   ├── pyproject.toml
│   └── src/fdi_pln_2612_p1/
│       ├── main.py
│       ├── strategy.py
│       ├── llm.py
│       ├── protocol.py
│       ├── models.py
│       ├── http_client.py
│       ├── butler_api.py
│       └── config.py
├── p2/                              # Audios y pangramas
│   ├── README.md
│   ├── originales/
│   └── sinteticos/
├── p3/                              # Criptoglifos PLNCG26
│   └── fdi-pln-2612-p3.py
├── p4/                              # IR sobre El Quijote
│   ├── README.md
│   ├── pyproject.toml
│   └── src/fdi_pln_2612_p4/
│       ├── data/
│       ├── corpus_loader.py
│       ├── ir_clasico.py
│       ├── embeddings.py
│       ├── rag.py
│       └── ui_terminal.py
├── p5/                              # LLM causal con Transformer
│   ├── README.md
│   ├── pyproject.toml
│   ├── corpus/
│   │   ├── alice_in_wonderland.txt
│   │   └── looking_glass.txt
│   ├── data_ner/
│   │   └── corpus_tag.json
│   └── src/
│       ├── tokenizer.py             
│       ├── attention.py             
│       ├── transformer.py           
│       ├── causal_llm.py            
│       ├── causal_train.py         
│       ├── ner.py                   
│       └── cli.py                   
└── README.md
```

---

## Guía rápida por práctica

### P1 — Agente Butler

Agente autónomo que participa en un mercado de trueques tomando decisiones de compra/venta asistidas por un LLM local (Ollama).

- Documentación: `p1/README.md`
- Requisitos: Python ≥ 3.11, `uv`, Ollama y acceso al Butler de clase.

```bash
cd p1
uv sync
uv run fdi-pln-2612-p1
```

---

### P2 — Audio y síntesis por concatenación

Pangramas grabados y sintetizados por concatenación de fonemas. Los resultados están en `p2/originales/` y `p2/sinteticos/`.

- Documentación: `p2/README.md`

---

### P3 — Criptoglifos PLNCG26

Script para codificar y decodificar ficheros entre UTF-8 y el formato binario PLNCG26 de la asignatura.

- Dependencia: `typer` (declarada en la cabecera del script).

```bash
python p3/fdi-pln-2612-p3.py decode archivo.plncg26 > salida.txt
python p3/fdi-pln-2612-p3.py encode entrada.txt > salida.plncg26
python p3/fdi-pln-2612-p3.py detect archivo.bin
```

---

### P4 — Buscador IR del Quijote

Buscador sobre el texto completo de *El Quijote* con tres modos de recuperación: clásico (TF-IDF), semántico (embeddings) y RAG con generación de respuesta.

- Documentación: `p4/README.md`
- Requisitos: Python ≥ 3.12, `uv`.

```bash
cd p4
uv sync
uv run fdi-pln-2612-p4
```

---

### P5 — LLM causal con Transformer

Modelo de lenguaje autoregresivo entrenado desde cero sobre *Alice in Wonderland* y *Through the Looking-Glass*. Implementa tokenización BPE, autoatención multi-cabezal con máscara causal, backbone Transformer y cabeza NER para reconocimiento de personas y lugares.

- Documentación: `p5/README.md`
- Requisitos: Python ≥ 3.11, `uv`.

```bash
cd p5
uv sync

# Entrenar el LLM
uv run fdi-pln-2612-p5 train-llm corpus \
  --vocab-size 300 --context-size 64 --d-model 128 \
  --n-heads 2 --n-layers 4 --epochs 4 --out p5_causal_2612.pth

# Generar texto
uv run fdi-pln-2612-p5 generate \
  --weights p5_causal_2612.pth \
  --prompt "alice was beginning to " --top-k 40

# Entrenar NER y predecir entidades
uv run fdi-pln-2612-p5 train-ner \
  --data data_ner/corpus_tag.json \
  --llm-weights p5_causal_2612.pth --out p5_ner_2612.pth

uv run fdi-pln-2612-p5 predict-ner \
  --weights p5_ner_2612.pth --input ruta/al/texto.txt
```

Ver `p5/README.md` para la lista completa de hiperparámetros y el smoke test.

---

## Instalación

```bash
git clone https://github.com/bautipelossi/fdi-pln2612.git
cd fdi-pln2612
```

Cada práctica tiene su propio entorno. Instalar las dependencias entrando en la carpeta correspondiente:

```bash
cd pN
uv sync
```

Las dependencias y la versión de Python requerida están especificadas en cada `pyproject.toml`.

---

## Autores

- Bautista Pelossi Schweizer
- Ignacio Ramirez Suarez

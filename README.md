# Procesamiento de Lenguaje Natural 
**Bautista Pelossi Schweizer · Ignacio Ramírez Suárez**  
Facultad de Informática | Universidad Complutense de Madrid  

[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Repo](https://img.shields.io/badge/github-bautipelossi%2Ffdi--pln2612-black.svg)](#)

> Repositorio con las actividades prácticas del curso **Procesamiento de Lenguaje Natural (PLN)**.  
> Organización modular por práctica

---

## Índice
1. [Descripción](#descripcion)
2. [Estructura del repositorio](#estructura)
3. [Instalación](#instalacion)
4. [Uso rápido](#uso)
5. [Autores](#autores)

---

<a id="descripcion"></a>
## Descripción

Este repositorio organiza de forma **modular** las prácticas de PLN:

- **Práctica 1 (P1):** Implementación de un **agente autónomo** que participa en un mercado de trueques coordinado por un servidor central (**Butler**). El agente debe negociar, responder ofertas, contraofertar y tomar decisiones estratégicas (usando LLM).
- **Práctica 2 (P2):** Trabajo con **audios/pangramas**, edición y generación de audios sintéticos, y entrega estructurada según consigna.

Cada práctica tiene su **carpeta propia** y su **README.md** específico.

---

<a id="estructura"></a>
## Estructura del repositorio

```text
fdi-pln2612/
│
├── p1/                              # Práctica 1: Agente Butler
│   ├── README.md 
│   ├── src/
│   │   └── fdi_pln_2612_p1/
│   │       ├── __init__.py
│   │       ├── main.py
│   │       ├── strategy.py
│   │       ├── llm.py
│   │       ├── protocol.py
│   │       ├── models.py
│   │       ├── http_client.py
│   │       ├── butler_api.py
│   │       └── config.py
│   ├── pyproject.toml
│   └── uv.lock
│
├── p2-g12/                          # Práctica 2: Audios / Pangramas (Grupo 12)
│   ├── README.md
│   ├── originales/                  #pangramas originales
│   │   └── es_b.mp3               
│   └── sinteticos/                  # pangramas finales (sintéticos / resultado)
│   │   └── es_b.mp3  
├── estado_butler.json                # archivo auxiliar de estado (si aplica)
├── .env                              # variables de entorno locales (NO subir si tiene secretos)
├── .gitignore
└── README.md                         # este README (raíz)
```

<a id="instalacion"></a>
## Instalación

### A) Clonar el repositorio
```bash
git clone https://github.com/bautipelossi/fdi-pln2612.git
cd fdi-pln2612
```
### B) Instalar dependencias de cada práctica
Buscar las dependencias correspondientes a cada practica en su README.me correspondiente

## Autores

- **Bautista Pelossi Schweizer**
- **Ignacio Ramírez Suárez**

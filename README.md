# Procesamiento de Lenguaje Natural
**Bautista Pelossi Schweizer, Ignacio Ramírez Suárez**  
Facultad de Informática | Universidad Complutense de Madrid

> Repositorio con las actividades prácticas del curso **Procesamiento de Lenguaje Natural (PLN)**.  

---

## Índice
1. [Descripción](#descripcion)
2. [Estructura del repositorio](#estructura)
5. [Notas importantes](#notas)
6. [Autores](#autor)

---

<a id="descripcion"></a>
## Descripción

Este repositorio organiza de forma **modular** las prácticas de PLN:

- **Práctica 1 (P1):** Implementación de un **agente autónomo** que participa en un mercado de trueque coordinado por un Butler. El agente debe negociar, responder ofertas, contraofertar y tomar decisiones estratégicas (opcionalmente con un LLM).
- **Práctica 2 (P2):** Trabajo con **audios** (pangramas), edición y entrega de resultados.

Cada práctica incluye su propio README.md con detalles sobre las actividades desarrolladas.
---

<a id="estructura"></a>
## Estructura del repositorio

```text
fdi-pln2612/
│
├── p1/                           # Práctica 1: Agente Butler
│   └── src/
│       └── fdi_pln_2612_p1/
│           ├── __init__.py
│           └── main.py
│
├── p2/                           # Práctica 2: Audios / Pangramas
│   ├── data/
│   │   ├── crudo/                  # audios originales (sin edición)
│   │   └── procesado/            # audios finales (editados / sintéticos)
├── .gitignore
└── README.md

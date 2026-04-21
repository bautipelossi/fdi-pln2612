# P5 - LLM Causal con Transformer

Practica de PLN para implementar un modelo autoregresivo completo:
tokenizacion BPE, autoatencion multi-cabezal, backbone Transformer,
cabeza causal y entrenamiento sobre corpus de texto.

## Estructura

```text
p5/
├── corpus/
│   ├── alice_in_wonderland.txt
│   └── looking_glass.txt
├── src/
│   ├── __init__.py
│   ├── attention.py
│   ├── corpus.py
│   ├── tokenizer.py
│   ├── transformer.py
│   ├── causal_llm.py
│   └── causal_train.py
├── pyproject.toml
└── README.md
```

## Modulos principales

- `src/tokenizer.py`: tokenizador BPE entrenado sobre el texto del corpus.
- `src/corpus.py`: utilidades para cargar corpus y construir batches.
- `src/attention.py`: self-attention multi-cabezal con mascara causal.
- `src/transformer.py`: bloques Transformer (embeddings, blocks, normalizacion).
- `src/causal_llm.py`: modelo causal con cabeza de lenguaje y generate.
- `src/causal_train.py`: entrenamiento por epocas y validacion.

## Flujo de trabajo

1. Se carga y concatena el corpus desde `corpus/*.txt`.
2. Se entrena `BPETokenizer` y se tokeniza el texto.
3. Se instancia `CausalLLM`.
4. Se entrena con `train(...)` en `src/causal_train.py`.
5. Se genera texto con `model.generate(...)`.

## Instalacion

```bash
uv sync
```

## Ejecucion

Ejecutar el script de entrenamiento con los defaults definidos en `main`:

```bash
uv run python -m src.causal_train
```

Si se quiere usar otra carpeta de textos, se puede pasar como argumento:

```bash
uv run python -m src.causal_train otra_carpeta
```

## Uso como modulos

```python
from src.causal_llm import CausalLLM
from src.causal_train import train
from src.corpus import load_corpus
from src.tokenizer import BPETokenizer
```

## Estado

- Estructura migrada al esquema de clase (`transformer` + `causal_llm` + `causal_train`).
- Se mantiene `src/__init__.py` con exports para importar componentes comunes.
- El Transformer basico ya completa los huecos principales de la plantilla.

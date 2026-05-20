# P5 - LLM Causal con Transformer
**Grupo 12**  
**Bautista Pelossi Schweizer · Ignacio Ramirez Suarez**  

Practica de PLN para implementar un modelo autoregresivo completo:
tokenizacion BPE, autoatencion multi-cabezal, backbone Transformer,
cabeza causal y entrenamiento sobre corpus de texto.


## Estructura

```text
p5/
├── corpus/
│   ├── alice_in_wonderland.txt
│   └── looking_glass.txt
├── data_ner/
│   └── corpus_tag.json
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
5. Se genera texto con `model.generate(...)`, opcionalmente restringiendo el
   muestreo con `top_k`.

## Instalacion

```bash
uv sync
```

## Dataset NER

El dataset etiquetado debe estar en `data_ner/corpus_tag.json` con el formato
`tokens`/`labels`. Las etiquetas esperadas son:

- `pi`: persona inicio
- `pc`: persona continuacion
- `li`: lugar inicio
- `lc`: lugar continuacion
- `o`: otro

## Ejecucion

Ejecutar el entrenamiento del LLM (guarda `p5_causal_2612.pth`):

```bash
uv run fdi-pln-2612-p5 train-llm --corpus corpus
```

Generar texto desde un prompt:

```bash
uv run fdi-pln-2612-p5 generate \
  --weights p5_causal_2612.pth \
  --prompt "alice and the cat were studying for the exam. "
```

Entrenar NER con datos etiquetados:

```bash
uv run fdi-pln-2612-p5 train-ner \
  --data data_ner/corpus_tag.json \
  --llm-weights p5_causal_2612.pth \
  --out p5_ner_2612.pth
```

Predecir entidades desde un archivo de texto:

```bash
uv run fdi-pln-2612-p5 predict-ner \
  --weights p5_ner_2612.pth \
  --input ruta/al/texto.txt
```

Verificar formato:

```bash
uv format --check
```

Prueba pequena recomendada para comprobar que todo funciona sin exigir mucha
CPU/RAM:

```bash
uv run fdi-pln-2612-p5 train-llm --corpus corpus \
  --max-chars 5000 \
  --max-tokens 512 \
  --vocab-size 80 \
  --context-size 32 \
  --d-model 32 \
  --n-heads 4 \
  --n-layers 1 \
  --expansion 2 \
  --dropout 0.0 \
  --epochs 1 \
  --batch-size 8 \
  --max-new-tokens 30 \
  --top-k 10
```

Los hiperparametros principales se pueden ajustar desde la CLI con opciones
como `--epochs`, `--batch-size`, `--d-model`, `--n-layers`, `--context-size`,
`--vocab-size`, `--max-chars` y `--max-tokens`.

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
- `generate` soporta `top_k` para limitar el muestreo a los tokens mas probables.

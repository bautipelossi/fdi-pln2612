# P5 — LLM Causal con Transformer

**Grupo 12** · Bautista Pelossi Schweizer · Ignacio Ramirez Suarez

Implementación de un modelo de lenguaje autoregresivo: tokenizador BPE, autoatención multi-cabezal con máscara causal, backbone Transformer con bloques pre-norma y conexiones residuales, y cabeza NER para reconocimiento de entidades.

---

## Índice

1. [Arquitectura](#arquitectura)
2. [Estructura del proyecto](#estructura-del-proyecto)
3. [Instalación](#instalación)
4. [Ejecución — LLM](#Ejecución--llm)
5. [Ejecución — NER](#Ejecución--ner)
6. [Hiperparámetros](#hiperparámetros)
7. [Prueba rápida (smoke test)](#prueba-rápida-smoke-test)
8. [Dataset NER](#dataset-ner)
9. [Ejecución como módulos Python](#Ejecución-como-módulos-python)

---

## Arquitectura

```
Texto crudo
    │
    ▼
BPETokenizer          ← vocabulario de caracteres + merges aprendidos
    │  encode / decode
    ▼
Token IDs  ─────────────────────────────────────────────────┐
    │                                                        │
    ▼                                                    (targets)
Transformer                                                  │
  ├─ tok_emb   (vocab_size → d_model)                        │
  ├─ pos_emb   (max_seq_len → d_model)                       │
  └─ N × Block                                               │
       ├─ LayerNorm                                          │
       ├─ MultiHeadAttention (causal mask)                   │
       ├─ LayerNorm                                          │
       └─ FeedForward (GELU, expansion × d_model)            │
    │                                                        │
    ▼                                                        │
Hidden states  / Capa Oculta                                 │
    │                                                        │
    ├──► lm_head → logits → cross_entropy ◄─────────────────┘
    │              (weight tying con tok_emb)
    │
    └──► NER head → logits por token → cross_entropy (pi/pc/li/lc/o)
```

**Decisiones de diseño:**

- **Weight tying**: los pesos de `tok_emb` y `lm_head` son los mismos, lo que reduce parámetros y mejora la generalización.
- **Pre-norm**: la normalización se aplica antes de cada sub-capa (`x + attn(norm(x))`), lo que estabiliza el entrenamiento.
- **Máscara causal**: triángulo superior de `-inf` en la matriz de atención; garantiza que cada token solo atiende a posiciones anteriores.
- **Generación con temperatura y top-k**: la función `generate` aplica temperatura para modular la distribución/aleatoriedad y opcionalmente restringe el muestreo a los `k` tokens más probables.
- **Alineamiento BPE → NER**: `align_to_bpe` propaga las etiquetas de nivel palabra a sub-tokens, convirtiendo la frontera `pi`/`li` al primer sub-token y los siguientes en `pc`/`lc`.

---

## Estructura del proyecto

```
p5/
├── corpus/
│   ├── alice_in_wonderland.txt
│   └── looking_glass.txt
├── data_ner/
│   └── corpus_tag.json          ← dataset anotado (tokens + labels)
├── src/
│   ├── tokenizer.py             ← BPETokenizer (train / encode / decode)
│   ├── corpus.py                ← carga de corpus y construcción de batches
│   ├── attention.py             ← MultiHeadAttention con máscara causal
│   ├── transformer.py           ← FeedForward, Block, Transformer backbone
│   ├── causal_llm.py            ← CausalLLM (lm_head, generate)
│   ├── causal_train.py          ← loop de entrenamiento y validación
│   ├── ner.py                   ← NERLLM, dataset, alineamiento y train NER
│   ├── cli.py                   ← CLI (Typer): train-llm, generate, train-ner, predict-ner
│   └── __init__.py
├── pyproject.toml
└── README.md
```

---

## Instalación

Requiere Python ≥ 3.11, < 3.13 y [`uv`](https://github.com/astral-sh/uv).

```bash
uv sync
```

Verificar que el CLI está disponible:

```bash
uv run fdi-pln-2612-p5 --help
```

> El proyecto usa PyTorch CPU por defecto (configurado en `pyproject.toml`). Para GPU, reemplazar el índice de torch en `pyproject.toml`.

---

## Ejecución — LLM

### 1. Entrenar el modelo de lenguaje

Lee todos los `.txt` de la carpeta indicada, entrena el tokenizador BPE y ajusta el modelo. Guarda pesos en `p5_causal_2612.pth`.

```bash
uv run fdi-pln-2612-p5 train-llm corpus \
  --vocab-size 300 \
  --context-size 64 \
  --d-model 128 \
  --n-heads 2 \
  --n-layers 4 \
  --expansion 4 \
  --dropout 0.1 \
  --epochs 4 \
  --batch-size 64 \
  --lr 3e-4 \
  --out p5_causal_2612.pth
```

### 2. Generar texto

```bash
uv run fdi-pln-2612-p5 generate \
  --weights p5_causal_2612.pth \
  --prompt "alice was beginning to " \
  --max-new-tokens 100 \
  --top-k 40
```

---

## Ejecución — NER

Las etiquetas reconocidas siguen el esquema BIO adaptado:

| Etiqueta | Significado |
|---|---|
| `o` | otro (no entidad) |
| `pi` | inicio de persona |
| `pc` | continuación de persona |
| `li` | inicio de lugar |
| `lc` | continuación de lugar |

### 1. Entrenar el modelo NER

Requiere pesos del LLM entrenado previamente. Guarda en `p5_ner_2612.pth`.

```bash
uv run fdi-pln-2612-p5 train-ner \
  --data data_ner/corpus_tag.json \
  --llm-weights p5_causal_2612.pth \
  --max-len 64 \
  --class-weight-power 0.2 \
  --class-weight-max 2.0 \
  --epochs 23 \
  --lr 1e-4 \
  --batch-size 4 \
  --train-ratio 0.8 \
  --out p5_ner_2612.pth
```

> `--class-weight-power` y `--class-weight-max` controlan el balanceo de clases para compensar el desbalance natural entre `o` y las etiquetas de entidad.

**Modelo final** Seleccionado por mejor F1 micro-average (0.646) con balance entre Precision (0.516) y Recall (0.865). Entrenado sobre elLLM base ajustando los pesos de clase. Pesos finales: `p5_ner_2612.pth`.

### 2. Predecir entidades

```bash
uv run fdi-pln-2612-p5 predict-ner \
  --weights p5_ner_2612.pth \
  --input ruta/al/texto.txt
```

---

## Hiperparámetros

| Opción CLI | Descripción | Valor final |
|---|---|---|
| `--vocab-size` | Tamaño del vocabulario BPE | 300 |
| `--context-size` | Longitud máxima de contexto (tokens) | 64 |
| `--d-model` | Dimensión de los embeddings | 128 |
| `--n-heads` | Número de cabezales de atención | 2 |
| `--n-layers` | Número de bloques Transformer | 4 |
| `--expansion` | Factor de expansión del FeedForward | 4 |
| `--dropout` | Tasa de dropout | 0.1 |
| `--epochs` | Épocas de entrenamiento | 4 |
| `--batch-size` | Tamaño del batch | 64 |
| `--lr` | Learning rate | 3e-4 |
| `--max-chars` | Límite de caracteres del corpus (debug) | — |
| `--max-tokens` | Límite de tokens tras tokenizar (debug) | — |

---

## Prueba rápida (smoke test)

Para verificar que el pipeline completo funciona sin un entrenamiento largo:

```bash
# Entrenar con modelo pequeño (corpus truncado, 1 época)
uv run fdi-pln-2612-p5 train-llm corpus \
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
  --out p5_causal_2612_small.pth

# Generar texto con el modelo pequeño
uv run fdi-pln-2612-p5 generate \
  --weights p5_causal_2612_small.pth \
  --prompt "alice was beginning to " \
  --max-new-tokens 30 \
  --top-k 10
```

---

## Dataset NER

El archivo `data_ner/corpus_tag.json` debe tener el formato:

```json
[
  {
    "tokens": ["Alice", "went", "to", "Wonderland"],
    "labels": ["pi",    "o",    "o",  "li"]
  },
  ...
]
```

---

## Ejecución como módulos Python

```python
from src.tokenizer import BPETokenizer
from src.corpus import load_corpus
from src.causal_llm import CausalLLM
from src.causal_train import train

# Cargar corpus y entrenar tokenizador
text = load_corpus("corpus/")
tok = BPETokenizer(text, vocab_size=300)
tokens = tok.encode(text)

# Instanciar y entrenar el modelo
model = CausalLLM(
    vocab_size=300, max_seq_len=64,
    d_model=128, n_heads=2, n_layers=4,
    expansion=4, dropout=0.1,
)
train(model, tokens, context_size=64, batch_size=64, epochs=4, lr=3e-4)

# Generar texto
prompt_ids = tok.encode("alice was ")
generated_ids = model.generate(prompt_ids, max_tokens=50, top_k=40)
print(tok.decode(generated_ids))
```

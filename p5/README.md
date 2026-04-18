# P5 - LLM basado en Transformer

Practica para implementar un LLM basado en Transformer: tokenizacion BPE, auto-atencion multi-cabezal, entrenamiento
autoregresivo y generacion de texto.

## Estado actual

El proyecto esta modularizado en `src/` y con carpeta `corpus/` para los datos.
Actualmente incluye:

- `src/tokenizer.py`: tokenizador BPE entrenado sobre el corpus.
- `src/attention.py`: capa de auto-atencion multi-cabezal con mascara causal.
- `src/data.py`: carga del corpus y creacion de batches.
- `src/model.py`: sitio de integracion de `TinyLLM`, entrenamiento y evaluacion.

Falta todavia integrar la implementacion concreta de `TinyLLM` dentro de
`src/model.py`.

## Estructura

```text
p5/
├── corpus/
│   └── *.txt
├── src/
│   ├── __init__.py
│   ├── attention.py
│   ├── data.py
│   ├── model.py
│   └── tokenizer.py
├── pyproject.toml
└── README.md
```

## Flujo del programa

1. `src/data.py` carga el corpus (`.txt`) de la carpeta `corpus/`.
2. `BPETokenizer` aprende un vocabulario a partir del texto.
3. El corpus se convierte en IDs de tokens.
4. `get_batch` crea pares `x` e `y`, donde `y` es `x` desplazado un token.
5. `TinyLLM` (en `src/model.py`) aprende a predecir el siguiente token.
6. Se estima `val_loss` periodicamente con `evaluate_loss`.
7. Al terminar, el modelo genera texto desde un prompt.

## Uso

Instalar dependencias:

```bash
uv sync
```

Crear/usar la carpeta `corpus/` con uno o varios `.txt`:

```text
corpus/
└── corpus.txt
```

Ejemplo de ejecucion desde Python:

```bash
uv run python -c "from types import SimpleNamespace; from src.model import train; train(SimpleNamespace(data_dir='corpus', vocab_size=300, d_model=128, n_heads=4, n_layers=2, max_seq_len=64, dropout=0.1, lr=3e-4, batch_size=16, steps=200, log_every=20, eval_steps=20, prompt='hola', max_new_tokens=80, temperature=0.9, top_k=40))"
```

Nota: este comando levanta el flujo de entrenamiento definido en `src/model.py`.

## Integracion de TinyLLM

Cuando este disponible el codigo del Transformer, el sitio natural para
integrarlo es `src/model.py`. Ese archivo debe exponer una clase `TinyLLM` con
esta interfaz:

```python
model = TinyLLM(
    vocab_size=...,
    d_model=...,
    n_heads=...,
    n_layers=...,
    max_seq_len=...,
    dropout=...,
)

logits, loss = model(x, y)
generated = model.generate(context, max_new_tokens=120, temperature=0.9, top_k=40)
```

`src/attention.py` queda como modulo reutilizable para los bloques Transformer.

## Estado para push

- Estructura del repo consistente con `src/` y `corpus/`.
- Carga de datos, tokenizacion y bucle de entrenamiento/evaluacion listos.
- Pendiente: implementar `TinyLLM` para poder entrenar y generar texto de extremo a extremo.

## Argumentos principales

- `--data-dir`: carpeta con archivos `.txt`.
- `--vocab-size`: tamano maximo del vocabulario BPE.
- `--d-model`: dimension de embeddings y estados internos.
- `--n-heads`: numero de cabezales de atencion.
- `--n-layers`: numero de bloques Transformer.
- `--max-seq-len`: longitud maxima de contexto.
- `--batch-size`: numero de secuencias por batch.
- `--steps`: pasos de entrenamiento.
- `--prompt`: texto inicial para generar.
- `--max-new-tokens`: tokens nuevos a generar.
- `--eval-steps`: batches para estimar validacion.

## Notas

La ejecucion recomendada es siempre mediante `uv run`, para usar el entorno y
las dependencias declaradas en `pyproject.toml`. La ejecucion completa requiere
`torch`, un corpus en `corpus/` y la clase `TinyLLM` implementada en
`src/model.py`.

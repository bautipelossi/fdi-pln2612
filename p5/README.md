# P5 - Tiny LLM

Practica para construir un modelo de lenguaje pequeno desde varias piezas
basicas: tokenizacion BPE, auto-atencion multi-cabezal, entrenamiento
autoregresivo y generacion de texto.

## Estado actual

El repositorio esta preparado para integrar la implementacion del Transformer
cuando este disponible. Ahora mismo ya incluye:

- `tokenizer.py`: tokenizador BPE entrenado sobre el corpus.
- `attention.py`: capa de auto-atencion multi-cabezal con mascara causal.
- `data.py`: carga del corpus y creacion de batches.
- `train.py`: bucle de entrenamiento y generacion final.
- `main.py`: punto de entrada por linea de comandos.

Falta todavia:

- `model.py`: debe definir `TinyLLM`.
- `resources/`: carpeta con archivos `.txt` para entrenar.

## Estructura

```text
p5/
├── attention.py
├── data.py
├── main.py
├── tokenizer.py
├── train.py
├── pyproject.toml
├── README.md
└── resources/
    └── corpus.txt
```

## Flujo del programa

1. `main.py` lee los argumentos de ejecucion.
2. `train.py` carga el corpus usando `data.load_corpus`.
3. `BPETokenizer` aprende un vocabulario a partir del texto.
4. El corpus se convierte en IDs de tokens.
5. `data.get_batch` crea pares `x` e `y`, donde `y` es `x` desplazado un token.
6. `TinyLLM` aprende a predecir el siguiente token.
7. Al terminar, el modelo genera texto desde un prompt.

## Uso

Instalar dependencias:

```bash
uv sync
```

Crear una carpeta `resources/` con uno o varios `.txt`:

```text
resources/
└── corpus.txt
```

Ver opciones disponibles:

```bash
python3 main.py --help
```

Entrenar y generar:

```bash
uv run python main.py --data-dir resources --steps 200
```

## Integracion de `model.py`

Cuando este disponible el codigo del Transformer, el sitio natural para
integrarlo es `model.py`. Ese archivo deberia exponer una clase `TinyLLM` con
esta interfaz aproximada:

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

`attention.py` queda como modulo reutilizable para los bloques Transformer.

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

## Notas

La ejecucion completa requiere `torch`, un corpus en `resources/` y la clase
`TinyLLM` en `model.py`. Mientras `model.py` no exista, `main.py --help`
funciona, pero el entrenamiento mostrara un error indicando que falta esa
pieza.

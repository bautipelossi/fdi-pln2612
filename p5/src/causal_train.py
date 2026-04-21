import argparse
import time

import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    """Ventana deslizante sobre un tensor de tokens para language modeling.

    Cada sample es un par (x, y) de longitud `seq_len`, donde y es x
    desplazado una posicion a la derecha (predecir el siguiente token).
    """

    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y


def _make_dataloaders(tokens, context_size, batch_size, train_ratio=0.9):
    """Los dataloaders se encargan de ir aportando pares para el entrenamiento,
    incluyendo batching, mezcla aleatoria, etc."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio debe estar entre 0 y 1.")

    data = torch.tensor(tokens, dtype=torch.long)

    # Separamos datos en entrenamiento y validación
    split = int(train_ratio * len(data))
    if split <= context_size or len(data) - split <= context_size:
        raise ValueError(
            "No hay tokens suficientes para crear train/val con el context_size actual."
        )
    train_ds = TextDataset(data[:split], context_size)
    val_ds = TextDataset(data[split:], context_size)
    logger.info(f"Train: {len(train_ds):,} muestras, Val: {len(val_ds):,}")

    # Los dataloaders implementan utilidades para el entrenamiento de
    # modelos. Devolvemos uno para train y otro para val
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
    )


def _run_epoch(model, dataloader, optimizer=None):
    """Ejecuta una epoch completa de entrenamiento o evaluación.

    Si se pasa optimizer, entrena el modelo (forward + backward + step).
    Si no, evalúa sin calcular gradientes.
    Devuelve la media de loss sobre todos los batches.
    """
    total_loss, n = 0, 0
    device = next(model.parameters()).device

    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    with torch.set_grad_enabled(training):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            if training:
                optimizer.zero_grad()

            # Pase forward, creando el grafo computacional y calculando loss
            _, loss = model(x, y)

            if training:
                # Propaga la pérdida hacia atrás siguiendo el grafo
                loss.backward()
                # Reducimos "gradientes explosivos" para evitar anomalías de train
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # Hacemos un paso del optimizador (eg un pequeño paso de descenso
                # siguiendo el gradiente, o lo que determine el optimizador)
                optimizer.step()

            total_loss += loss.item()
            n += 1

    # Devolvemos la media de loss en este epoch
    return total_loss / n


def train(
    model,
    tokens,
    epochs=5,
    context_size=128,
    batch_size=64,
    lr=3e-4,
    train_ratio=0.9,
):
    """Entrena el modelo de lenguaje causal sobre los tokens dados.

    Realiza `epochs` épocas de entrenamiento con AdamW, registrando train/val
    loss en cada época.
    """

    train_dl, val_dl = _make_dataloaders(tokens, context_size, batch_size, train_ratio)

    # El optimizador ajusta los parámetros que le pasamos en función del
    # gradiente (calculado con forward y backward) y la tasa de aprendizaje
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    t0 = time.time()
    for epoch in range(epochs):
        train_loss = _run_epoch(model, train_dl, optimizer)
        val_loss = _run_epoch(model, val_dl, None)
        elapsed = time.time() - t0
        logger.info(
            f"Epoca {epoch + 1}/{epochs} | train={train_loss:.4f} | "
            f"val={val_loss:.4f} | tiempo={elapsed:.1f}s"
        )

    elapsed = time.time() - t0
    logger.info(f"Entrenamiento finalizado en {elapsed:.1f}s")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Entrena un LLM causal pequeno.")
    parser.add_argument(
        "corpus_dir",
        nargs="?",
        default="corpus",
        help="Carpeta con archivos .txt para entrenar.",
    )
    parser.add_argument("--vocab-size", type=int, default=300)
    parser.add_argument("--context-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Limita caracteres del corpus antes de tokenizar.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Limita tokens despues de tokenizar.",
    )
    parser.add_argument(
        "--prompt",
        default="alice and the cat were studying for the exam. what ",
    )
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None):
    from src.causal_llm import CausalLLM
    from src.corpus import load_corpus
    from src.tokenizer import BPETokenizer

    args = parse_args(argv)
    torch.manual_seed(args.seed)

    text = load_corpus(args.corpus_dir)
    if args.max_chars is not None:
        text = text[: args.max_chars]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = BPETokenizer(text, vocab_size=args.vocab_size)
    tokens = tokenizer.encode(text)
    if args.max_tokens is not None:
        tokens = tokens[: args.max_tokens]

    logger.info(
        "Config | device={} | chars={} | tokens={} | vocab={} | d_model={} | "
        "layers={} | heads={} | context={} | batch={} | epochs={}",
        device,
        len(text),
        len(tokens),
        len(tokenizer.vocab),
        args.d_model,
        args.n_layers,
        args.n_heads,
        args.context_size,
        args.batch_size,
        args.epochs,
    )

    model = CausalLLM(
        vocab_size=len(tokenizer.vocab),
        max_seq_len=args.context_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        expansion=args.expansion,
        dropout=args.dropout,
    ).to(device)

    logger.info(
        "Parametros entrenables: {:,}", sum(p.numel() for p in model.parameters())
    )
    train(
        model,
        tokens,
        epochs=args.epochs,
        context_size=args.context_size,
        batch_size=args.batch_size,
        lr=args.lr,
        train_ratio=args.train_ratio,
    )

    pred = model.generate(
        tokenizer.encode(args.prompt),
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    logger.opt(colors=True).info(
        f"<cyan>{args.prompt}</cyan>{tokenizer.decode(pred)[:500]}"
    )


if __name__ == "__main__":
    main()

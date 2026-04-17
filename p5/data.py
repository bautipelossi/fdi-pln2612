from pathlib import Path

import torch


def load_corpus(data_dir):
    """Carga todos los .txt de una carpeta y los concatena."""
    files = sorted(Path(data_dir).glob("*.txt"))
    if not files:
        raise ValueError(f"No hay archivos .txt en: {Path(data_dir).resolve()}")
    return "\n\n".join(path.read_text(encoding="utf-8") for path in files)


def get_batch(data, batch_size, block_size, device):
    """Devuelve pares x/y para predecir el siguiente token."""
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y

"""Modelo NER, dataset y alineamiento tokenizado -> BPE.

El flujo de datos:

    palabras anotadas (word-level, pi/pc/li/lc/o)
        |
        |  align_to_bpe()       <- tokeniza el texto
        v
    sub-tokens + etiquetas pi/pc/li/lc/o
        |
        |  NERDataset + collate_ner
        v
  batches (ids, labels) listos para cross_entropy
        |
        v
  NERLLM (Transformer + cabeza lineal por token)
"""

import json
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, Dataset

from src.transformer import Transformer

# Etiquetas NER en esquema pi/pc/li/lc/o
LABEL2ID = {"o": 0, "pi": 1, "pc": 2, "li": 3, "lc": 4}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)


def align_to_bpe(words, word_labels, tokenizer):
    """Alinea etiquetas pi/pc/li/lc/o de palabras a sub-tokens de BPE.

    Como el tokenizador BPE puede partir una palabra en varios sub-tokens,
    hay que decidir que etiqueta dar a cada trozo. Regla: la B- se queda
    en el primer sub-token y los siguientes son I-; asi la cabeza marca
    la frontera de la entidad a nivel palabra.

        palabra 'alice' con etiqueta pi, BPE la parte en ['al', 'ice']
            -> al: pi, ice: pc
        palabra 'cheshire' con pc, BPE en ['ch', 'es', 'h', 'ire']
            -> todos pc
        palabra 'wonderland' con li, BPE en ['won', 'der', 'land']
            -> won: li, der: lc, land: lc
        palabras o -> todos sus sub-tokens o
      espacios entre palabras -> O

    Devuelve (token_ids, token_labels) con etiquetas como strings.
    """
    token_ids = []
    token_labels = []
    space_ids = tokenizer.encode(" ")
    for i, (word, label) in enumerate(zip(words, word_labels)):
        if i > 0:
            token_ids.extend(space_ids)
            token_labels.extend(["o"] * len(space_ids))
        word_ids = tokenizer.encode(word)
        token_ids.extend(word_ids)
        if label == "pi":
            token_labels.append("pi")
            token_labels.extend(["pc"] * (len(word_ids) - 1))
        elif label == "li":
            token_labels.append("li")
            token_labels.extend(["lc"] * (len(word_ids) - 1))
        else:
            token_labels.extend([label] * len(word_ids))
    return token_ids, token_labels


def align_tokens_to_bpe(tokens, token_labels, tokenizer):
    """Alinea etiquetas pi/pc/li/lc/o a sub-tokens BPE para tokens pre-segmentados."""
    token_ids = []
    token_out_labels = []
    for token, label in zip(tokens, token_labels):
        sub_ids = tokenizer.encode(token)
        token_ids.extend(sub_ids)
        if not sub_ids:
            continue
        if label == "pi":
            token_out_labels.append("pi")
            token_out_labels.extend(["pc"] * (len(sub_ids) - 1))
        elif label == "li":
            token_out_labels.append("li")
            token_out_labels.extend(["lc"] * (len(sub_ids) - 1))
        else:
            token_out_labels.extend([label] * len(sub_ids))
    return token_ids, token_out_labels


def _normalize_label(label):
    if label is None:
        return "o"
    if label in LABEL2ID:
        return label
    lowered = label.lower()
    if lowered in LABEL2ID:
        return lowered
    raise ValueError(f"Etiqueta NER desconocida: {label}")


def load_ner_json(path):
    """Carga datos etiquetados desde merged.json.

    Devuelve lista de (tokens, labels) con etiquetas pi/pc/li/lc/o.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = []
    for item in data:
        tokens = item.get("tokens")
        labels = item.get("labels")
        if tokens is None or labels is None:
            raise ValueError("Cada entrada debe tener 'tokens' y 'labels'.")
        if len(tokens) != len(labels):
            raise ValueError("tokens y labels deben tener la misma longitud.")
        norm_labels = [_normalize_label(lab) for lab in labels]
        samples.append((tokens, norm_labels))
    return samples


def tokenize_for_ner(text):
    """Tokenizacion simple que preserva espacios y puntuacion."""
    pattern = re.compile(r"\s+|\w+|[^\w\s]", re.UNICODE)
    return pattern.findall(text)


def explain_alignment(words, word_labels, tokenizer):
    """Imprime el alineamiento palabra -> sub-tokens BPE para una frase."""
    print(f"  frase: {' '.join(words)}")
    for word, label in zip(words, word_labels):
        ids = tokenizer.encode(word)
        pieces = [tokenizer.decode([i]) for i in ids]
        if label == "pi":
            labs = ["pi"] + ["pc"] * (len(ids) - 1)
        elif label == "li":
            labs = ["li"] + ["lc"] * (len(ids) - 1)
        else:
            labs = [label] * len(ids)
        pairs = "  ".join(f"{p}/{l}" for p, l in zip(pieces, labs))
        print(f"    {word:<15} {label:<6} -> {pairs}")


class NERLLM(Transformer):
    """Transformer con cabeza de clasificación por token para NER.

    Extiende Transformer añadiendo una cabeza lineal que asigna etiquetas
    pi/pc/li/lc/o a cada token. Usa atención bidireccional (causal=False):
    para etiquetar un token podemos mirar el contexto a derecha e izquierda.

    Los pesos del backbone se deben inicializar desde un CausalLLM pre-entrenado
    con load_state_dict(strict=False), que ignora las diferencias en las cabezas
    (lm_head vs ner_head) y transfiere solo el backbone compartido.
    """

    def __init__(
        self,
        vocab_size,
        max_seq_len,
        d_model,
        n_heads,
        n_layers,
        expansion,
        dropout,
        num_labels,
    ):
        super().__init__(
            vocab_size, max_seq_len, d_model, n_heads, n_layers, expansion, dropout
        )
        # El transformer ya tiene una representación suficientemente rica,
        # no tenemos más que proyectarla al espacio de etiquetas
        self.ner_head = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, labels=None):
        hidden = super().forward(input_ids, causal=False)
        logits = self.ner_head(hidden)
        loss = None
        if labels is not None:
            # cross_entropy espera logits 2D: para cada elemento, una
            # probabilidad por etiqueta.
            # Aplanamos batch y secuencia y tratamos cada token como una muestra
            # independiente:
            #   logits  (n_batches, n_tokens, num_labels) -> (n_batches*n_tokens, num_labels)
            #   labels  (n_batches, n_tokens)             -> (n_batches*n_tokens,)
            # Las posiciones de padding llevan -100 e ignore_index las descarta.
            flat_logits = logits.flatten(0, 1)
            flat_labels = labels.flatten()
            loss = cross_entropy(flat_logits, flat_labels, ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def predict_entities(self, words, tokenizer):
        """Predice entidades sobre una lista de **palabras**.

        Codifica la frase con `align_to_bpe` (etiquetas ficticias o), corre el
        modelo y agrupa pi/pc y li/lc consecutivos en entidades.

        Devuelve las entidades nombradas ya compuestas [(texto, tipo), ...].
        """
        self.eval()
        ids, _ = align_to_bpe(words, ["o"] * len(words), tokenizer)
        device = next(self.parameters()).device
        logits, _ = self(torch.tensor([ids], device=device))
        pred_labels = [ID2LABEL[p] for p in logits.argmax(-1)[0].tolist()]

        entities = []
        i = 0
        while i < len(ids):
            label = pred_labels[i]
            if label in ("pi", "li"):
                cont = "pc" if label == "pi" else "lc"
                kind = "pi" if label == "pi" else "li"
                j = i + 1
                while j < len(ids) and pred_labels[j] == cont:
                    j += 1
                text = tokenizer.decode(ids[i:j]).strip()
                if text:
                    entities.append((text, kind))
                i = j
            else:
                i += 1
        return entities

    @torch.no_grad()
    def predict_entities_from_tokens(self, tokens, tokenizer):
        """Predice entidades sobre tokens pre-segmentados."""
        self.eval()
        ids, _ = align_tokens_to_bpe(tokens, ["o"] * len(tokens), tokenizer)
        device = next(self.parameters()).device
        logits, _ = self(torch.tensor([ids], device=device))
        pred_labels = [ID2LABEL[p] for p in logits.argmax(-1)[0].tolist()]

        entities = []
        i = 0
        while i < len(ids):
            label = pred_labels[i]
            if label in ("pi", "li"):
                cont = "pc" if label == "pi" else "lc"
                kind = "pi" if label == "pi" else "li"
                j = i + 1
                while j < len(ids) and pred_labels[j] == cont:
                    j += 1
                text = tokenizer.decode(ids[i:j]).strip()
                if text:
                    entities.append((text, kind))
                i = j
            else:
                i += 1
        return entities


class NERDataset(Dataset):
    """Dataset de NER: aplica `align_to_bpe` a cada frase y convierte a tensores.

    `ner_data` es una lista de pares (words, labels), donde words es la lista
    de palabras de una frase y labels las etiquetas pi/pc/li/lc/o alineadas.
    """

    def __init__(self, ner_data, tokenizer, max_len=128):
        self.samples = []
        for words, labels in ner_data:
            ids, labs = align_to_bpe(words, labels, tokenizer)
            ids = ids[:max_len]
            labs = labs[:max_len]
            self.samples.append(
                (
                    torch.tensor(ids, dtype=torch.long),
                    torch.tensor([LABEL2ID[l] for l in labs], dtype=torch.long),
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class NERTokenDataset(Dataset):
    """Dataset de NER a partir de tokens pre-segmentados (merged.json)."""

    def __init__(self, ner_data, tokenizer, max_len=256):
        self.samples = []
        for tokens, labels in ner_data:
            ids, labs = align_tokens_to_bpe(tokens, labels, tokenizer)
            ids = ids[:max_len]
            labs = labs[:max_len]
            self.samples.append(
                (
                    torch.tensor(ids, dtype=torch.long),
                    torch.tensor([LABEL2ID[l] for l in labs], dtype=torch.long),
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_ner(batch):
    """Padding al largo maximo del batch. Las posiciones de padding usan -100
    en las etiquetas para que cross_entropy las ignore (no son tokens reales)."""
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)
    padded_x = torch.zeros(len(xs), max_len, dtype=torch.long)
    padded_y = torch.full((len(ys), max_len), -100, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        padded_x[i, : len(x)] = x
        padded_y[i, : len(y)] = y
    return padded_x, padded_y


def _split_dataset(dataset, train_ratio):
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio debe estar entre 0 y 1.")
    n = len(dataset)
    split = int(train_ratio * n)
    if split <= 0 or split >= n:
        raise ValueError("train_ratio produce split invalido.")
    train_ds = torch.utils.data.Subset(dataset, range(0, split))
    val_ds = torch.utils.data.Subset(dataset, range(split, n))
    return train_ds, val_ds


def _run_epoch(model, dataloader, optimizer=None):
    total_loss, n = 0.0, 0
    device = next(model.parameters()).device
    training = optimizer is not None
    model.train() if training else model.eval()

    with torch.set_grad_enabled(training):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            if training:
                optimizer.zero_grad()
            _, loss = model(x, y)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item()
            n += 1
    return total_loss / max(n, 1)


def train_ner(
    model,
    dataset,
    epochs=4,
    batch_size=32,
    lr=3e-4,
    train_ratio=0.9,
):
    train_ds, val_ds = _split_dataset(dataset, train_ratio)
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_ner
    )
    val_dl = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_ner)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    t0 = time.time()
    for epoch in range(epochs):
        train_loss = _run_epoch(model, train_dl, optimizer)
        val_loss = _run_epoch(model, val_dl)
        elapsed = time.time() - t0
        logger.info(
            "Epoca {}/{} | train={:.4f} | val={:.4f} | tiempo={:.1f}s",
            epoch + 1,
            epochs,
            train_loss,
            val_loss,
            elapsed,
        )
    logger.info("Entrenamiento NER finalizado en {:.1f}s", time.time() - t0)

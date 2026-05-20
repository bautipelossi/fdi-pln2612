from pathlib import Path
from typing import Optional

import torch
import typer
from loguru import logger

from src.causal_llm import CausalLLM
from src.causal_train import train as train_llm
from src.corpus import load_corpus
from src.ner import (
    LABEL2ID,
    NERLLM,
    NERTokenDataset,
    load_ner_json,
    tokenize_for_ner,
    train_ner,
)
from src.tokenizer import BPETokenizer

app = typer.Typer(help="CLI para P5: LLM causal y NER.")


def _save_bundle(path, model, tokenizer, config, extra=None):
    bundle = {
        "model_state": model.state_dict(),
        "tokenizer": tokenizer.to_state(),
        "config": config,
    }
    if extra:
        bundle.update(extra)
    torch.save(bundle, path)


def _load_bundle(path):
    return torch.load(path, map_location="cpu")


def _build_llm_from_config(config):
    return CausalLLM(
        vocab_size=config["vocab_size"],
        max_seq_len=config["max_seq_len"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        expansion=config["expansion"],
        dropout=config["dropout"],
    )


def _build_ner_from_config(config):
    return NERLLM(
        vocab_size=config["vocab_size"],
        max_seq_len=config["max_seq_len"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        expansion=config["expansion"],
        dropout=config["dropout"],
        num_labels=len(LABEL2ID),
    )


@app.command("train-llm")
def train_llm_command(
    corpus_dir: Path = typer.Argument(Path("corpus")),
    out: Path = typer.Option(Path("p5_causal_2612.pth"), "--out"),
    vocab_size: int = typer.Option(300, "--vocab-size"),
    context_size: int = typer.Option(128, "--context-size"),
    d_model: int = typer.Option(128, "--d-model"),
    n_heads: int = typer.Option(2, "--n-heads"),
    n_layers: int = typer.Option(4, "--n-layers"),
    expansion: int = typer.Option(4, "--expansion"),
    dropout: float = typer.Option(0.1, "--dropout"),
    epochs: int = typer.Option(4, "--epochs"),
    batch_size: int = typer.Option(64, "--batch-size"),
    lr: float = typer.Option(3e-4, "--lr"),
    train_ratio: float = typer.Option(0.9, "--train-ratio"),
    max_chars: Optional[int] = typer.Option(None, "--max-chars"),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens"),
    seed: int = typer.Option(0, "--seed"),
):
    """Entrena el LLM y guarda pesos + tokenizer."""
    torch.manual_seed(seed)
    text = load_corpus(corpus_dir)
    if max_chars is not None:
        text = text[:max_chars]

    tokenizer = BPETokenizer(text, vocab_size=vocab_size)
    tokens = tokenizer.encode(text)
    if max_tokens is not None:
        tokens = tokens[:max_tokens]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CausalLLM(
        vocab_size=len(tokenizer.vocab),
        max_seq_len=context_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        expansion=expansion,
        dropout=dropout,
    ).to(device)

    logger.info(
        "Config | device={} | chars={} | tokens={} | vocab={} | d_model={} | "
        "layers={} | heads={} | context={} | batch={} | epochs={}",
        device,
        len(text),
        len(tokens),
        len(tokenizer.vocab),
        d_model,
        n_layers,
        n_heads,
        context_size,
        batch_size,
        epochs,
    )

    train_llm(
        model,
        tokens,
        epochs=epochs,
        context_size=context_size,
        batch_size=batch_size,
        lr=lr,
        train_ratio=train_ratio,
    )

    config = {
        "vocab_size": len(tokenizer.vocab),
        "max_seq_len": context_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "expansion": expansion,
        "dropout": dropout,
    }
    _save_bundle(out, model, tokenizer, config)
    logger.info("Pesos guardados en {}", out)


@app.command("generate")
def generate_command(
    weights: Path = typer.Option(..., "--weights"),
    prompt: str = typer.Option(..., "--prompt"),
    max_new_tokens: int = typer.Option(200, "--max-new-tokens"),
    temperature: float = typer.Option(0.8, "--temperature"),
    top_k: Optional[int] = typer.Option(None, "--top-k"),
):
    """Genera texto desde un prompt usando pesos entrenados."""
    bundle = _load_bundle(weights)
    config = bundle["config"]
    tokenizer = BPETokenizer.from_state(bundle["tokenizer"])
    model = _build_llm_from_config(config)
    model.load_state_dict(bundle["model_state"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    ids = tokenizer.encode(prompt)
    generated = model.generate(
        ids, max_tokens=max_new_tokens, temperature=temperature, top_k=top_k
    )
    text = prompt + tokenizer.decode(generated)
    typer.echo(text)


@app.command("train-ner")
def train_ner_command(
    data: Path = typer.Option(Path("data_ner/corpus_tag.json"), "--data"),
    llm_weights: Path = typer.Option(..., "--llm-weights"),
    out: Path = typer.Option(Path("p5_ner_2612.pth"), "--out"),
    max_len: int = typer.Option(256, "--max-len"),
    epochs: int = typer.Option(4, "--epochs"),
    batch_size: int = typer.Option(32, "--batch-size"),
    lr: float = typer.Option(3e-4, "--lr"),
    train_ratio: float = typer.Option(0.9, "--train-ratio"),
    seed: int = typer.Option(0, "--seed"),
):
    """Entrena el cabezal NER a partir de datos etiquetados."""
    torch.manual_seed(seed)
    llm_bundle = _load_bundle(llm_weights)
    config = llm_bundle["config"]
    tokenizer = BPETokenizer.from_state(llm_bundle["tokenizer"])

    if max_len > config["max_seq_len"]:
        logger.warning(
            "max_len={} supera max_seq_len del modelo ({}), se ajusta.",
            max_len,
            config["max_seq_len"],
        )
        max_len = config["max_seq_len"]

    model = _build_ner_from_config(config)
    model.load_state_dict(llm_bundle["model_state"], strict=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    ner_data = load_ner_json(data)
    dataset = NERTokenDataset(ner_data, tokenizer, max_len=max_len)
    train_ner(
        model,
        dataset,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        train_ratio=train_ratio,
    )

    _save_bundle(out, model, tokenizer, config, extra={"label2id": LABEL2ID})
    logger.info("Pesos NER guardados en {}", out)


@app.command("predict-ner")
def predict_ner_command(
    weights: Path = typer.Option(..., "--weights"),
    input_path: Path = typer.Option(..., "--input"),
):
    """Predice entidades nombradas en un archivo de texto."""
    bundle = _load_bundle(weights)
    config = bundle["config"]
    tokenizer = BPETokenizer.from_state(bundle["tokenizer"])
    model = _build_ner_from_config(config)
    model.load_state_dict(bundle["model_state"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    text = Path(input_path).read_text(encoding="utf-8")
    tokens = tokenize_for_ner(text)
    if len(tokens) > config["max_seq_len"]:
        tokens = tokens[: config["max_seq_len"]]
    entities = model.predict_entities_from_tokens(tokens, tokenizer)

    for ent_text, ent_type in entities:
        typer.echo(f"{ent_text}\t{ent_type}")


if __name__ == "__main__":
    app()

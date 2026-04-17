import torch

from data import get_batch, load_corpus
from tokenizer import BPETokenizer


def _load_model_class():
    try:
        from model import TinyLLM
    except ModuleNotFoundError as exc:
        if exc.name == "model":
            raise ModuleNotFoundError(
                "Falta model.py con la clase TinyLLM. "
                "Ese es el sitio natural para integrar el Transformer del profesor."
            ) from exc
        raise
    return TinyLLM


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = load_corpus(args.data_dir)
    tokenizer = BPETokenizer(text, vocab_size=args.vocab_size)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    if len(ids) < args.max_seq_len + 2:
        raise ValueError("Corpus demasiado pequeno para entrenar con max_seq_len actual.")

    split = int(0.9 * len(ids))
    train_data = ids[:split]

    TinyLLM = _load_model_class()
    model = TinyLLM(
        vocab_size=len(tokenizer.vocab),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for step in range(1, args.steps + 1):
        xb, yb = get_batch(train_data, args.batch_size, args.max_seq_len, device)
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            print(f"step={step:5d} | loss={loss.item():.4f}")

    model.eval()
    prompt_ids = tokenizer.encode(args.prompt) if args.prompt else [0]
    context = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = model.generate(
        context,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    generated_text = tokenizer.decode(generated[0].tolist())
    print("\n=== TEXTO GENERADO ===")
    print(generated_text)

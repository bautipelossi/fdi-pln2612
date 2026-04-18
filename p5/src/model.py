import torch
import torch.nn as nn

from src.data import get_batch, load_corpus
from src.tokenizer import BPETokenizer


class TinyLLM(nn.Module):
    """Placeholder de TinyLLM para integrar el Transformer completo."""

    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len, dropout):
        super().__init__()
        raise NotImplementedError(
            "TinyLLM aun no esta implementado. Integra aqui el Transformer de la catedra."
        )

    def forward(self, x, y=None):
        raise NotImplementedError

    def generate(self, context, max_new_tokens, temperature=1.0, top_k=None):
        raise NotImplementedError


@torch.no_grad()
def evaluate_loss(model, data, batch_size, block_size, device, eval_steps=20):
    """Estimacion rapida de loss promedio para validacion."""
    model.eval()
    losses = []
    for _ in range(eval_steps):
        xb, yb = get_batch(data, batch_size, block_size, device)
        _, loss = model(xb, yb)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = load_corpus(args.data_dir)
    tokenizer = BPETokenizer(text, vocab_size=args.vocab_size)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    if len(ids) < args.max_seq_len + 2:
        raise ValueError("Corpus demasiado pequeno para entrenar con max_seq_len actual.")

    split = int(0.9 * len(ids))
    train_data = ids[:split]
    val_data = ids[split:]
    if len(val_data) <= args.max_seq_len + 1:
        val_data = train_data

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
            val_loss = evaluate_loss(
                model,
                val_data,
                args.batch_size,
                args.max_seq_len,
                device,
                eval_steps=args.eval_steps,
            )
            print(
                f"step={step:5d} | train_loss={loss.item():.4f} | val_loss={val_loss:.4f}"
            )

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

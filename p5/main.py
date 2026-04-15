import argparse
from pathlib import Path

import torch

from model import TinyLLM
from tokenizer import BPETokenizer


def load_corpus(data_dir):
    files = sorted(Path(data_dir).glob("*.txt"))
    if not files:
        raise ValueError(f"No hay archivos .txt en: {Path(data_dir).resolve()}")
    return "\n\n".join(p.read_text(encoding="utf-8") for p in files)


def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = load_corpus(args.data_dir)
    tokenizer = BPETokenizer(text, vocab_size=args.vocab_size)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    if len(ids) < args.max_seq_len + 2:
        raise ValueError("Corpus demasiado pequeno para entrenar con max_seq_len actual.")

    split = int(0.9 * len(ids))
    train_data = ids[:split]

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


def parse_args():
    parser = argparse.ArgumentParser(description="Entrena y prueba un Tiny LLM autoregresivo")
    parser.add_argument("--data-dir", default="resources", help="Carpeta con archivos .txt")
    parser.add_argument("--vocab-size", type=int, default=300)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--prompt", default="Alice was ")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

import argparse


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


def main():
    args = parse_args()
    from train import train

    train(args)


if __name__ == "__main__":
    main()

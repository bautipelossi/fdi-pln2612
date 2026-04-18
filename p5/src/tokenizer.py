from collections import Counter

class BPETokenizer:
    """Byte Pair Encoding entrenado sobre un texto.

    Vocabulario inicial: caracteres unicos del texto. Durante el
    entrenamiento se buscan los pares adyacentes mas frecuentes y se
    fusionan en nuevos tokens, hasta alcanzar `vocab_size` tokens.
    """

    def __init__(self, text, vocab_size=300):

        self.vocab_size = vocab_size
        # Inicializamos con caracteres encontrados en el texto
        self.vocab = sorted(set(text))  #generar un diccionario donde las claves son los chars y el primer elemento el ìndice y se le asigna a cada caracter un id
        self.tok2id = {tok: i for i, tok in enumerate(self.vocab)}

        tokens = [self.tok2id[c] for c in text] #tokenizo el texto
        self.merges = []  # lista de ((id_a, id_b), nuevo_id), para encode(), mergea los mas frecuentes

        for new_id in range(len(self.vocab), vocab_size):
            pairs = Counter(zip(tokens, tokens[1:]))
            if not pairs:
                break
            best = pairs.most_common(1)[0][0]

            new_tok = self.vocab[best[0]] + self.vocab[best[1]]
            self.tok2id[new_tok] = new_id
            self.vocab.append(new_tok)
            self.merges.append((best, new_id))

            tokens = self._apply_merge(tokens, best[0], best[1], new_id)

    @staticmethod
    def _apply_merge(tokens, a, b, new_id):
        """Reemplaza todas las ocurrencias del par (a, b) por new_id."""
        out = []
        i = 0
        n = len(tokens)
        while i < n:
            # Si encontramos el par exacto (a, b), lo fusionamos
            if i < n - 1 and tokens[i] == a and tokens[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        
        return out

    def encode(self, text):
        """Codifica un texto aplicando los merges aprendidos."""
        tokens = [self.tok2id.get(c, 0) for c in text]
        for (a, b), new_id in self.merges:
            tokens = self._apply_merge(tokens, a, b, new_id)
        return tokens

    def decode(self, ids):
        """Decodifica una lista de ids a texto."""
        out = []
        for i in ids:
            if i < 0 or i >= len(self.vocab):
                raise ValueError(f"id fuera de vocabulario: {i}")
            out.append(self.vocab[i])
        return "".join(out)
       

    def __repr__(self):
        pretty = [t.replace("\n", "\\n").replace(" ", "▁") for t in self.vocab]
        return f"{len(self.vocab)} tokens: ['{"', '".join(pretty)}']"

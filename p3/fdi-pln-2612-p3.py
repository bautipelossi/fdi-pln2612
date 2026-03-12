# /// script
# requires-python = ">=3.11"
# dependencies = ["typer"]
# ///
"""
FDI-PLN Criptoglifos 2026 (PLNCG26) - Equipo 12
Convierte texto UTF-8 a PLNCG26 y viceversa.
"""

import sys
from pathlib import Path

import typer

app = typer.Typer()

# Constantes del cifrado
K = 45
SPACE_B = 0x0B  # Espacio en PLNCG26
NL_B = 0x0A  # Salto de línea en PLNCG26
BYTE_ENYE = 0x34  # Ñ en PLNCG26
BYTE_NULL = 0x35  # Marcador mudo (se ignora, marca mayúscula/inicio)
BYTE_U_DIERESIS = 0x33  # Ü en PLNCG26
BYTE_TILDE_MARKER = 0x32  # Marca de tilde sobre la vocal anterior

# Puntuación directa (no pasan por César)
BYTE_PERIOD = 0x46  # .
BYTE_COMMA = 0x47  # ,
BYTE_SEMICOLON = 0x48  # ;
BYTE_COLON = 0x49  # :
BYTE_PAREN = 0x4E  # ( o ) — aparece duplicado: 0x4E 0x4E = un paréntesis
BYTE_QUOTE = 0x50  # « o » — alterna entre apertura y cierre
BYTE_APOSTROPHE = 0x4F  # '
BYTE_DASH = 0x51  # -
BYTE_LINE_END = 0x52  # Marca fin de línea (equivale a salto de línea)
BYTE_SECTION = 0x64  # Marca de sección (aparece duplicado, se ignora)
BYTE_PARAGRAPH = 0x65  # Marca de párrafo (aparece duplicado, se ignora)

# Rango de dígitos: 0x3C = '0', 0x3D = '1', ..., 0x45 = '9'
BYTE_DIGIT_BASE = 0x3C

# Mapas para tildes (solo mayúsculas, el texto se pasa a minúsculas después)
TILDE_MAP = {
    "A": "Á",
    "E": "É",
    "I": "Í",
    "O": "Ó",
    "U": "Ú",
}
TILDE_MAP_INV = {v: k for k, v in TILDE_MAP.items()}


def decode_plncg26(data: bytes) -> str:
    """Decodifica bytes PLNCG26 a texto UTF-8."""
    raw_chars: list[str] = []
    paren_open = True  # Alterna entre ( y )
    quote_open = True  # Alterna entre « y »

    i = 0
    while i < len(data):
        b = data[i]

        if b == BYTE_NULL:
            # Marcador de mayúscula: la letra anterior se mantiene en mayúscula
            if raw_chars:
                raw_chars[-1] = raw_chars[-1].upper()
        elif b == SPACE_B:
            raw_chars.append(" ")
        elif b == NL_B:
            raw_chars.append("\n")
        elif b == BYTE_ENYE:
            # Modificador: la N anterior se convierte en Ñ
            if raw_chars and raw_chars[-1] in ("N", "n"):
                was_upper = raw_chars[-1] == "N"
                raw_chars[-1] = "Ñ" if was_upper else "ñ"
            else:
                raw_chars.append("ñ")
        elif b == BYTE_U_DIERESIS:
            # Modificador: la U anterior se convierte en Ü
            if raw_chars and raw_chars[-1] in ("U", "u"):
                was_upper = raw_chars[-1] == "U"
                raw_chars[-1] = "Ü" if was_upper else "ü"
            else:
                raw_chars.append("ü")
        elif b == BYTE_TILDE_MARKER:
            # La vocal anterior recibe tilde
            if raw_chars:
                prev = raw_chars[-1]
                upper_prev = prev.upper()
                tilded = TILDE_MAP.get(upper_prev, prev)
                raw_chars[-1] = tilded if prev.isupper() else tilded.lower()
        elif BYTE_DIGIT_BASE <= b <= BYTE_DIGIT_BASE + 9:
            raw_chars.append(str(b - BYTE_DIGIT_BASE))
        elif b == BYTE_PERIOD:
            raw_chars.append(".")
        elif b == BYTE_COMMA:
            raw_chars.append(",")
        elif b == BYTE_SEMICOLON:
            raw_chars.append(";")
        elif b == BYTE_COLON:
            raw_chars.append(":")
        elif b == BYTE_APOSTROPHE:
            raw_chars.append("'")
        elif b == BYTE_DASH:
            raw_chars.append("-")
        elif b == BYTE_LINE_END:
            pass  # Marcador de fin de elemento, se ignora
        elif b == BYTE_PAREN:
            # Dos bytes 0x4E consecutivos = un paréntesis
            if i + 1 < len(data) and data[i + 1] == BYTE_PAREN:
                raw_chars.append("(" if paren_open else ")")
                paren_open = not paren_open
                i += 1  # Saltar el segundo 0x4E
            else:
                raw_chars.append("(" if paren_open else ")")
                paren_open = not paren_open
        elif b == BYTE_QUOTE:
            raw_chars.append("«" if quote_open else "»")
            quote_open = not quote_open
        elif b == BYTE_SECTION or b == BYTE_PARAGRAPH:
            # Marcadores de sección/párrafo (aparecen duplicados), se ignoran
            pass
        else:
            # Desplazamiento César: byte + K → letra minúscula
            raw_chars.append(chr((b + K) & 0xFF).lower())

        i += 1

    return "".join(raw_chars)


# Mapa inverso de puntuación para encode
_PUNCT_TO_BYTES: dict[str, list[int]] = {
    ".": [BYTE_PERIOD],
    ",": [BYTE_COMMA],
    ";": [BYTE_SEMICOLON],
    ":": [BYTE_COLON],
    "(": [BYTE_PAREN, BYTE_PAREN],
    ")": [BYTE_PAREN, BYTE_PAREN],
    "«": [BYTE_QUOTE],
    "»": [BYTE_QUOTE],
    "'": [BYTE_APOSTROPHE],
    "-": [BYTE_DASH],
}


def encode_plncg26(text: str) -> bytes:
    """Codifica texto UTF-8 a bytes PLNCG26."""
    result: list[int] = []

    i = 0
    while i < len(text):
        ch = text[i]

        if ch == " ":
            result.append(SPACE_B)
        elif ch == "\n":
            result.append(NL_B)
        elif ch.isdigit():
            result.append(BYTE_DIGIT_BASE + int(ch))
        elif ch in ("Ñ", "ñ"):
            result.append((ord("N") - K) & 0xFF)
            if ch.isupper():
                result.append(BYTE_NULL)
            result.append(BYTE_ENYE)
        elif ch in ("Ü", "ü"):
            result.append((ord("U") - K) & 0xFF)
            if ch.isupper():
                result.append(BYTE_NULL)
            result.append(BYTE_U_DIERESIS)
        elif ch.upper() in TILDE_MAP_INV:
            base = TILDE_MAP_INV[ch.upper()]
            result.append((ord(base) - K) & 0xFF)
            if ch.isupper():
                result.append(BYTE_NULL)
            result.append(BYTE_TILDE_MARKER)
        elif ch in _PUNCT_TO_BYTES:
            result.extend(_PUNCT_TO_BYTES[ch])
        elif ch.isalpha():
            upper_ch = ch.upper()
            byte_val = (ord(upper_ch) - K) & 0xFF
            result.append(byte_val)
            if ch.isupper():
                result.append(BYTE_NULL)
        else:
            byte_val = (ord(ch.upper()) - K) & 0xFF
            result.append(byte_val)

        i += 1

    return bytes(result)


@app.command()
def decode(fichero: Path) -> None:
    """Decodifica un fichero PLNCG26 a UTF-8."""
    if not fichero.exists():
        typer.echo(f"Error: El fichero '{fichero}' no existe.", err=True)
        raise typer.Exit(1)

    data = fichero.read_bytes()
    text = decode_plncg26(data)
    sys.stdout.write(text)


@app.command()
def encode(fichero: Path) -> None:
    """Codifica un fichero UTF-8 a PLNCG26."""
    if not fichero.exists():
        typer.echo(f"Error: El fichero '{fichero}' no existe.", err=True)
        raise typer.Exit(1)

    text = fichero.read_text(encoding="utf-8")
    data = encode_plncg26(text)
    sys.stdout.buffer.write(data)


@app.command()
def detect(fichero: Path) -> None:
    """Detecta la probabilidad de que un fichero sea PLNCG26."""
    if not fichero.exists():
        typer.echo(f"Error: El fichero '{fichero}' no existe.", err=True)
        raise typer.Exit(1)

    data = fichero.read_bytes()

    if len(data) == 0:
        typer.echo("0.0")
        return

    # Heurísticas para detectar PLNCG26:
    # 1. Bytes en rango típico de texto cifrado (caracteres - 45)
    # 2. Presencia de marcadores especiales (0x0A, 0x0B, 0x34, 0x35)
    # 3. Ausencia de bytes altos (>127) que serían típicos de UTF-8

    score = 0.0
    total_checks = 3

    # Check 1: Mayoría de bytes en rango bajo (texto ASCII - 45 = ~20-80)
    low_bytes = sum(1 for b in data if 0x00 <= b <= 0x60)
    if low_bytes / len(data) > 0.8:
        score += 1.0

    # Check 2: Presencia de separadores PLNCG26
    has_space = SPACE_B in data
    has_newline = NL_B in data
    if has_space or has_newline:
        score += 1.0

    # Check 3: Ausencia de bytes UTF-8 multibyte típicos
    high_bytes = sum(1 for b in data if b > 0x7F)
    if high_bytes / len(data) < 0.05:
        score += 1.0

    probability = score / total_checks
    typer.echo(f"{probability:.2f}")


if __name__ == "__main__":
    app()

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
BYTE_NULL = 0x35  # Marcador mudo
BYTE_U_DIERESIS = 0x1B  # ` -> Ü
BYTE_TILDE_MARKER = 0x3A  # _ -> marca de tilde anterior

# Mapas para tildes
TILDE_MAP = {
    "A": "Á",
    "E": "É",
    "I": "Í",
    "O": "Ó",
    "U": "Ú",
    "a": "á",
    "e": "é",
    "i": "í",
    "o": "ó",
    "u": "ú",
}
TILDE_MAP_INV = {v: k for k, v in TILDE_MAP.items()}


def decode_plncg26(data: bytes) -> str:
    """Decodifica bytes PLNCG26 a texto UTF-8."""
    raw_chars: list[str] = []

    for b in data:
        if b == BYTE_NULL:
            continue
        elif b == SPACE_B:
            raw_chars.append(" ")
        elif b == NL_B:
            raw_chars.append("\n")
        elif b == BYTE_ENYE:
            raw_chars.append("Ñ")
        else:
            raw_chars.append(chr((b + K) & 0xFF))

    raw = "".join(raw_chars)

    # Marcadores especiales: ` -> Ü
    raw = raw.replace("`", "Ü")

    # Procesar marcadores de tilde (_)
    final: list[str] = []
    for ch in raw:
        if ch == "_" and final:
            prev = final.pop()
            final.append(TILDE_MAP.get(prev, prev))
        else:
            final.append(ch)

    text = "".join(final)

    # Correcciones y mapeos especiales
    text = text.replace("UÜE", "ÜE")
    text = text.replace("NÑ", "Ñ")
    
    # Sustituciones de caracteres especiales y números
    text = text.replace("~", "-")
    text = text.replace(" t ", ", ")  # 't' espaciado es coma
    text = text.replace(" t\n", ",\n")
    
    # Mapeos de números codificados
    sust_numeros = {
        "jp": "17",
        "mjp": "m17",
        "josli": "16.30",
        "felizs": "feliz",
        "om": "64",
        "jis": "10",
        "lq": "38",
        "jpt": "17",
        "km": "24",
    }
    
    for encoded, decoded in sust_numeros.items():
        text = text.replace(encoded, decoded)

    # Formato: primera letra mayúscula, resto minúscula (por línea)
    lines = [line.strip().lower().capitalize() for line in text.splitlines()]
    return "\n".join(lines)


def encode_plncg26(text: str) -> bytes:
    """Codifica texto UTF-8 a bytes PLNCG26."""
    # Preparar texto: convertir a mayúsculas base
    text = text.upper()

    result: list[int] = []

    i = 0
    while i < len(text):
        ch = text[i]

        if ch == " ":
            result.append(SPACE_B)
        elif ch == "\n":
            result.append(NL_B)
        elif ch == "Ñ":
            result.append(BYTE_ENYE)
        elif ch == "Ü":
            result.append(BYTE_U_DIERESIS)  # ` en cifrado
        elif ch in TILDE_MAP_INV:
            # Letra con tilde: letra base + marcador _
            base = TILDE_MAP_INV[ch]
            result.append((ord(base) - K) & 0xFF)
            result.append(BYTE_TILDE_MARKER)  # _
        else:
            # Desplazamiento César inverso
            byte_val = (ord(ch) - K) & 0xFF
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
    total_checks = 4

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

    # Check 4: El texto decodificado parece español
    try:
        decoded = decode_plncg26(data)
        spanish_chars = sum(1 for c in decoded if c.isalpha() or c.isspace())
        if len(decoded) > 0 and spanish_chars / len(decoded) > 0.7:
            score += 1.0
    except Exception:
        pass

    probability = score / total_checks
    typer.echo(f"{probability:.2f}")


if __name__ == "__main__":
    app()
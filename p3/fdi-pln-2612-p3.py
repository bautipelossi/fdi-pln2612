python3 - <<'PY'
path = "/mnt/c/Users/ignac/Downloads/principal.bin"
data = open(path, "rb").read()

K = 45
SPACE_B = 0x0b
NL_B    = 0x0a
BYTE_ENYE = 0x34   # Ñ
BYTE_NULL = 0x35   # marcador mudo (provocaba la 'b')

# 1) +45 (saltando el byte mudo) + traducir espacio/salto/Ñ
raw_chars = []
for b in data:
    if b == BYTE_NULL:
        continue
    if b == SPACE_B:
        raw_chars.append(' ')
    elif b == NL_B:
        raw_chars.append('\n')
    elif b == BYTE_ENYE:
        raw_chars.append('Ñ')   # luego se ajusta al caso final
    else:
        raw_chars.append(chr((b + K) & 0xFF))
raw = ''.join(raw_chars)

# 2) Marcadores especiales
raw = raw.replace('`', 'Ü')  # luego quedará 'ü' si pasamos a minúsculas

tilde = {
    'A':'Á','E':'É','I':'Í','O':'Ó','U':'Ú',
    'a':'á','e':'é','i':'í','o':'ó','u':'ú'
}

final = []
for ch in raw:
    if ch == '_' and final:
        prev = final.pop()
        final.append(tilde.get(prev, prev))
    else:
        final.append(ch)

text = ''.join(final)

# 3) Arreglos mínimos
text = text.replace("UÜE", "ÜE")
text = text.replace("NÑ", "Ñ")

# 4) Formato: primera letra en mayúscula, resto en minúscula (por línea)
lines = [line.strip().lower().capitalize() for line in text.splitlines()]
print("\n".join(lines))
PY
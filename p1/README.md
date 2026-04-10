# Práctica 1 — Agente de Trueque Butler

**Grupo 12 · Procesamiento del Lenguaje Natural (2612)**
Bautista Pelossi Schweizer · Ignacio Ramírez Suarez

---

## Descripción

Agente autónomo que se conecta al servidor **Butler** y negocia trueques de recursos con otros agentes mediante un **LLM local** (Ollama).

El agente opera en un ciclo continuo:

1. Consulta su estado (`/info`) — recursos actuales, objetivo, buzón.
2. Procesa el buzón — acepta ofertas válidas y envía los recursos comprometidos.
3. Decide la siguiente acción mediante el LLM — aceptar una oferta, enviar una nueva, o esperar.
4. Ejecuta la decisión (enviar carta, enviar paquete, etc.).
5. Repite hasta cumplir el objetivo.

### Protocolo de comunicación

Los mensajes siguen un protocolo basado en tags estructurados:

| Tag | Uso | Ejemplo |
|---|---|---|
| `[OFERTA_V1]` | Proponer un trueque | `[OFERTA_V1] quiero={"aceite": 2} ofrezco={"madera": 3}` |
| `[ACEPTO_V1]` | Aceptar una oferta | `[ACEPTO_V1] te_envio={"madera": 3} espero={"aceite": 2}` |

---

## Arquitectura

```text
src/fdi_pln_2612_p1/
├── main.py          # Punto de entrada · ciclo principal del agente
├── config.py        # Variables de entorno y logging
├── models.py        # Modelos Pydantic (InfoPuesto, Decision)
├── butler_api.py    # Cliente HTTP para la API de Butler
├── http_client.py   # Wrapper con retry y backoff exponencial
├── protocol.py      # Construcción y parseo de mensajes [OFERTA/ACEPTO]
├── strategy.py      # Lógica de trueque, cooldowns anti-spam
└── llm.py           # Integración con Ollama · prompts · validación
```

| Módulo | Responsabilidad |
|---|---|
| `config.py` | Lee toda la configuración de variables de entorno (`FDI_PLN__*`). |
| `models.py` | Define `InfoPuesto` (compatible case-insensitive con la API) y `Decision`. |
| `butler_api.py` | Abstrae los endpoints de Butler (`/info`, `/alias`, `/carta`, `/paquete`, `/gente`). |
| `protocol.py` | Parsea y construye mensajes con regex sobre los tags `[OFERTA_V1]` y `[ACEPTO_V1]`. |
| `strategy.py` | Calcula faltantes, excedentes, cooldowns y procesa aceptaciones automáticas. |
| `llm.py` | Construye el prompt, invoca Ollama, valida y corrige la respuesta del modelo. |
| `main.py` | Orquesta el bucle: info → buzón → LLM → ejecución → sleep. |

---

## Requisitos

- **Python** ≥ 3.11
- **[uv](https://docs.astral.sh/uv/)** (gestor de paquetes)
- **[Ollama](https://ollama.com/)** corriendo localmente con el modelo `qwen2.5:3b`
- Acceso al servidor Butler (URL proporcionada por el profesor)

---

## Instalación

```bash
cd p1
uv sync
```

Descargar el modelo de Ollama (si no lo tenés):

```bash
ollama pull qwen2.5:3b
```

---

## Uso

### Variables de entorno

| Variable | Requerida | Default | Descripción |
|---|:---:|---|---|
| `FDI_PLN__BUTLER_ADDRESS` | **Sí** | — | URL del servidor Butler (`http://host:port`) |
| `FDI_PLN__ALIAS` | No | `fdi-pln-2612` | Nombre del agente en el mercado |
| `FDI_PLN__LLM_MODEL` | No | `qwen2.5:3b` | Modelo de Ollama a usar |
| `FDI_PLN__SLEEP_SECONDS` | No | `6` | Segundos entre ciclos |
| `FDI_PLN__LLM_TIMEOUT` | No | `45` | Timeout en segundos para la llamada al LLM |
| `FDI_PLN__DEBUG` | No | `1` | Mostrar logs detallados |
| `FDI_PLN__OFFER_COOLDOWN` | No | `25` | Cooldown (seg) entre ofertas al mismo destino |
| `FDI_PLN__OFFER_COOLDOWN_GLOBAL` | No | `5` | Cooldown (seg) global entre ofertas |

### Ejecución

Linux/macOS (bash/zsh):

```bash
export FDI_PLN__BUTLER_ADDRESS="http://IP_DEL_BUTLER:PUERTO"
uv run fdi-pln-2612-p1
```

Windows PowerShell:

```powershell
$env:FDI_PLN__BUTLER_ADDRESS = "http://IP_DEL_BUTLER:PUERTO"
uv run fdi-pln-2612-p1
```

Windows CMD:

```bat
set FDI_PLN__BUTLER_ADDRESS=http://IP_DEL_BUTLER:PUERTO
uv run fdi-pln-2612-p1
```

Nota: el ejecutable se lanza directamente. No usar subcomando `run`.

### Verificación previa a entrega

Desde la carpeta `p1`, comprobar:

```bash
uv format --check
uv build
uv run fdi-pln-2612-p1 --help
```

La ayuda debe mostrar:

```text
Usage: fdi-pln-2612-p1 [OPTIONS]
```

y no debe listar subcomandos.

### Prueba local (monopuesto)

Lanzar el servidor Butler en una terminal:

```bash
fdi-pln-butler server --monopuesto --buzon -p 7719
```

Lanzar dos agentes en terminales separadas:

```bash
# Terminal 2
FDI_PLN__BUTLER_ADDRESS=http://127.0.0.1:7719 FDI_PLN__ALIAS=agente-alpha FDI_PLN__SLEEP_SECONDS=8 uv run fdi-pln-2612-p1

# Terminal 3
FDI_PLN__BUTLER_ADDRESS=http://127.0.0.1:7719 FDI_PLN__ALIAS=agente-beta FDI_PLN__SLEEP_SECONDS=10 uv run fdi-pln-2612-p1
```

---

## Decisiones de diseño

- **LLM**: todas las decisiones de trading pasan por el modelo, con fallback por si el modelo falla.
- **Accept-first**: el prompt prioriza aceptar ofertas que nos benefician antes que crear nuevas.
- **Validación post-LLM**: si el modelo elige `esperar` u `ofertar` pero hay ofertas aceptables en el buzón, se fuerza una aceptación automática.
- **Auto-accept en timeout**: si el LLM no responde a tiempo y hay ofertas válidas, se acepta la mejor disponible.
- **Anti-spam**: cooldowns por destino y globales evitan saturar a otros agentes con ofertas repetidas.
- **Retry con backoff**: las llamadas HTTP usan reintentos con backoff exponencial (3 intentos).

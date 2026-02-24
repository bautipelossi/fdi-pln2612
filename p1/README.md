# Práctica 1 (P1) — Agente Butler (Trueque / Mercado)
**Grupo 12: Bautista Pelossi Schweizer e Ignacio Ramírez Suarez**

En esta prácica se implementa un **agente autónomo** que se conecta a Butler y participa en un **mercado dinámico** de trueques con otros agentes (grupos).

El agente:
- consulta el estado global,
- recibe mensajes en el buzón,
- envía ofertas y contraofertas,
- decide aceptar / rechazar / esperar,

---

## Estructura (P1)

```text
p1/
├── src/
│   └── fdi_pln_2612_p1/
│       ├── __init__.py
│       ├── main.py           
│       ├── config.py         
│       ├── butler_api.py     
│       ├── http_client.py    
│       ├── models.py         
│       ├── protocol.py       
│       ├── strategy.py       
│       └── llm.py            
├── pyproject.toml
└── uv.lock
```
## Requisitos
-Python 3.12+
-uv
-Acceso al Butler (variable de entorno obligatoria)

<a id="instalacion"></a>
##  Instalación

> La práctica P1 tiene su propio entorno y dependencias dentro de `p1/`.  
> Por eso **siempre** instalá desde esa carpeta (no desde la raíz del repo).

1) Entrar a la carpeta de la práctica:
```bash
cd p1
```

2) Instalar dependencias (crea/actualiza el entorno y respeta uv.lock):
```bash
uv sync
```

---

<a id="uso"></a>
##  Uso y Ejecución


### Paso 1: Ir a la p1/

```bash
cd p1
```
### Paso 2: Definir variables

```bash
export FDI_PLN__BUTLER_ADDRESS="http://IP_O_HOST_DEL_BUTLER:PORT"
export FDI_PLN__ALIAS="fdi-pln-2612"
export FDI_PLN__MODEL="mistral"
```
### Paso 3: Ejecutar
```bash
uv run python -m fdi_pln_2612_p1.main
```

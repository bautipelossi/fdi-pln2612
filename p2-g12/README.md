# PLN: Práctica 2

**Integrante A:** Ignacio Ramirez Suarez  
**Integrante B:** Bautista Pelossi Schweizer

**Pangrama objetivo (español)**: Un jugoso zumo de piña y kiwi bien frío es exquisito, y no lleva alcohol. 


## Informe Integrante B
### 1. Diagnóstico del Problema y exploración inicial
El objetivo inicial fue sintetizar el pangrama objetivo aislando fonemas del pangrama origen ("El veloz murciélago...") mediante una lectura intencionalmente lenta y silabeada. 

**Resultado:** Estrategia fallida (solo se pudo extraer limpiamente la palabra "jugoso").
**Posible explicación:** El habla no es una concatenación de unidades discretas. Al analizar las grabaciones en Praat, el espectrograma —que representa el sonido en tres dimensiones: Tiempo, Frecuencia y Energía— revela el problema de la **coarticulación**. Al hablar, la modulación de las frecuencias en el tracto vocal es continua. Leer "lento" deforma la envolvente natural del sonido y altera la transición de los formantes, haciendo que los cortes suenen sintéticos y llenos de artefactos (clics) al romper la onda.

### 2. Decisión Arquitectónica: Creación de un "Corpus" Personalizado
Ante la incompatibilidad fonética del pangrama original con la prosodia y las transiciones de mi dialecto, decidí diseñar una frase origen alternativa que contuviera los fonemas exactos que necesitaba, adaptados a mi fonética natural:
> *"Julián, zurdo frívolo, mezcló jungla de dedos: uña, ñandú, pipa; KIWI vino. Extra quimera: silla, tótem; noche llena, vaca al alma, copa Holanda."*

### 3. Ejecución Acústica en Praat
Para el proceso de "corta-pega", las decisiones de corte en el espectrograma se basaron en la naturaleza acústica de cada fonema vista en clase:

* **Vocales:** Identificables por sus ondas estables en el espacio. Los cortes se realizaron buscando los cruces por cero en la forma de onda para evitar saltos bruscos de fase, intentando empalmar vocales con formantes similares.
* **Consonantes Oclusivas (p, t, k, b, d, g):** Son los segmentos más críticos. Acústicamente se componen de un silencio (cierre del tracto) seguido de un *release* o explosión abrupta de energía. Para evitar que sonaran cortadas, fue vital incluir la fase de silencio previo al realizar el empalme, aunque a veces fue imposible.
* **Consonantes Fricativas (f, s):** Al estar formadas por ruido continuo, su representación en el espectrograma es ruido aperiódico en altas frecuencias. Fueron relativamente más eficientes al corte, siempre que se respetara su duración relativa frente a las vocales adyacentes.

### 4. Conclusiones del Paradigma
A pesar de la optimización y la grabación de una nueva frase, el resultado mantiene un tono "robótico". Esto demuestra por qué la síntesis por concatenación clásica tiene un límite infranqueable en la naturalidad: carece de un modelo prosódico global. En la comunicación humana, el cerebro del receptor predice patrones continuos. Al ensamblar fragmentos, rompemos la curva de entonación natural. Esta limitación evidencia por qué la industria de PLN (Sistemas TTS) ha migrado hacia modelos basados en Redes Neuronales como vimos en clases, que no pegan audios, sino que aprenden a generar el espectrograma desde cero prediciendo el contexto completo.

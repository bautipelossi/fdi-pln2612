# PLN: Práctica 2

**Integrante A:** Ignacio Ramirez Suarez  
**Integrante B:** Bautista Pelossi Schweizer


## Informe Integrante A
### 1. Planteamiento inicial y primeras hipótesis
Mi objetivo fue sintetizar los pangramas objetivos mediante concatenación (corta-pega) de segmentos extraídos del pangrama origen usando Praat. Al principio asumí que la tarea sería relativamente directa si aislaba bien los “sonidos” y los empalmaba con cuidado.

Una idea que tuve desde el inicio fue que en español (mi lengua materna) el proceso sería más sencillo porque la pronunciación es más estable y “plana”: los fonemas tienden a mantenerse más consistentes entre palabras. En cambio, en inglés ocurre lo contrario: una misma letra puede corresponder a sonidos muy diferentes dependiendo de la palabra, la posición y el acento, lo que complica la extracción “limpia” de fonemas.
En cuanto a los otros idiomas, tras realizar los audios sintéticos de inglés y español, me he dado cuenta de lo difícil que se me haría la tarea de formar los audios en idiomas que no conozco, sumandole tambíen que los audios originales grabados no tendrían una buena pronunciación, lo que dificultaría mucho el proceso.

### 2. Observaciones clave: inglés como dificultad… y como ventaja
Durante la exploración noté que el inglés, aunque más irregular, también ofrece un “margen de maniobra” interesante:

- Dificultad: hay mucha variación contextual (vocales reducidas, diptongos, aspiración, cambios por coarticulación). Esto hace que el mismo “sonido esperado” rara vez aparezca aislado y uniforme.

- Ventaja: esa variación permite reutilizar fragmentos de una palabra para construir otra cosa distinta. En la práctica, pude usar segmentos donde una letra suena “como otra” (por ejemplo, aprovechar fricativas, aproximantes o transiciones vocálicas que encajan mejor que el fonema “ideal” buscado). Es decir, no siempre se trata de “copiar el fonema exacto”, sino de encontrar el fragmento acústico que más se parezca al objetivo.

Esta parte fue especialmente evidente al trabajar con consonantes: algunas fricativas y transiciones entre sonidos resultaron más útiles que intentar extraer un fonema “puro”.

### 3. Dificultades reales en Praat: vocales y segmentación
Una conclusión importante fue que las vocales no fueron tan fáciles como esperaba, sobre todo cuando aparecen “solas” o en contextos muy abiertos:
Me resultó más fácil recortar y reutilizar una sílaba que una vocal suelta. Las razones principales:

La consonante aporta un “anclaje” acústico (cierre, explosión, fricción), la transición CV o VC incluye información natural de coarticulación, y el oído tolera mejor empalmes cuando hay estructura silábica.

En la edición intenté realizar cortes en zonas estables, cuando fue posible. Aun así, la coarticulación sigue imponiendo límites: el habla no se compone de piezas totalmente independientes.

### 4. Conclusiones sobre el método de concatenación
El resultado final (aunque funcional) conserva un carácter robótico y evidencia la limitación del paradigma:

La concatenación rompe la prosodia global (curva de entonación, ritmo, acento), que en el habla natural se mantiene coherente a lo largo de la frase.

Incluso con cortes “correctos”, el oído detecta discontinuidades en:timbre vocal (formantes), energía, y transiciones entre fonemas.

En resumen, el experimento muestra por qué la síntesis clásica por concatenación tiene dificultades para alcanzar naturalidad: puede construir la secuencia segmental, pero el reultado no se acerca a la claridad de los pangramas originales.


## Informe Integrante B
### 1. Diagnóstico del Problema y exploración inicial
El objetivo inicial fue sintetizar el pangrama objetivo en español aislando fonemas del pangrama origen ("El veloz murciélago...") mediante una lectura intencionalmente lenta y haciendo énfasis en las sílabas. El resultado no fue óptimo: solo se pudo extraer de manera entendible la palabra "jugoso"

**Posible explicación:** El habla no es una concatenación de unidades discretas. Al analizar las grabaciones en Praat, el espectrograma —que representa el sonido en tres dimensiones: Tiempo, Frecuencia y Energía— revela el problema de la **coarticulación**. Al hablar, la modulación es continua y leer lento o tratar de tener una entonación "neutra" no soluciona estos problemas.

### 2. Nuevo pangrama
Ante la incompatibilidad del pangrama original con la prosodia y las transiciones de mi dialecto, decidí diseñar una frase origen alternativa que contuviera los fonemas exactos que necesitaba, adaptados a mi fonética natural rioplatense:
> *"Julián, zurdo frívolo, mezcló jungla de dedos: uña, ñandú, pipa; KIWI vino. Extra quimera: silla, tótem; noche llena, vaca al alma, copa Holanda."* (es_secundaria_b.mp3)

### 3. Ejecución Acústica en Praat
Para el proceso de "corta-pega", las decisiones de corte en el espectrograma se basaron en la naturaleza acústica de cada fonema vista en clase:

* **Vocales:** Identificables por sus ondas estables en el espacio. Los cortes se realizaron buscando los cruces en la forma de onda para evitar saltos bruscos de fase, intentando empalmar vocales con formantes similares.
* **Consonantes Oclusivas (p, t, k, b, d, g):** Son los segmentos más críticos y dificultosos. Acústicamente se componen de un silencio (cierre del tracto) seguido de un *release* o explosión abrupta de energía. Para evitar que sonaran cortadas, fue vital incluir la fase de silencio previo al realizar el empalme, aunque a veces fue imposible por el lugar donde estaba la palabra.
* **Consonantes Fricativas (f, s):** Al estar formadas por ruido continuo, su representación en el espectrograma es ruido en altas frecuencias. Fueron relativamente más eficientes al corte, siempre que se respetara su duración relativa frente a las vocales adyacentes.

### 4. Conclusiones del Paradigma
A pesar de la optimización y la grabación de una nueva frase, el resultado mantiene un tono "robótico" y poco natural. Esto demuestra por qué la síntesis por concatenación clásica tiene un límite en la naturalidad: carece de un modelo prosódico global. Como vimos en clase, en la comunicación humana, el cerebro del receptor intenta predecir patrones continuos. Al ensamblar fragmentos, rompemos la curva de entonación natural. Esta limitación evidencia por qué los Sistemas TTS (text to speech) ha migrado hacia modelos basados en Redes Neuronales como vimos en clases, que no pegan audios, sino que aprenden a generar el espectrograma desde cero prediciendo el contexto completo.





# 2. Jueces y calibración

## Qué es un "LLM-as-judge"

Para medir si una respuesta es fiel, relevante o correcta, necesitamos a alguien que la
puntúe. Hacerlo a mano no escala. La idea de **LLM-as-judge** es usar un modelo (o un
método automático) que puntúe cada respuesta de 0 a 1 según una rúbrica.

EvalForge tiene dos jueces, y los dos usan la misma interfaz (`evalforge/judges/base.py`):

### Juez determinista (por defecto)

Está en `evalforge/judges/deterministic.py`. No usa ningún LLM: puntúa con métodos
léxicos. Es gratis, funciona sin internet y da siempre el mismo resultado.

* **faithfulness:** parte la respuesta en frases y, para cada una, mide la similitud
  (TF-IDF coseno) contra los contextos. Si la similitud supera un umbral, la frase está
  "respaldada". La puntuación es frases respaldadas / frases totales.
* **answer_relevance:** similitud entre la respuesta y la pregunta.
* **answer_correctness:** combina similitud y F1 de tokens entre la respuesta y la
  verdad de referencia.

> **Concepto — TF-IDF.** Convierte un texto en un vector donde pesan más las palabras
> poco comunes. La **similitud coseno** entre dos vectores mide qué tan parecidos son
> (1 = iguales, 0 = sin palabras en común). Es la base de la similitud léxica.

> **Concepto — F1 de tokens.** Mezcla precisión y recall a nivel de palabras. Es una
> forma simple y clásica (estilo SQuAD) de medir si dos textos dicen casi lo mismo.

### Juez LLM (opcional, agnóstico de proveedor)

Está en `evalforge/judges/llm.py`. No depende de un proveedor fijo: usa una **cascada**
de proveedores (`evalforge/llm/`) y cae al siguiente si uno falla. Soporta Anthropic,
OpenAI, Gemini, DeepSeek y modelos locales open-source vía Ollama. Le pide al modelo una
puntuación en JSON con una rúbrica por métrica. Es más fuerte en matices (entiende
paráfrasis), pero cuesta dinero y necesita clave de API (salvo el local).

El orden de la cascada se configura con `EF_LLM_PROVIDERS` o `--providers`, por ejemplo
`anthropic,openai,ollama`. Cada proveedor lee su propia clave del entorno. Esto es
**orquestación multi-LLM**: resiliencia (si un proveedor cae, sigues) y libertad de
preferir un modelo barato o local primero.

## Por qué el juez por defecto es determinista

Las pruebas y el CI necesitan algo gratis, offline y repetible. Si cada corrida diera un
número distinto, no podrías comparar. Por eso el default es el léxico, y el juez LLM
queda detrás de la misma interfaz para cuando haga falta. (Ver
[ADR-0003](../adr/0003-deterministic-judge-by-default.md).)

## Lo más importante: calibrar

**Un juez que nadie ha comparado contra personas no es confiable.** Si el juez no está
de acuerdo con la opinión humana, sus puntuaciones son ruido, y una compuerta de CI
basada en ellas es peor que nada.

`evalforge calibrate` toma puntuaciones humanas de los mismos casos y, por métrica,
reporta:

* **Correlación de Pearson:** mide relación lineal (¿cuando el humano sube, el juez
  sube?). De -1 a 1.
* **Correlación de Spearman:** mide si coinciden en el **orden** (ranking).
* **MAE (error absoluto medio):** qué tan lejos está el juez del humano en promedio.

En el ejemplo del proyecto, el juez léxico da:

```
faithfulness       pearson=0.997   <- casi perfecto: confiable para fidelidad
answer_correctness pearson=0.991
answer_relevance   pearson=0.732   <- más débil: aquí conviene el juez LLM
```

Esto enseña la lección clave: no basta con tener una métrica automática; hay que
**demostrar** que coincide con el juicio humano antes de confiar en ella. (Ver
[ADR-0004](../adr/0004-calibrate-the-judge.md).)

## Validación con datos reales: RAGTruth

La calibración de arriba usa las etiquetas del *suite* de ejemplo. La prueba de fuego es
dato **real y etiquetado por humanos**: **RAGTruth**, un benchmark de alucinaciones en
RAG (respuestas reales de LLM sobre pares (pregunta, contexto) reales, con anotaciones
humanas que marcan el contenido alucinado). Se vendorea un subconjunto balanceado de
1000 filas (500 fieles / 500 no fieles) en `datasets/ragtruth.jsonl`. La etiqueta binaria
se **deriva** de la anotación humana: no fiel (0) si hay cualquier marca de alucinación
(un span o un conteo `evident_conflict`/`baseless_info` > 0), fiel (1) solo si todo está
limpio.

Con `python -m evalforge.real_benchmark`:

```
ROC-AUC real        : 0.598   accuracy 0.589
control etiquetas barajadas : 0.518   (-> ~azar)
control contexto aleatorio  : 0.496   (-> ~azar)
```

### Por qué el número modesto es, aquí, la demostración

RAGTruth es el benchmark **más difícil**: sus alucinaciones son ediciones sutiles
*dentro* del contexto (un dato falso insertado), no respuestas obviamente desconectadas.
Un juez **léxico** (coseno TF-IDF entre respuesta y contexto) detecta eso por encima del
azar (AUC ≈ 0,60), gratis y sin red, pero no llega a 1,0 — y eso es correcto, no un
defecto.

Lo importante no es el número alto, es **medirlo**. Los dos controles lo validan:

- **Etiquetas barajadas → AUC 0,52.** Si el juez "acertara" con etiquetas aleatorias,
  estaría midiendo fuga, no señal. Colapsa al azar, así que la señal es real.
- **Contexto aleatorio → AUC 0,50.** Emparejar cada respuesta con un contexto ajeno
  destruye la discriminación: el juez sí usa el *grounding* real, no el estilo del texto.

Ese es justo el propósito de EvalForge: te dice **exactamente** dónde está tu juez gratis
frente a humanos reales, para que sepas cuándo basta el léxico y cuándo calibrar o pasar
al juez LLM (misma interfaz). **Conocer el número es la capacidad.** La métrica
determinista ROC-AUC se define `AUC = P(score(fiel) > score(no fiel))` sobre todos los
pares — por eso es independiente del umbral y la reportamos como titular.

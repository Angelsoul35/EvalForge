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

### Juez Anthropic (opcional)

Está en `evalforge/judges/anthropic.py`. Usa la API de Anthropic (por defecto
`claude-sonnet-5-5`, que es el modelo recomendado para un rol de evaluador secundario).
Le pide al modelo una puntuación en JSON con una rúbrica por métrica. Es más fuerte en
matices (entiende paráfrasis), pero cuesta dinero y necesita clave de API.

Se instala aparte: `pip install 'evalforge[anthropic]'`.

## Por qué el juez por defecto es determinista

Las pruebas y el CI necesitan algo gratis, offline y repetible. Si cada corrida diera un
número distinto, no podrías comparar. Por eso el default es el léxico, y el de Anthropic
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
answer_relevance   pearson=0.732   <- más débil: aquí conviene el juez Anthropic
```

Esto enseña la lección clave: no basta con tener una métrica automática; hay que
**demostrar** que coincide con el juicio humano antes de confiar en ella. (Ver
[ADR-0004](../adr/0004-calibrate-the-judge.md).)

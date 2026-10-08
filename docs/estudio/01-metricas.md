# 1. Métricas de evaluación

Un sistema RAG (Retrieval-Augmented Generation) hace dos cosas: **recupera** textos
relevantes y, con ellos, **genera** una respuesta. Por eso medimos las dos partes por
separado. Si solo tuviéramos un número, no sabríamos cuál de las dos falló.

## Métricas de recuperación (basadas en etiquetas)

En cada caso, cada contexto recuperado viene marcado como relevante o no (`relevant:
true/false`). Esa etiqueta es la **verdad de referencia**, no la pone ningún modelo. Con
ella calculamos:

| Métrica | Qué mide | Fórmula intuitiva |
|---|---|---|
| `hit_rate` | ¿Al menos un contexto recuperado era relevante? | 1 si hay alguno relevante, 0 si no |
| `context_precision` | Qué proporción de lo recuperado es relevante | relevantes / recuperados |
| `context_recall` | Qué proporción de lo relevante se recuperó | relevantes recuperados / total relevantes |
| `mrr` | Posición del primer contexto relevante | 1 / (posición del primero) |

> **Concepto — precisión vs. recall.** Precisión responde "de lo que traje, ¿cuánto
> sirve?". Recall responde "de lo que servía, ¿cuánto traje?". Un buscador puede tener
> precisión alta y recall bajo (trae poco, pero bueno) o lo contrario.

> **Concepto — MRR (Mean Reciprocal Rank).** Premia que el primer resultado relevante
> aparezca arriba. Si el relevante está en la posición 1, aporta 1.0; en la 2, 0.5; en
> la 3, 0.33. Importa porque el modelo lee primero los contextos de arriba.

El código está en `evalforge/metrics/retrieval.py`. Son funciones puras y deterministas.

## Métricas de generación (las puntúa el juez)

| Métrica | Qué mide | Por qué importa |
|---|---|---|
| `faithfulness` | ¿Las afirmaciones de la respuesta salen del contexto? | Detecta **alucinaciones** |
| `answer_relevance` | ¿La respuesta contesta la pregunta? | Una respuesta correcta pero fuera de tema no sirve |
| `answer_correctness` | ¿La respuesta coincide con la verdad de referencia? | Mide si es correcta de fondo |

> **Concepto — fidelidad (faithfulness).** Es la métrica anti-alucinación. Partimos la
> respuesta en afirmaciones (frases) y revisamos si cada una está respaldada por algún
> contexto. Si la respuesta inventa datos que no están en el contexto, la fidelidad
> baja. En el juez determinista esto se mide con similitud léxica (ver capítulo 2).

## Por qué separar las dos familias

Imagina una respuesta correcta con `hit_rate = 0`: el buscador no trajo nada útil, pero
el modelo ya se sabía la respuesta. Con un solo número no verías ese problema; con las
métricas separadas, sí. Esa es la decisión de diseño en
[ADR-0002](../adr/0002-separate-retrieval-and-generation.md).

## Agregación

El `EvalRunner` calcula estas métricas por caso y luego saca el **promedio** de cada una
sobre todos los casos. Ese promedio es lo que ves en el reporte y lo que usa la
compuerta de CI.

# 4. Preguntas de entrevista

## Nivel junior / media

**¿Qué hace EvalForge en una frase?**
Mide la calidad de un sistema RAG con métricas de recuperación y de generación, usa un
juez que se puede calibrar contra humanos, y puede bloquear un cambio en CI si la calidad
baja.

**¿Por qué separas métricas de recuperación y de generación?**
Porque cuando una respuesta es mala, la causa puede estar en el buscador (trajo mal
contexto) o en el modelo (escribió mal con buen contexto). Un solo número mezcla las dos
y esconde el problema. Separadas, sé cuál falló.

**¿Qué es `faithfulness` y por qué importa?**
Mide si las afirmaciones de la respuesta salen del contexto. Es la métrica
anti-alucinación: si el modelo inventa datos que no están en el contexto, baja.

**¿Qué significa un `hit_rate` de 0 con una respuesta correcta?**
Que el buscador no trajo ningún contexto relevante, pero el modelo acertó por su cuenta.
Es una señal de que la recuperación está fallando aunque la respuesta salga bien.

## Nivel media / senior

**¿Por qué el juez por defecto es determinista y no un LLM?**
Porque las pruebas y el CI necesitan algo gratis, offline y repetible. Un LLM cuesta,
necesita red y da resultados distintos cada vez. El juez léxico es determinista; el de
Anthropic queda detrás de la misma interfaz para cuando se necesite más matiz.

**¿Cuál es la debilidad del juez léxico?**
Que se puede engañar con paráfrasis o con una respuesta fluida pero incorrecta que reusa
las palabras del contexto. Por eso es importante calibrar y, en suites difíciles, usar el
juez de Anthropic.

**¿Cómo sabes si puedes confiar en el juez?**
Con `calibrate`: comparo las puntuaciones del juez contra puntuaciones humanas de los
mismos casos y miro correlación de Pearson/Spearman y el error medio. Si la correlación
es baja, no confío en esa métrica para esa compuerta.

**¿Qué diferencia hay entre Pearson y Spearman aquí?**
Pearson mide relación lineal entre las puntuaciones; Spearman mide si coinciden en el
orden. Un juez puede ordenar bien (Spearman alto) aunque sus valores estén corridos
(MAE alto). Mirar los dos da una imagen más completa.

## Nivel senior / diseño

**¿Cómo meterías esto en un pipeline de CI real?**
Con `evalforge gate suite.yaml --min-faithfulness 0.8 ...` como paso del workflow. El
comando devuelve código distinto de cero si una métrica cae del umbral, y eso tumba el
PR. Lo mismo que un umbral de cobertura, pero para la calidad del RAG.

**¿Cómo agregarías una métrica nueva?**
Si es de recuperación, agrego una función en `metrics/retrieval.py` y la registro en el
runner. Si es de generación, la añado a `GENERATION_METRICS` y la implemento en los dos
jueces (o en la interfaz). El diseño por capas hace el cambio local.

**¿Qué le falta para producción?**
Soporte para embeddings semánticos reales en el juez léxico (hoy es TF-IDF), más métricas
(por ejemplo, toxicidad o fuga de datos personales), muestreo de tráfico real para armar
suites, y un panel para seguir las métricas en el tiempo.

## Reto para ti

* Agrega una métrica `answer_completeness` (¿la respuesta cubre todos los puntos de la
  verdad de referencia?) y califícala en los dos jueces.
* Cambia la similitud léxica por embeddings locales y vuelve a calibrar: ¿sube la
  correlación en `answer_relevance`?
* Exporta los spans de OpenTelemetry a un backend real (por ejemplo, un colector local)
  y observa una corrida completa.

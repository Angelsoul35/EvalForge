# Guía de estudio — EvalForge

Esta guía está en español y explica el proyecto para que lo estudies a fondo y lo
puedas sustentar en una entrevista. El código y la documentación técnica están en
inglés; aquí va el *por qué* y el *cómo*.

## Orden de lectura

1. [Métricas de evaluación](01-metricas.md) — qué medimos en recuperación y en
   generación, y por qué se separan.
2. [Jueces y calibración](02-jueces-y-calibracion.md) — qué es un LLM-as-judge, por qué
   el juez por defecto es determinista, y por qué hay que calibrarlo.
3. [Arquitectura y diseño](03-arquitectura.md) — las capas, la interfaz del juez y la
   compuerta de CI.
4. [Preguntas de entrevista](04-preguntas-de-entrevista.md) — preguntas típicas con
   respuestas.

## Cómo correrlo mientras estudias

```bash
make install
make demo     # run + gate + calibrate sobre el suite de ejemplo
make test     # 31 pruebas
```

El concepto central: **medir la calidad de un sistema RAG con números que puedes seguir
en el tiempo y usar como compuerta en CI.**

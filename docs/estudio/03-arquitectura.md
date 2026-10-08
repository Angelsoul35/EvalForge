# 3. Arquitectura y diseño

## Las capas

```
evalforge/
  schemas.py     modelos pydantic: Suite, EvalCase, SuiteReport
  metrics/       métricas de recuperación (por etiquetas) y similitud léxica
  judges/        interfaz Judge, juez determinista, juez Anthropic, fábrica
  runner.py      corre un suite y devuelve un reporte
  gate.py        compuerta de umbrales para CI
  calibrate.py   correlación juez vs. humano
  report.py      salida en JSON, Markdown y consola
  tracing.py     spans opcionales de OpenTelemetry
  cli.py         run / gate / calibrate
```

Cada capa tiene una responsabilidad. La CLI es una cáscara delgada sobre el `EvalRunner`;
el runner no sabe de formatos de salida; el juez no sabe del runner. Esa separación hace
el sistema fácil de probar y de extender.

## La interfaz del juez (inversión de dependencias)

El `EvalRunner` no depende de un juez concreto; depende del `Protocol` `Judge`:

```python
class Judge(Protocol):
    name: str
    def score(self, case: EvalCase, metric: str) -> MetricScore: ...
```

> **Principio SOLID — Inversión de Dependencias.** El módulo de alto nivel (el runner)
> depende de una abstracción (la interfaz Judge), no de una implementación concreta
> (web3, la API de Anthropic, etc.). Así, cambiar de juez es cambiar una línea, y todas
> las pruebas corren con el juez determinista sin necesidad de clave ni de internet.

La fábrica `build_judge(name)` en `judges/__init__.py` arma el juez por nombre
(`deterministic` o `anthropic`).

## El runner

`EvalRunner.run_case` hace dos cosas por cada caso:

1. Si hay contextos, calcula las métricas de recuperación (de las etiquetas).
2. Para cada métrica de generación, llama al juez. `answer_correctness` solo se calcula
   si el caso tiene verdad de referencia.

`EvalRunner.run` recorre todos los casos y arma el `SuiteReport` con los resultados por
caso y los promedios.

## La compuerta de CI

`gate.py` recibe un reporte y un diccionario de mínimos. Pasa solo si **todas** las
métricas con umbral llegan a su mínimo. La CLI devuelve código de salida distinto de
cero si falla, así que una regresión tumba el pull request.

> **Concepto — quality gate.** Es un control automático que bloquea un cambio si la
> calidad baja de cierto punto. Es la misma idea que un umbral de cobertura de pruebas,
> pero aplicado a la calidad del RAG.

## Observabilidad

`tracing.py` envuelve cada corrida y cada caso en un *span* de OpenTelemetry, si está
activado. Así, una evaluación se puede observar en el mismo backend de trazas que el
sistema que se está evaluando. Por defecto está apagado y es un no-op.

## Configuración

Las variables usan el prefijo `EF_` (ver `.env.example`): el juez, el umbral de similitud
para fidelidad, y si se emiten spans. El juez Anthropic lee `ANTHROPIC_API_KEY` del
entorno.

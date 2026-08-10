# Semana 3 — Día 5 · Depurar el experimento de chunking + primer retrieval comparativo

**Fecha:** 10 de agosto
**Horas aprox:**
**Estado de ánimo:**

---

## Conceptos nuevos hoy

- **Baseline ciego a la estructura** — una rama de control que ignora
  deliberadamente las fronteras del documento, para poder medir cuánto vale
  respetarlas. Sin baseline honesto no hay comparación.
- **Hipótesis falsable** — el experimento se diseñó de forma que pudiera
  matar la explicación previa. Y la mató.
- **Inflación de similitud por longitud** — la similitud coseno favorece a los
  textos cortos. Un chunk de 10 tokens puntúa altísimo sin aportar contenido.
- **Pipeline en dos etapas** (chunk → fichero → embed) — chunkear es gratis,
  embeber es caro. Separarlos permite iterar sobre el troceo sin re-embeber.

---

## Momento "ajá" del día

<!-- ¿Cuál es la frase, el código, el hallazgo o el error que me cambió
     el entendimiento? El momento en el que algo hizo "clic".
     Si no hubo uno claro, escribe "ninguno" y ya está. No inventes. -->



---

## Lo que más me costó

<!-- ¿Dónde me quedé atascado? ¿Qué tuve que releer dos veces?
     ¿Qué concepto no me entraba?
     Ser honesto aquí es lo que hace el diario útil en el futuro. -->



---

## Decisiones técnicas tomadas

- **Reescribir la recursiva como baseline plano** en vez de bajar
  `CHUNK_SIZE_TOKENS`. Bajar el tamaño habría hecho divergir los números, pero
  las dos ramas seguirían respetando units: habría medido ruido de
  implementación en lugar del valor de la estructura. Y habría dejado en pie la
  mentira del comentario, disfrazada de experimento que funciona.
- **Agrupar por documento, no por corpus entero.** Ignorar la estructura interna
  es el baseline; ignorar la identidad del documento es un bug.
- **Unir con `"\n\n"`** (separador de máxima preferencia del splitter) y no con
  `" "`. Con espacio el baseline sería hostil a la estructura, no ciego a ella,
  y el resultado estaría sesgado a favor de la híbrida.
- **Descartar la metadata de unit** (`page`, `section_path`…) en el baseline.
  Un chunk que abarca las páginas 12-14 no puede declarar `page=12`. La pérdida
  de trazabilidad es el coste real de ignorar la estructura; maquillarla
  ocultaría el principal defecto del baseline.
- **Descartado:** heurísticas de tamaño de fuente para detectar encabezados en
  PDF. Es un proyecto de parseo de PDF, no de RAG.

---

## Conexiones con cosas que ya sabía

<!-- ¿Con qué experiencia previa se conecta lo de hoy?
     Automatización de procesos, Python previo, otros conceptos del sprint,
     algo que leí en un newsletter, una charla... -->

-

---

## Preguntas abiertas / para explorar

- ¿Cómo se arreglan los chunks que son solo un encabezado? ¿Descartar por
  debajo de un mínimo de tokens, o fusionar la sección padre con su primera
  hija? ¿Qué hace LangChain con esto?
- Si la similitud coseno premia lo corto, ¿cómo se compara la calidad entre
  colecciones con chunks de tamaños muy distintos? ¿Qué métrica no tiene ese
  sesgo?
- ¿Merece la pena un baseline plano en producción, o es solo un instrumento de
  medida?

---

## Para el próximo día

- Arreglar los chunks-encabezado (punto 1 del siguiente paso en `estado.md`).
- Día 4 del plan: re-ranking, filtros, evaluación.
- Volver sobre lo que se saltó al pedir el código escrito: `defaultdict(list)`,
  dict comprehension con filtro, y el criterio del separador de unión.

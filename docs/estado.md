# Estado del sprint

Actualizado: 10 agosto 2026.

Este fichero es la verdad operativa del repo: dónde estoy, qué bloquea, qué
sigue. Cambia semanalmente. Las convenciones y el contrato de aprendizaje —que
casi no cambian— viven en `CLAUDE.md`, no aquí.

## Dónde estoy

**Semana 3, día 3.** RAG. El retrieval funciona; el experimento de chunking está
bloqueado por un defecto de montaje, no de código.

Cronología: arranque el 6 de abril, parón del 28 de abril al 8 de agosto,
retomado desde entonces.

## El bloqueo, y su causa raíz

Las dos estrategias de chunking (recursiva e híbrida) producen chunks idénticos.

El motivo: **el único documento del corpus es un PDF**
(`data/pdf/anthropic_economic_index.pdf`), y la extracción de texto de PDF
aplana la estructura. Un encabezado en un PDF no está marcado como tal — es solo
texto más grande. Sin secciones detectables, el chunker híbrido degrada al
camino recursivo y ambas estrategias convergen.

**El código no está roto.** El montaje experimental sí: dos estrategias
comparadas sobre una entrada incapaz de distinguirlas.

Corolario: un solo documento no es un corpus. Por eso el defecto del
`chunk_index` (ver deuda) está latente y todavía no se manifiesta.

## Siguiente paso, en orden

1. Abrir `13_chunking_strategies.py` y averiguar **qué señal exacta busca** para
   considerar que existe una sección. Hay que saber qué espera antes de darle un
   documento que lo tenga.
2. Añadir un documento Markdown al corpus. El cargador de Markdown ya existe
   (commit `dee1bae`), así que no hay nada que construir. Candidatos
   recomendados: los propios ficheros de `docs/` de este repo — tienen
   estructura de encabezados y, sobre todo, su contenido se conoce de memoria,
   lo que permite juzgar la calidad del retrieval sin adivinar. Eso importa para
   el día 4 y para la Semana 8.
3. Re-chunkear, re-ingestar y comprobar que las colecciones ya **no** son
   idénticas. Con eso el día 4 (re-ranking, filtros, evaluación) tiene material
   real sobre el que trabajar.

**Descartado por ahora:** detectar encabezados en el PDF mediante heurísticas de
tamaño de fuente. Es trabajo legítimo y en producción acabaría haciéndose, pero
es un proyecto de parseo de PDF, no de RAG. Anotado como deuda.

## Deuda pendiente

Por prioridad:

- **Model IDs obsoletos.** Verificado el 10 de agosto: `03_benchmark.py` falla
  con `404 - model: claude-sonnet-4-20250514`. Los IDs vigentes son
  `claude-opus-5`, `claude-sonnet-5`, `claude-haiku-4-5`. La tabla de precios
  del `README.md` está desfasada por lo mismo. **Esto es el entregable de la
  Semana 1 y está roto en un repo que es portfolio.**
- **No hay `requirements.txt`.** Bloqueante para mover el proyecto a la VM.
- **No hay tests.** Se usó *golden master* (capturar la salida antes y después)
  como sustituto en el refactor. Suficiente para eso, insuficiente a medio plazo.
- `chunk_index` se recalcula con `enumerate`, ignorando `chunk_index_in_unit` y
  `unit_index`. Se manifestará al añadir el segundo documento.
- `section` vacío en 92/92 chunks: el ejemplo `--filter section=...` del CLI
  siempre devuelve cero resultados.
- Colección residual `week3_demo` (9 embeddings) en ChromaDB, del script 11.
- Rotación de claves en `.env` sin confirmar. Los límites de gasto sí están
  puestos en ambas consolas.
- `docs/development.md` redactado el 8 de agosto pero nunca colocado en el repo.

## Historial reciente

**8 agosto.** Diagnóstico del repo tras el parón: Semanas 1 y 2 cerradas,
Semana 3 por el día 3 — bastante más avanzado de lo que sugerían los ficheros
subidos en abril. Punto de abandono localizado: 28 de abril, 22:44, justo tras
ejecutar la ingesta. Trabajo huérfano rescatado y commiteado (`dd0c534`) tras
tres meses viviendo solo en disco. Entorno de permisos montado y verificado
funcionalmente. Refactor de librería completado (`f15c153`): `embeddings.py` y
`retrieval.py` extraídos como módulos importables; `15_retrieval.py` pasó de 246
a 110 líneas quedándose solo con el CLI, y el modelo de embeddings tiene ahora
una única definición —antes ingesta y recuperación podían divergir en silencio.

**10 agosto.** Escrito este fichero y el `CLAUDE.md`. Detectado que la
convención de ramas y PRs, adoptada el 8 de agosto, no se había aplicado:
`f15c153` fue directo a `main`. Se aplica desde la rama `chore/claude-md` en
adelante. Comandos del pipeline RAG verificados contra los scripts reales.

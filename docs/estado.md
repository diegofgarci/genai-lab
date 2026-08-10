# Estado del sprint

Actualizado: 10 agosto 2026 (noche).

Este fichero es la verdad operativa del repo: dónde estoy, qué bloquea, qué
sigue. Cambia semanalmente. Las convenciones y el contrato de aprendizaje —que
casi no cambian— viven en `CLAUDE.md`, no aquí.

## Dónde estoy

**Semana 3, día 3 cerrado.** RAG. El bloqueo del chunking está resuelto: era un
defecto de código, no de datos. Las dos estrategias divergen y el experimento de
retrieval ya produce un resultado interpretable.

Cronología: arranque el 6 de abril, parón del 28 de abril al 8 de agosto,
retomado desde entonces.

## El bloqueo, resuelto — y la hipótesis que resultó falsa

Las dos estrategias de chunking producían chunks byte a byte idénticos.

**Lo que se creía (y estaba escrito aquí hasta esta noche):** que el corpus tenía
un solo PDF, que la extracción de PDF aplana los encabezados, y que sin secciones
detectables la híbrida degradaba a la recursiva. Diagnóstico plausible y falso.

**Cómo se refutó:** se añadió un Markdown con 36 secciones al corpus. Si la
hipótesis fuese cierta, las estrategias tendrían que haber divergido. Salieron
otra vez 92 y 92, idénticas. Hipótesis muerta.

Nota adicional: el Markdown **ya estaba en el corpus desde abril**. Las rutas
están hardcodeadas en `12_document_loaders.py` (`MD_PATH = docs/rag_fundamentals.md`),
así que el "corpus de un solo PDF" nunca fue cierto. Copiar el fichero a
`data/markdown/` no tiene efecto alguno; esa carpeta no la lee nadie.

**La causa real:** `chunk_recursive` iteraba sobre las mismas units del loader que
la híbrida. Respetaba exactamente las mismas fronteras. Y como el 98% de las
units ya cabían en 512 tokens, ninguna de las dos partía nada: dos funciones
distintas sin efecto sobre su entrada devuelven lo mismo.

El comentario del propio fichero afirmaba lo contrario ("IGNORES the unit
boundaries") y se contradecía dentro de la misma frase ("within each unit").
Los números lo desmentían: 91 units → 92 chunks, mínimos de 9 tokens. Un
baseline realmente plano habría dado ~46 chunks pegados al máximo.

**Lección transferible:** cuando dos ramas de un experimento dan idéntico
resultado, sospecha primero que ninguna de las dos está haciendo nada, antes
que buscar la causa en los datos. Y no te fíes de los comentarios; verifica
contra la salida.

## Lo hecho (rama `fix/week3-recursive-baseline`, commit `7ffde40`)

`chunk_recursive` reescrita como baseline ciego a la estructura: agrupa units por
documento, concatena y parte uniformemente. Decisiones tomadas:

- **Por documento, no por corpus entero.** Concatenar PDF y Markdown juntos
  produciría chunks que cruzan documentos sin relación. Ignorar la estructura
  interna es el baseline; ignorar la identidad del documento es un bug.
- **Unión con `"\n\n"`**, el separador de máxima preferencia del splitter. Con
  `" "` el baseline sería *hostil* a la estructura en vez de ciego a ella, y
  sesgaría la comparación a favor de la híbrida.
- **Metadata de unit descartada** (`page`, `section_path`, `header_level`,
  `section_index`). Un chunk que abarca las páginas 12-14 no puede declarar
  `page=12`. La pérdida de trazabilidad **es** el coste de ignorar la estructura;
  maquillarla ocultaría el principal defecto del baseline.
- `chunk_index_in_unit` → `chunk_index_in_doc`. Verificado con grep que ningún
  consumidor leía las claves viejas.

Resultado: recursive 92 → **59 chunks**, media 255 → 401 tokens. Divergen.

## El experimento de retrieval, y lo que dice

Misma query (*"que es un embedding"*), misma k=3, dos colecciones:

| | Recursive | Hybrid |
|---|---|---|
| Mejor similitud | 0.3136 | 0.4978 |
| Media | 0.2868 | 0.4788 |
| ¿Contiene la definición? | cortada | completa (puesto #2) |

La recursiva devuelve bloques de ~494 tokens; el #1 arranca a mitad de un bloque
de código y recupera el ejemplo sin la frase que lo define. La híbrida devuelve
`### Definition` entera y autocontenida, 89 tokens.

**Salvedad importante:** los números no son directamente comparables entre
colecciones. La similitud coseno favorece a los textos cortos — un chunk de 10
tokens que coincide con las palabras de la query puntúa altísimo sin aportar
nada. Parte de la ventaja de la híbrida es inflación por longitud. La calidad
real se juzga leyendo el chunk, no la métrica.

## Siguiente paso, en orden

1. **Chunks que son solo un encabezado.** Los puestos #1 y #3 de la híbrida son
   `## 2. Embeddings — The Core Primitive` (10 tokens) y `## 4. Choosing an
   Embedding Model` (9 tokens). Cero contenido, y se llevan la similitud más
   alta. Dos tercios de la ventana de contexto desperdiciados. El loader emite
   una unit por encabezado incluso cuando la sección no tiene cuerpo propio
   porque solo contiene subsecciones. Decidir el arreglo: descartar units por
   debajo de un mínimo de tokens, o fusionar la sección padre con su primera
   hija.
2. **Día 4 propiamente dicho:** re-ranking, filtros, evaluación. Ya hay material
   real sobre el que trabajar.
3. Recuperar los conceptos que se saltaron al pedir el código escrito:
   `defaultdict(list)`, dict comprehension con filtro, y el criterio del
   separador de unión.

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
- **Chunks que son solo un encabezado** (9-10 tokens, sin cuerpo). Ver punto 1
  del siguiente paso. Afecta a la calidad del retrieval de la híbrida hoy.
- **Rutas de corpus hardcodeadas** en `12_document_loaders.py`: `PDF_PATH` y
  `MD_PATH` apuntan a ficheros concretos. `data/markdown/` existe pero no se lee.
  Añadir un documento al corpus exige editar código, y la estructura de
  directorios miente sobre de dónde sale el corpus. Debería ser un `glob` sobre
  `data/`.
- `chunk_index` se recalcula con `enumerate`, ignorando `chunk_index_in_unit` y
  `unit_index`. Se manifestará al añadir el segundo documento.
- **Nadie escribe la clave `section`.** El loader de Markdown escribe
  `section_path` (breadcrumb completa) y el de PDF escribe `page`. El ejemplo
  `--filter section=...` del CLI devuelve cero resultados siempre, y los
  diagnósticos imprimen `section=(none)` en el 100% de los chunks. Decidir si se
  filtra por `section_path` o si se deriva un `section` corto a partir de él.
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

**10 agosto (noche).** Bloqueo del chunking resuelto. La hipótesis del "corpus de
un solo PDF" refutada experimentalmente y sustituida por la causa real: la
estrategia recursiva no era un baseline plano, iteraba sobre units igual que la
híbrida. Reescrita en la rama `fix/week3-recursive-baseline` (`7ffde40`);
recursive pasa de 92 a 59 chunks. Primer experimento de retrieval comparativo
ejecutado sobre las dos colecciones: la híbrida devuelve la sección completa, la
recursiva la corta a mitad. Detectados dos defectos nuevos: chunks que son solo
un encabezado, y rutas de corpus hardcodeadas.

**10 agosto.** Escrito este fichero y el `CLAUDE.md`. Detectado que la
convención de ramas y PRs, adoptada el 8 de agosto, no se había aplicado:
`f15c153` fue directo a `main`. Se aplica desde la rama `chore/claude-md` en
adelante. Comandos del pipeline RAG verificados contra los scripts reales.

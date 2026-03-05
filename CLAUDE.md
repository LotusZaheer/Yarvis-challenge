# CLAUDE.md

## Pipeline
CSV único → carga y validación → limpieza y normalización → EDA → clustering de contactos conectados + análisis de sentimiento propio → evaluación de desempeño del agente de IA → generación de entregables (PDF de reporte, CSVs de clústeres, PDF de métodos técnicos, código).

## Reglas esenciales
- Cada vez que reciba el prompt "commit", revisa los Staged Changes, determina cómo los cambios afectan `PROJECT_REVIEW.md` y actualízalo solo si es necesario, siempre ignorando `sanity_checks/` y `logs/`, y sugiere un nombre de commit en inglés como texto plano.
- Cada etapa del pipeline recibe un `pl.DataFrame` y retorna un `pl.DataFrame` transformado.
- El CSV de entrada es la única fuente de datos; sin acceso a APIs ni bases de datos externas.
- Archivos de salida: `CSV` para clústeres, `PDF` para reportes, `.ipynb` o `.py` para código fuente.
- Nombres de archivos en `snake_case`; columnas normalizadas en `snake_case` durante limpieza.
- Modelos de clustering y sentimiento con `random_state` fijo para reproducibilidad.
- Separación estricta: `sanity_checks/` para validaciones y análisis exploratorio; `scripts/` exclusivamente para pipeline ejecutable y reproducible
- No modificar el CSV original en `data/raw/` es inmutable
- `sanity_checks/` escribe únicamente en `data/interim/`
- `scripts/` solo genera outputs finales en `data/processed/`


## Estructura
```
project/
├── data/
│   ├── raw/           # CSV original, inmutable
│   └── processed/     # CSVs limpios y clústeres de salida
├── sanity_checks/     # Exploración y análisis interactivo (scripts numerados)
├── scripts/           # Módulos del pipeline
├── reports/           # PDFs generados
├── models/            # Modelos serializados (si aplica)
└── .claude/
    └── skills/
```

## Estado del proyecto
**Fase:** Etapa 1 completada — Carga y validación inicial.

### Completado
- `scripts/load_data.py`: carga CSV con polars, renombra columnas a snake_case, valida columnas esperadas, crea columna derivada `call_completed`, exporta muestra de inconsistencias a `data/processed/`.
- `main.py`: punto de entrada del pipeline, muestra resumen visual con `rich`.
- `requeriments.txt`: dependencias versionadas (`>=`), sin pandas.

### Pendiente
- Limpieza y normalización avanzada
- EDA y visualizaciones
- Clustering de contactos conectados
- Análisis de sentimiento propio
- Evaluación de desempeño del agente de IA
- Generación de entregables (PDFs, CSVs finales)

## Skills de referencia
- `.claude/skills/architecture/pipeline.md`
- `.claude/skills/architecture/suggestions.md`
- `.claude/skills/methods/call_analysis.md`
- `.claude/skills/languages/python.md`
- `.claude/skills/frameworks/data_analysis.md`
- `.claude/skills/hardware/local_env.md`

## Consideraciones
- Always reply in spanish (code, functions, variables in english, only the comments in spanish) even if input is in english.
- Not use large (simplify) explanatory unless requested.
- Cada que se agregue una nueva libreria al env de python agregala a requirements.txt
- Mantener únicamente dependencias directas en `requirements.txt` con versión mínima `>=` (evitar `pip freeze` completo).

# CLAUDE.md

## Pipeline
CSV único → carga y validación → limpieza y normalización → EDA → clustering de contactos conectados + análisis de sentimiento propio → evaluación de desempeño del agente de IA → generación de entregables (PDF de reporte, CSVs de clústeres, PDF de métodos técnicos, código).

## Reglas esenciales
- Cada vez que reciba el prompt "commit": revisa los Staged Changes, determina cómo los cambios afectan `PROJECT_REVIEW.md` y actualízalo solo si es necesario (ignorando `sanity_checks/` y `logs/`), y **SUGIERE** un nombre de commit en inglés impreso en consola. No hagas el commit — el usuario lo hará manualmente.
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
├── scripts/           # Módulos del pipeline (load_data, clean_data, etc.)
├── utils/             # Código reutilizable entre main.py y main.ipynb (display, helpers, etc.)
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

## Filosofía de output del pipeline
- Los módulos de `scripts/` (load_data, clean_data, etc.) imprimen **únicamente lo mínimo indispensable**: conteo de registros cargados, advertencias de validación (`[WARN]`), y ruta de archivos exportados. **Nunca** imprimen tipos de dato, listas de columnas ni estadísticas detalladas — eso es responsabilidad de la capa de visualización.
- `main.py` recibe el parámetro `verbose=True` (por defecto) y es la capa de presentación: muestra tablas rich completas en consola. Importa funciones de visualización desde `utils/display.py`.
- `main.ipynb` llama `main(verbose=False)` para ejecutar el pipeline en silencio, y expone `df_clean` al final de la última celda para aprovechar el renderizado nativo del notebook. Importa funciones de visualización desde `utils/display.py`.
- `utils/display.py` contiene funciones de presentación compartidas (`mostrar_resumen`, `mostrar_muestra`, `mostrar_columnas_nuevas`) que usan `rich` para renderizar tablas y paneles. Estas funciones son utilizadas solo cuando `verbose=True`.

# PROJECT_REVIEW.md

## Infraestructura
- [x] Creación de un environment local segun el sistema operativo (`env/`)
- [x] Verificar encoding y delimitador del CSV fuente (73,487 registros, cargado correctamente)
- [x] Confirmar librerías disponibles (polars, scikit-learn, scipy, matplotlib, rich)
- [x] Definir entorno virtual y archivo de dependencias (`requeriments.txt` con versiones `>=`)
- [x] Crear estructura de directorios del proyecto
- [x] Refactorización: Separación de funciones de visualización en `utils/display.py` (reutilizables entre `main.py` y `main.ipynb`)

## Fases
- [x] **Fase 1 – Carga y validación:** `scripts/load_data.py` funcional
  - Columnas renombradas a snake_case (`campaign_id`, `target_id`, `call_url`, `connected`, `disconnected_reason`, `duration_ms`, `transcript`, `post_call_analysis`, `executed_at`)
- [x] **Fase 2 – Limpieza y normalización:** `scripts/clean_data.py` funcional
  - `executed_at` → Datetime + features temporales (`hour`, `day_of_week`, `date`)
  - `duration_ms` → Float64 + `duration_sec` + `duration_outlier` (flag p99)
  - `post_call_analysis` JSON desempaquetado en 6 columnas `pca_*` (`pca_sentimiento`, `pca_razon_churn`, etc.)
  - `transcript` JSON → texto plano `transcript_text` + `transcript_length`
  - `inconsistency_flag`: 16,700 registros con `connected != call_completed`
  - Salida: `data/processed/calls_clean.csv` (73,487 filas, 25 columnas)
- [ ] Fase 3 – Análisis exploratorio de datos (EDA) con visualizaciones
- [ ] Fase 4 – Análisis de patrones de contactabilidad
- [ ] Fase 5 – Clustering de contactos conectados
- [ ] Fase 6 – Análisis de sentimiento propio (independiente del post-llamada existente)
- [ ] Fase 7 – Evaluación de desempeño del agente de IA
- [ ] Fase 8 – Generación de reporte PDF con hallazgos y visualizaciones
- [ ] Fase 9 – Exportación de CSVs con clústeres resultantes
- [ ] Fase 10 – Generación de PDF de métodos técnicos empleados

## Integración
- [ ] Validar flujo completo de extremo a extremo con muestra del dataset
- [ ] Asegurar compatibilidad de formatos entre módulos (DataFrame → CSV → PDF)
- [ ] Verificar que outputs de clustering alimentan correctamente la exportación CSV

## Validación
- [ ] Verificar reproducibilidad de modelos (seeds fijos documentados)
- [ ] Validar cobertura de los ~73.000 registros sin pérdida de datos relevante
- [ ] Verificar que los entregables finales cumplen el formato esperado
- [ ] Confirmar que el análisis de sentimiento propio difiere del campo post-llamada existente

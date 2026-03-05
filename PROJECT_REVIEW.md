# PROJECT_REVIEW.md

## Infraestructura
- [x] Creación de un environment local segun el sistema operativo (`env/`)
- [x] Verificar encoding y delimitador del CSV fuente (73,487 registros, cargado correctamente)
- [x] Confirmar librerías disponibles (polars, scikit-learn, scipy, matplotlib, rich)
- [x] Definir entorno virtual y archivo de dependencias (`requeriments.txt` con versiones `>=`)
- [x] Crear estructura de directorios del proyecto
- [ ] Crear carpeta `notebooks/` (pendiente)

## Fases
- [x] **Fase 1 – Carga y validación:** `scripts/load_data.py` funcional
  - Columnas renombradas a snake_case (`campaign_id`, `target_id`, `call_url`, `connected`, `disconnected_reason`, `duration_ms`, `transcript`, `post_call_analysis`, `executed_at`)
  - Columna derivada `call_completed` (`call_url.is_not_null()`)
  - Muestra de inconsistencias exportada → `data/processed/muestra_10_diff_y_10_false.csv`
- [ ] Fase 2 – Limpieza y normalización de columnas
- [ ] Fase 3 – Análisis exploratorio de datos (EDA) con visualizaciones
- [ ] Fase 4 – Análisis de patrones de contactabilidad
- [ ] Fase 5 – Clustering de contactos conectados
- [ ] Fase 6 – Análisis de sentimiento propio (independiente del post-llamada existente)
- [ ] Fase 7 – Evaluación de desempeño del agente de IA
- [ ] Fase 8 – Generación de reporte PDF con hallazgos y visualizaciones
- [ ] Fase 9 – Exportación de CSVs con clústeres resultantes
- [ ] Fase 10 – Generación de PDF de métodos técnicos empleados

## Archivos generados
- `data/processed/muestra_10_diff_y_10_false.csv` — 20 registros: 10 con `call_completed ≠ connected`, 10 con `connected == False`

## Integración
- [ ] Validar flujo completo de extremo a extremo con muestra del dataset
- [ ] Asegurar compatibilidad de formatos entre módulos (DataFrame → CSV → PDF)
- [ ] Verificar que outputs de clustering alimentan correctamente la exportación CSV

## Validación
- [ ] Verificar reproducibilidad de modelos (seeds fijos documentados)
- [ ] Validar cobertura de los ~73.000 registros sin pérdida de datos relevante
- [ ] Verificar que los entregables finales cumplen el formato esperado
- [ ] Confirmar que el análisis de sentimiento propio difiere del campo post-llamada existente

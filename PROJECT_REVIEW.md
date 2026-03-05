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
- [ ] **Fase 3 – EDA + Tarea 0 — Patrones de Contactabilidad:** `scripts/contactability.py`
  - Tasa de conexión (`connected=True / total`) por `hour`, `day_of_week`, tipo de campaña (extraído de `name`) y cruce hora × día (heatmap)
  - Determinar ventana óptima: top N horas × días con mayor tasa de conexión
  - Exportar visualizaciones a `reports/figures/`
  - Retorna `pl.DataFrame` con métricas de contactabilidad
- [ ] **Fase 4 – Tarea 1 — Clustering de Contactos Conectados:** `scripts/cluster_contacts.py`
  - Filtrar `connected = True`; features: `duration_sec`, `transcript_length`, `hour`, `pca_sentimiento` (one-hot), `pca_posible_recuperacion` (one-hot), `disconnected_reason` (one-hot top categorías)
  - Normalizar con `StandardScaler`; KMeans `random_state=42`; k óptimo via elbow + silhouette (rango 2–8)
  - Perfil descriptivo por cluster; exportar `data/processed/clusters_contacts.csv` con columna `cluster_id`
- [ ] **Fase 5 – Tarea 2 — Análisis de Sentimiento Propio:** `scripts/sentiment_analysis.py`
  - **Nota:** `pca_sentimiento` (ya extraído del JSON post-llamada) es la clasificación del sistema Yarvis — **NO** es el sentimiento propio a derivar.
  - Input: `transcript_text`; modelo: `pysentimiento` (robertuito, preentrenado en español conversacional)
  - Clasificar cada transcripción → `sentiment_own` (negativo / neutral / positivo)
  - Comparar distribución `sentiment_own` vs. `pca_sentimiento`; analizar por campaña, `hour`, `duration_sec`
  - Exportar `sentiment_own` integrado (CSV separado o columna adicional)
- [ ] **Fase 6 – Tarea 3 — Desempeño del Agente Yarvis:** `scripts/agent_performance.py`
  - Input: `transcript_text`, `pca_resumen`, `pca_razon_churn`, `duration_sec`, `sentiment_own`, `disconnected_reason`
  - Detectar patrones de falla: respuestas repetitivas (regex), silencios/inactividad, no manejo de objeciones, malentendidos (llamadas cortas + user_hangup + sentimiento negativo)
  - Cuantificar frecuencia por tipo de falla; cruzar con campaña y hora; generar recomendaciones
  - Output: métricas + ejemplos (va al reporte, sin CSV separado)
- [ ] **Fase 7 – Generación de Entregables**
  - `reports/report.pdf`: hallazgos tareas 0–3, visualizaciones, razonamiento (weasyprint o reportlab)
  - `reports/methods.pdf`: métodos técnicos empleados
  - `data/processed/clusters_contacts.csv`: generado en Fase 4

## Integración
- [ ] Validar flujo completo de extremo a extremo con muestra del dataset
- [ ] Asegurar compatibilidad de formatos entre módulos (DataFrame → CSV → PDF)
- [ ] Verificar que outputs de clustering alimentan correctamente la exportación CSV

## Validación
- [ ] Verificar reproducibilidad de modelos (seeds fijos documentados)
- [ ] Validar cobertura de los ~73.000 registros sin pérdida de datos relevante
- [ ] Verificar que los entregables finales cumplen el formato esperado
- [ ] Confirmar que el análisis de sentimiento propio difiere del campo post-llamada existente

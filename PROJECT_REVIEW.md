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
  - Extraer `campaign_type` del campo `name` (churn, upsell, retención, etc.) como feature reutilizable
  - Salida: `data/processed/calls_clean.csv` (73,487 filas, ~26 columnas)
- [ ] **Fase 3 – EDA + Tarea 0 — Patrones de Contactabilidad:** `scripts/contactability.py`
  - Tasa de conexión (`connected=True / total`) por `hour`, `day_of_week`, `campaign_type` y cruce hora × día (heatmap)
  - Determinar ventana óptima: top N horas × días con mayor tasa de conexión
  - **Visualizaciones:** heatmap hora × día, barras por `campaign_type`, línea de tasa por hora
  - Exportar visualizaciones a `reports/figures/contactability_*.png`
  - Retorna `pl.DataFrame` con métricas de contactabilidad
- [ ] **Fase 4 – Tarea 1 — Clustering de Contactos Conectados:** `scripts/cluster_contacts.py`
  - Filtrar `connected = True`; features: `duration_sec`, `transcript_length`, `hour`, `pca_sentimiento` (one-hot), `pca_posible_recuperacion` (one-hot), `disconnected_reason` (one-hot top categorías), `campaign_type` (one-hot)
  - Normalizar con `StandardScaler`; KMeans `random_state=42`; k óptimo via elbow + silhouette (rango 2–8)
  - Perfil descriptivo por cluster
  - **Visualizaciones:** elbow plot, silhouette plot, scatter PCA 2D coloreado por cluster, barras de distribución de features por cluster
  - Exportar visualizaciones a `reports/figures/clusters_*.png`
  - Exportar `data/processed/clusters_contacts.csv` con columna `cluster_id`
- [ ] **Fase 5 – Tarea 2 — Análisis de Sentimiento Propio:** `scripts/sentiment_analysis.py`
  - **Nota:** `pca_sentimiento` (ya extraído del JSON post-llamada) es la clasificación del sistema Yarvis — **NO** es el sentimiento propio a derivar.
  - Input: `transcript_text`
  - **Estrategia de clasificación (con fallback):**
    1. Opción A: `pysentimiento` (robertuito, preentrenado en español conversacional) — requiere internet
    2. Opción B (fallback): Clasificación rule-based con lexicón de keywords positivos/negativos en español sobre el transcript
  - Clasificar cada transcripción → `sentiment_own` (negativo / neutral / positivo)
  - Comparar distribución `sentiment_own` vs. `pca_sentimiento`
  - Analizar por `campaign_type`, `hour`, `duration_sec`
  - **Validación cruzada:** correlacionar `sentiment_own` con `disconnected_reason` (¿negativo → `user_hangup`?) y `duration_sec` (¿negativo → llamadas más cortas?)
  - **Ejemplos concretos:** Extraer 2–3 fragmentos representativos de transcript por categoría de sentimiento para incluir en el reporte
  - **Visualizaciones:** barras de distribución sentimiento propio vs. Yarvis, heatmap sentimiento × campaign_type, boxplot duración por sentimiento
  - Exportar visualizaciones a `reports/figures/sentiment_*.png`
  - Exportar `sentiment_own` como columna adicional en CSV o archivo separado
- [ ] **Fase 6 – Tarea 3 — Desempeño del Agente Yarvis:** `scripts/agent_performance.py`
  - Input: `transcript_text`, `pca_resumen`, `pca_razon_churn`, `duration_sec`, `sentiment_own`, `disconnected_reason`
  - Detectar patrones de falla:
    - Respuestas repetitivas (regex / n-gram repetition)
    - Silencios / inactividad (`disconnected_reason = inactivity`)
    - No manejo de objeciones (keywords de objeción sin respuesta adaptada)
    - Malentendidos (llamadas cortas + `user_hangup` + sentimiento negativo)
  - Cuantificar frecuencia por tipo de falla; cruzar con `campaign_type` y `hour`
  - **Ejemplos concretos:** Extraer 2–3 fragmentos de transcript por tipo de fallo para incluir en el reporte
  - **Visualizaciones:** barras de frecuencia por tipo de fallo, heatmap fallos × campaign_type, tabla resumen con ejemplos
  - Exportar visualizaciones a `reports/figures/agent_*.png`
  - Generar recomendaciones concretas de mejora
  - Output: métricas + ejemplos + recomendaciones (va al reporte, sin CSV separado)
- [ ] **Fase 7 – Generación de Entregables**
  - `reports/report.pdf`: Reporte principal para stakeholders no técnicos
    - **Página 1: Resumen ejecutivo** con los 3–5 hallazgos clave y recomendaciones accionables
    - Secciones por tarea (0–3) con visualizaciones, razonamiento y ejemplos de transcripts
    - Tono ejecutivo, orientado a decisiones
    - Generado con weasyprint o reportlab
  - `reports/methods.pdf`: Métodos técnicos empleados (corto y concreto)
  - `data/processed/clusters_contacts.csv`: Generado en Fase 4
  - `main.ipynb`: Notebook con el flujo completo de análisis (entregable explícito junto a scripts)

## Integración
- [ ] Validar flujo completo de extremo a extremo con muestra del dataset
- [ ] Asegurar compatibilidad de formatos entre módulos (DataFrame → CSV → PDF)
- [ ] Verificar que outputs de clustering alimentan correctamente la exportación CSV
- [ ] Verificar que `sentiment_own` de Fase 5 alimenta correctamente Fase 6 (agent performance)

## Validación
- [ ] Verificar reproducibilidad de modelos (seeds fijos documentados)
- [ ] Validar cobertura de los ~73.000 registros sin pérdida de datos relevante
- [ ] Verificar que los entregables finales cumplen el formato esperado
- [ ] Confirmar que el análisis de sentimiento propio difiere del campo post-llamada existente
- [ ] Validar que cada sección del reporte incluye visualizaciones y ejemplos de transcript

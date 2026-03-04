# PROJECT_REVIEW.md

## Infraestructura
- [ ] Creación de un environment local segun el sistema operativo
- [ ] Verificar encoding y delimitador del CSV fuente
- [ ] Confirmar librerías disponibles (pandas, polars, scikit-learn, scipy, matplotlib, generador PDF)
- [ ] Definir entorno virtual y archivo de dependencias (`requirements.txt`)
- [ ] Crear estructura de directorios del proyecto

## Fases
- [ ] Implementar módulo de carga y validación del CSV (~73k registros)
- [ ] Implementar módulo de limpieza y normalización de columnas
- [ ] Implementar análisis exploratorio de datos (EDA) con visualizaciones
- [ ] Implementar análisis de patrones de contactabilidad
- [ ] Implementar clustering de contactos conectados
- [ ] Implementar análisis de sentimiento propio (independiente del post-llamada existente)
- [ ] Implementar evaluación de desempeño del agente de IA
- [ ] Implementar generación de reporte PDF con hallazgos y visualizaciones
- [ ] Implementar exportación de CSVs con clústeres resultantes
- [ ] Implementar generación de PDF de métodos técnicos empleados

## Integración
- [ ] Validar flujo completo de extremo a extremo con muestra del dataset
- [ ] Asegurar compatibilidad de formatos entre módulos (DataFrame → CSV → PDF)
- [ ] Verificar que outputs de clustering alimentan correctamente la exportación CSV

## Validación
- [ ] Verificar reproducibilidad de modelos // (seeds fijos documentados)
- [ ] Validar cobertura de los ~73.000 registros sin pérdida de datos relevante
- [ ] Verificar que los entregables finales cumplen el formato esperado
- [ ] Confirmar que el análisis de sentimiento propio difiere del campo post-llamada existente

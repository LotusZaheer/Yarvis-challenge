# Markdown a PDF

# Desafío de Análisis de Datos de Llamadas

## Contexto

Estás trabajando en Yarvis, una empresa que construye agentes de voz de IA para campañas de llamadas salientes. Las organizaciones usan Yarvis para automatizar llamadas para prevención de churn, upselling, retención, seguimientos de calidad de servicio, y más.

Se te da un dataset real de ~73,000 registros de llamadas que abarca múltiples campañas. Cada registro incluye estado de conexión, razón de desconexión, duración, una transcripción completa (cuando está conectada), y un análisis post-llamada generado por la IA después de cada llamada.

Tu tarea es analizar estos datos y entregar insights accionables.

---

# Datos

Se proporciona un solo archivo CSV: **data_calls.csv**

## Descripción de Columnas

| Columna             | Descripción                                                                                                        |
| ------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Campaign Id         | Identificador numérico de campaña                                                                                  |
| Name                | Nombre de la campaña (incluye tipo de campaña y fecha)                                                             |
| Target Id           | Identificador único para el contacto/target                                                                        |
| Call URL            | URL a la grabación de la llamada (NULL si no está conectada)                                                       |
| Connected           | Si la llamada fue respondida (True / False)                                                                        |
| Disconnected Reason | Por qué la llamada terminó (ej. dial_no_answer, voicemail_reached, user_hangup, inactivity, agent_hangup, etc.)    |
| Duration (ms)       | Duración de la llamada en milisegundos (NULL si no está conectada)                                                 |
| Transcript          | Array de diálogo agente/usuario (NULL si no hay conexión)                                                          |
| Post Call Analysis  | Objeto JSON con campos de análisis generados por IA (ej. resumen, resultado, interés_mostrado, objeción_principal) |

### Notas

Los datos están en español.

Los nombres de campañas codifican el tipo de campaña y el propósito de contactar a los targets (ej. "Posibles churn 60d Febrero", "Upsell TV Digital - 20/02").

---

# La Tarea

Analiza el dataset y produce un reporte respondiendo las siguientes preguntas, en orden de prioridad:

---

# 0. Patrones de Contactabilidad

Identifica cuándo los contactos son más alcanzables.

* ¿Qué horas del día producen las tasas de conexión más altas?
* ¿Hay diferencias por día de la semana?
* ¿Hay diferencias por tipo de campaña?
* ¿Cuál es la ventana óptima de llamadas para maximizar conexiones exitosas?

---

# 1. Clusters de Contactos

De las llamadas ejecutadas (conectadas), identifica clusters distintos de contactos.

* ¿Qué clusters emergen de los datos?
* ¿Cuáles son las características y tendencias comunes dentro de cada cluster?
* ¿Por qué están agrupados juntos? ¿Qué diferencia a cada cluster?
* Proporciona un razonamiento claro para tu enfoque de clustering y el número de clusters elegido.

---

# 2. Análisis de Sentimiento

Clasifica los contactos en sentimiento Negativo, Neutral, y Positivo.

* ¿Cuál es la distribución de sentimiento a través de las llamadas?

* ¿Qué impulsa cada categoría de sentimiento? Proporciona ejemplos concretos de transcripciones.

* ¿Hay patrones por tipo de campaña, hora del día, o duración de la llamada?

Ve más allá de los campos existentes de Post Call Analysis — deriva tu propia clasificación de sentimiento y justifícala.

---

# 3. Desempeño del Agente Yarvis

Identifica casos donde el agente de IA de Yarvis se desempeñó pobremente durante llamadas.

* ¿Qué tipos de fallas o problemas ocurren? (ej. respuestas repetitivas, fallar al manejar objeciones, silencios incómodos, malentender al cliente)
* ¿Qué tan frecuentes son estos problemas?
* ¿Qué patrones llevan a un mal desempeño?
* ¿Qué mejoras concretas recomendarías para arreglarlos?

---

# Entregables

* Un reporte (PDF, o slides) con tus hallazgos, visualizaciones, y razonamiento.
  (El reporte debería ser lo suficientemente claro para que un stakeholder no técnico entienda los principales puntos clave.)

* Los archivos csv de clusters de targets

* Cualquier código usado para el análisis (scripts, notebooks).

* Un PDF con los métodos técnicos usados.
  (Sé corto y concreto, esto no es un paper)

Esperamos que todo el challenge sea hecho usando herramientas de codificación con IA, muestra cómo lo hiciste.

---

# Formato

**1 hora de trabajo sincrónico** con el CTO — compartiendo pantalla, recorrido en vivo de tu enfoque y hallazgos iniciales.

**1 día de trabajo asincrónico** — finalizar tu análisis, pulir el reporte, y entregar.

---

# Criterios de Evaluación

**Profundidad analítica** — ¿Vas más allá de observaciones de superficie? ¿Encuentras insights no obvios?

**Razonamiento** — ¿Puedes explicar por qué las cosas son como son, no solo qué muestran los datos?

**Priorización** — Dado el tiempo limitado, ¿te enfocaste en lo que importa más?

**Habilidad técnica** — ¿Tu código es limpio, tu metodología es sólida?

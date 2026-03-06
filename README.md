# Yarvis-challenge

Análisis de ~73,000 registros de llamadas salientes de un agente de voz IA. Incluye limpieza, clustering, análisis de sentimiento, contactabilidad y evaluación de desempeño.

## Datos

El archivo de llamadas **no está incluido en el repositorio**. Antes de ejecutar el pipeline, coloca el CSV en:

```
data/raw/data_calls.csv
```

## Requisitos

- Python 3.12+
- Dependencias en `requeriments.txt`

## Instalación local

```bash
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate
pip install -r requeriments.txt
```

## Uso

### Pipeline en consola

```bash
python main.py
```

### Notebook interactivo

```bash
jupyter notebook main.ipynb
```

### Docker

```bash
# Build
docker build -t yarvis .

# Ejecutar pipeline
docker run yarvis

# Jupyter notebook
docker run -p 8888:8888 yarvis jupyter notebook --ip=0.0.0.0 --allow-root --no-browser
```

## Estructura

```
data/raw/          → CSV original (inmutable)
data/processed/    → CSVs de salida y clusters
scripts/           → Módulos del pipeline
utils/             → Código reutilizable
reports/figures/   → Figuras generadas (PNG)
reports/           → Reporte de hallazgos (LaTeX)
```

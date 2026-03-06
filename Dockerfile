# --- Stage 1: builder ---
FROM python:3.12-slim AS builder

WORKDIR /build

COPY requeriments.txt .

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r requeriments.txt

# --- Stage 2: runtime ---
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY . .

EXPOSE 8888

CMD ["python", "main.py"]

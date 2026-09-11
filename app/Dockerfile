# Use Python 3.11 slim image (3.9 reaches end of life in 2025)
FROM python:3.11-slim

WORKDIR /app

# Build context is this directory (Disaster_Relief_App/app). The previous
# Dockerfile mixed two contexts -- `COPY requirements.txt .` only resolves from
# here, while `COPY ./app /app/app/` only resolves from the repo root -- so one
# COPY always failed and the image could not be built from either location.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/app/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Drop root
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s \
    CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/_stcore/health')" || exit 1

CMD streamlit run app/app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false

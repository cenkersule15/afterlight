FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 AFTERLIGHT_ROOT=/workspace PYTHONPATH=/app
WORKDIR /app
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu && pip install -r /tmp/requirements.txt
COPY backend/afterlight /app/afterlight
RUN useradd --uid 10001 --create-home researcher
USER researcher
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=3)"
CMD ["uvicorn","afterlight.api:app","--host","0.0.0.0","--port","8000"]

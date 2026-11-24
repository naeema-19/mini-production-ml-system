# Optional container for the online scoring service (artifacts must exist: run `make data train` first)
FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY serving/ serving/
COPY monitoring/ monitoring/
COPY artifacts/ artifacts/
ENV PYTHONUNBUFFERED=1 USE_FAST_PATH=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"
USER nobody
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

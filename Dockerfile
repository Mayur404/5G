FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app \
    PRORAN_HOST=0.0.0.0 \
    PRORAN_PORT=8080

WORKDIR /app
COPY requirements-web.txt ./
COPY src ./src
COPY web ./web
COPY results/results.json results/ppo_model.npz results/training_history.csv results/imitation_history.csv results/paired_comparisons.csv ./results/
RUN pip install --no-cache-dir -r requirements-web.txt && \
    useradd --create-home --uid 10001 proran && chown -R proran:proran /app

USER proran
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2)"

CMD ["python", "-m", "web.server"]

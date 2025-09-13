# mlflow.Dockerfile
FROM python:3.10-slim-bookworm

WORKDIR /mlflow

# Install MLflow only
RUN pip install --no-cache-dir mlflow

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--gunicorn-opts", "--workers 2 --timeout 300"]
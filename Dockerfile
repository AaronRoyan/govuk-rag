FROM python:3.12-slim

WORKDIR /app
ENV FASTEMBED_CACHE_PATH=/app/.models PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY src src
RUN pip install --no-cache-dir .

COPY data data
# Download models and embed the corpus at build time so the container starts fast.
RUN python -c "from govrag.chunk import load_chunks; from govrag.retrieve import Retriever; Retriever.build(load_chunks()).search('warm up')"

EXPOSE 8000
CMD ["uvicorn", "govrag.api:app", "--host", "0.0.0.0", "--port", "8000"]

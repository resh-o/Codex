# Backend image for Codex Stage 1 (ingestion + AST chunking).
FROM python:3.13-slim

# git is required at runtime: GitPython shells out to it to clone repos.
# tree-sitter-language-pack ships prebuilt manylinux wheels, so no compiler is
# needed for the grammars themselves.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CODEX_WORKDIR=/data/repos

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App source.
COPY app ./app

# Where cloned repos land (also declared via CODEX_WORKDIR above).
RUN mkdir -p /data/repos

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

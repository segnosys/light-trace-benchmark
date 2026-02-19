FROM python:3.12-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl procps bash \
    && rm -rf /var/lib/apt/lists/*

# Copy the files needed for installation
COPY setup.py .
COPY pyproject.toml .
COPY README.md .
COPY lightrace/ ./lightrace/

# Install the package with development dependencies
RUN pip install --no-cache-dir -e "."

ENTRYPOINT ["/bin/sh", "-c"]

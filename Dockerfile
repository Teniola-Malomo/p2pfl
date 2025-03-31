FROM python:3.10-slim

WORKDIR /app

COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.8.3 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    HOME=/app

# Update and install build dependencies required for some packages
RUN apt-get update && \
    apt-get install -y gcc python3-dev build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install "poetry==$POETRY_VERSION"

# Install dependencies
RUN poetry install --without dev,docs

# Install torch (cpu) - poetry present problem with torch-cpu installation
RUN poetry run pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu

# Install torchmetrics and pytorch-lightning
RUN poetry run pip install torchmetrics==1.4.0.post0 pytorch-lightning==1.9.5

RUN pip install "p2pfl[torch]"

RUN pip install uvicorn fastapi

# Expose the default port
EXPOSE 5001

# Run with a default port (or override with Docker command args)
CMD ["python", "p2pfl/examples/node1.py", "--port", "5001"]
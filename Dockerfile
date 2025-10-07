FROM ubuntu:latest

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    libpq-dev \
    postgresql-client \
    fish \
    vim \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Install Python 3.14 via UV
RUN uv python install 3.14

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN uv sync

# Set up environment
ENV DJANGO_SETTINGS_MODULE=fuz.settings
ENV DB_HOST=postgres
ENV DB_NAME=fuz_db
ENV DB_USER=postgres
ENV DB_PASSWORD=postgres
ENV DB_PORT=5432

# Build the binary on container start (optional - can be done manually)
RUN echo '#!/bin/bash\n\
echo "=== Fuz Development Environment ==="\n\
echo ""\n\
echo "Available commands:"\n\
echo "  uv run fuz --help          - Run fuz CLI"\n\
echo "  uv run fuz migrate         - Run database migrations"\n\
echo "  uv run pyinstaller fuz.spec - Build binary"\n\
echo "  ./dist/fuz --help          - Run built binary (after building)"\n\
echo ""\n\
echo "Database: postgres://postgres:postgres@postgres:5432/fuz_db"\n\
echo ""\n\
exec fish' > /entrypoint.sh && chmod +x /entrypoint.sh

# Default command
CMD ["/entrypoint.sh"]

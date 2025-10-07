#!/bin/bash
set -e

# This script runs during PostgreSQL initialization
# It enables pgvector and pg_trgm extensions

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Enable pgvector extension for vector similarity search
    CREATE EXTENSION IF NOT EXISTS vector;

    -- Enable pg_trgm extension for trigram-based text search
    CREATE EXTENSION IF NOT EXISTS pg_trgm;

    -- Verify extensions are installed
    SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'pg_trgm');
EOSQL

echo "PostgreSQL extensions initialized: pgvector, pg_trgm"

# Database Migration Guide

## How to Generate and Apply Migrations

### Step 1: Make Model Changes
Edit `src/fuz/models.py` to add/modify your Django models.

### Step 2: Generate Migration
```bash
# Using uv (recommended)
uv run fuz migrate

# Or manually with Django management command
uv run python -m django makemigrations fuz
```

This will:
- Detect changes in your models
- Create a new migration file in `src/fuz/migrations/`
- The file will be named like `0001_initial.py`, `0002_add_field.py`, etc.

### Step 3: Apply Migration to Database
The `fuz migrate` command does both steps automatically:
```bash
uv run fuz migrate
```

Or apply separately:
```bash
# Only generate migration
uv run python -m django makemigrations fuz

# Only apply migrations
uv run python -m django migrate
```

## Current Setup: Pattern Model

The `Pattern` model includes:
- **version** (CharField, max 32 chars, indexed)
- **text** (TextField for log content)
- **GIN index with pg_trgm** for fast text similarity queries

### Initial Migration

```bash
# Generate and apply the initial migration for Pattern model
uv run fuz migrate
```

This creates:
1. `patterns` table
2. Standard B-tree index on `version` field
3. GIN index on `text` field with `gin_trgm_ops` for trigram similarity

## Using Docker

```bash
# Start containers
docker-compose up -d

# Enter dev container
docker-compose exec dev fish

# Inside container, run migration
uv run fuz migrate
```

## Checking Migration Status

```bash
# List all migrations and their status
uv run python -m django showmigrations

# Show SQL that will be executed
uv run python -m django sqlmigrate fuz 0001
```

## Rolling Back Migrations

```bash
# Roll back to previous migration
uv run python -m django migrate fuz 0001

# Roll back all migrations for fuz app
uv run python -m django migrate fuz zero
```

## Common Issues

### pg_trgm Extension Not Found
If you get an error about `pg_trgm`, ensure the extension is enabled:
```sql
-- Connect to database
psql -h localhost -U postgres -d fuz_db

-- Enable extension
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

With Docker, this is automatically handled by `init-db.sh`.

### Migration Conflicts
If you have migration conflicts:
```bash
# Remove all migrations
rm src/fuz/migrations/0*.py

# Recreate from scratch
uv run python -m django makemigrations fuz
uv run python -m django migrate
```

## Manual Migration Creation

For complex migrations (like data migrations):
```bash
# Create empty migration
uv run python -m django makemigrations fuz --empty --name custom_migration

# Edit the generated file to add custom operations
```

## Verifying the Index

After migration, verify the pg_trgm index exists:
```sql
-- Connect to database
psql -h localhost -U postgres -d fuz_db

-- List indexes on patterns table
\d patterns

-- Or query directly
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'patterns';
```

You should see `patterns_text_trgm_idx` with `gin_trgm_ops`.

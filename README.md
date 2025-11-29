# Fuz

A pattern analysis and similarity search platform using Django, PostgreSQL, and vector embeddings for semantic text analysis.
100% vibecoded before the internal hackathon days to prove some concepts.

⚠️ Vibecode alert — This tool was implemented using LLMs.

## Features

- ✅ **REST API** - Full CRUD operations for text patterns
- ✅ **Vector Embeddings** - Three embedding models (MiniLM, MPNet, Jina) for semantic search
- ✅ **Similarity Search** - Find nearest neighbors using cosine distance
- ✅ **3D Visualization** - Interactive cluster visualization with PCA/t-SNE
- ✅ **Clustering** - HDBSCAN, DBSCAN, K-Means, and user-defined labels
- ✅ **Background Processing** - Lightweight task queue for async embedding generation
- ✅ **Bulk Upload** - Upload thousands of patterns efficiently
- ✅ **PostgreSQL Extensions** - pgvector (HNSW index) and pg_trgm for fuzzy search
- ✅ **Python 3.13** - Modern Python with UV package manager
- ✅ **Docker Ready** - PostgreSQL 18 with pre-configured extensions

## Prerequisites

### Option 1: Docker (Recommended)
- Docker
- Docker Compose

### Option 2: Local Installation
- Python 3.13
- PostgreSQL 14+ with pgvector and pg_trgm extensions
- UV package manager

## Quick Start with Docker

```bash
# Start the environment (PostgreSQL 18)
docker-compose up -d

# Wait for PostgreSQL to be ready
docker-compose logs -f postgres

# Run migrations
python manage.py migrate

# Create a superuser (for Django admin)
python manage.py createsuperuser

# Start the development server
python manage.py runserver

# Access the application
# API: http://localhost:8000/api/
# Admin: http://localhost:8000/admin/
# Visualization: http://localhost:8000/visualize/
# API Docs: http://localhost:8000/api/docs/
```

The Docker setup includes:
- **PostgreSQL 18** with `pgvector` and `pg_trgm` extensions pre-installed
- Automatic database initialization
- Persistent data storage
- Network connectivity for local development

### Docker Commands

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f postgres

# Stop services
docker-compose down

# Remove volumes (clears database data)
docker-compose down -v

# Rebuild containers
docker-compose up -d --build
```

## Local Installation

```bash
# Install dependencies
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env with your database credentials
```

## Database Setup (Local)

```bash
# Create database
createdb fuz_db

# Enable extensions
psql fuz_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql fuz_db -c "CREATE EXTENSION IF NOT EXISTS pg_trgm;"

# Run migrations
uv run fuz migrate
```

## Usage

### REST API

The application provides a comprehensive REST API for pattern management:

#### Pattern CRUD Operations

```bash
# Create a pattern
curl -X POST http://localhost:8000/api/patterns/ \
  -H "Content-Type: application/json" \
  -d '{
    "version": "1.0.0",
    "entity_id": 123,
    "text": "Your log text here...",
    "label": "error"
  }'

# List all patterns
curl http://localhost:8000/api/patterns/

# Get a specific pattern
curl http://localhost:8000/api/patterns/1/

# Update a pattern
curl -X PUT http://localhost:8000/api/patterns/1/ \
  -H "Content-Type: application/json" \
  -d '{
    "version": "1.0.1",
    "entity_id": 123,
    "text": "Updated log text",
    "label": "warning"
  }'

# Delete a pattern
curl -X DELETE http://localhost:8000/api/patterns/1/

# Search patterns
curl "http://localhost:8000/api/patterns/?search=error&version=1.0.0"
```

#### Bulk Upload

Upload multiple patterns at once. Patterns are saved immediately, embeddings generated asynchronously:

```bash
curl -X POST http://localhost:8000/api/bulk-upload \
  -H "Content-Type: application/json" \
  -d '[
    {"version": "1.0", "entity_id": 1, "text": "First log...", "label": "info"},
    {"version": "1.0", "entity_id": 2, "text": "Second log...", "label": "error"},
    {"version": "1.0", "entity_id": 3, "text": "Third log...", "label": "warning"}
  ]'

# Response:
# {
#   "created": 3,
#   "created_ids": [1, 2, 3],
#   "errors": [],
#   "queued_for_embeddings": 3
# }
```

#### Check Queue Status

Monitor background embedding generation:

```bash
curl http://localhost:8000/api/queue-status

# Response:
# {
#   "queue_size": 157,
#   "worker_running": true
# }
```

#### Nearest Neighbors Search

Find similar patterns using semantic embeddings:

```bash
# POST request
curl -X POST http://localhost:8000/api/nn \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Error connecting to database",
    "version": "1.0.0"
  }'

# GET request
curl "http://localhost:8000/api/nn?text=Error%20connecting%20to%20database&version=1.0.0"

# Returns 15 nearest neighbors for each embedding model:
# {
#   "minilm_l6_v2_384": [
#     [0.123, 5, "1.0.0", 42, "Database connection failed..."],
#     ...
#   ],
#   "mpnet_base_v2_768": [...],
#   "jina_embeddings_v2_base_code": [...]
# }
```

### 3D Visualization

Access the interactive 3D visualization at `http://localhost:8000/visualize/`

Features:
- **Embedding Models**: Choose from MiniLM, MPNet, or Jina embeddings
- **Dimensionality Reduction**: PCA (fast) or t-SNE (better clusters)
- **Clustering Algorithms**:
  - HDBSCAN (auto-detect) - automatically finds dense clusters
  - DBSCAN - density-based clustering
  - K-Means - partition into K clusters
  - User defined labels - color by pattern labels
  - None - color by version
- **Interactive Controls**: Adjust clustering parameters in real-time
- **Hover Details**: View pattern information on hover
- **Cluster Representatives**: Points with smallest entity_id highlighted

### Django Admin

Access the admin interface at `http://localhost:8000/admin/`

Features:
- Full CRUD for patterns
- Filter by version and label
- Search across all fields
- View embeddings inline
- Embedding status indicators (✓/✗)

### Python API

Use Django ORM directly in Python:

```python
from fuz.models import Pattern, MiniLMEmbedding

# Create a pattern
pattern = Pattern.objects.create(
    version="1.0.0",
    entity_id=123,
    text="Log text here",
    label="error"
)

# Query patterns
patterns = Pattern.objects.filter(version="1.0.0")

# Access embeddings
minilm_emb = pattern.minilm_embedding.vector  # OneToOne relationship
has_embedding = hasattr(pattern, 'minilm_embedding')

# Search by similarity (requires embeddings)
from pgvector.django import CosineDistance
from fuz.embeddings import get_minilm_embedding

query_vector = get_minilm_embedding("Search text")
similar = (
    MiniLMEmbedding.objects
    .annotate(distance=CosineDistance('vector', query_vector))
    .order_by('distance')[:10]
)
```

### Management Commands

```bash
# Run migrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Generate embeddings for existing patterns (if needed)
python manage.py shell
>>> from fuz.background_tasks import task_queue
>>> from fuz.models import Pattern
>>> pattern_ids = Pattern.objects.values_list('id', flat=True)
>>> task_queue.enqueue_patterns(list(pattern_ids))

# Collect static files
python manage.py collectstatic

# Run development server
python manage.py runserver

# Access Django shell
python manage.py shell

# Access database shell
python manage.py dbshell
```

## API Endpoints

- `GET/POST /api/patterns/` - List/create patterns
- `GET/PUT/PATCH/DELETE /api/patterns/{id}/` - Retrieve/update/delete pattern
- `POST /api/bulk-upload` - Bulk upload patterns
- `GET /api/queue-status` - Check embedding queue status
- `GET/POST /api/nn` - Find nearest neighbors
- `GET /api/embeddings-3d` - Get 3D projection data
- `GET /api/versions` - List available versions
- `GET /api/docs/` - Swagger API documentation
- `GET /visualize/` - 3D visualization UI
- `GET /admin/` - Django admin interface

### Environment Variables

Configure via `.env` file or environment variables:

- `DB_NAME` - Database name (default: fuz_db)
- `DB_USER` - Database user (default: postgres)
- `DB_PASSWORD` - Database password
- `DB_HOST` - Database host (default: localhost)
- `DB_PORT` - Database port (default: 5432)
- `DJANGO_SECRET_KEY` - Django secret key
- `DEBUG` - Debug mode (default: False)

## Data Models

### Pattern
Main model for storing text logs with semantic search:
- `id` - Auto-incrementing primary key
- `version` - Version identifier (CharField, max 32 chars, indexed)
- `entity_id` - Entity identifier (IntegerField)
- `text` - Log text content (TextField)
- `label` - User-defined label for grouping (CharField, max 64 chars, nullable)
- Unique constraint on (version, entity_id)
- GIN index on text for pg_trgm similarity

### Embedding Models
Separate tables for each embedding model with HNSW indexes:

**MiniLMEmbedding** (384 dimensions)
- OneToOne relationship with Pattern
- Uses `all-MiniLM-L6-v2` model
- Fast and lightweight

**MPNetEmbedding** (768 dimensions)
- OneToOne relationship with Pattern
- Uses `all-mpnet-base-v2` model
- Better quality, slower

**JinaEmbedding** (768 dimensions)
- OneToOne relationship with Pattern
- Uses `jina-embeddings-v2-base-code` model
- Specialized for code/technical text

## Project Structure

```
fuz/
├── src/fuz/
│   ├── __init__.py              # Django initialization
│   ├── settings.py              # Django settings (full web stack)
│   ├── urls.py                  # URL routing
│   ├── models.py                # Data models (Pattern, Embeddings)
│   ├── serializers.py           # DRF serializers
│   ├── views.py                 # API views and endpoints
│   ├── admin.py                 # Django admin configuration
│   ├── background_tasks.py      # Lightweight task queue
│   ├── embeddings.py            # Embedding generation functions
│   ├── templates/               # HTML templates
│   │   └── visualize.html       # 3D visualization UI
│   └── migrations/              # Database migrations
├── manage.py                    # Django management script
├── docker-compose.yml           # Docker services (PostgreSQL 18)
├── init-db.sh                   # PostgreSQL initialization script
├── pyproject.toml               # Python dependencies (UV)
├── .env.example                 # Environment variables template
└── .gitignore                   # Git ignore rules
```

## Architecture

### Background Task System
- **Lightweight threading** - Uses Python's built-in `threading` and `queue` modules
- **No external dependencies** - No Celery/Redis/RabbitMQ required
- **Automatic processing** - Worker thread starts on Django initialization
- **Eventually consistent** - Embeddings generated asynchronously after pattern creation

### Vector Search
- **pgvector extension** - PostgreSQL native vector operations
- **HNSW indexes** - Fast approximate nearest neighbor search
- **Cosine distance** - Similarity metric for embeddings
- **Multiple models** - Compare results across different embedding approaches

### Caching
- **In-memory cache** - Django's `LocMemCache` for API responses
- **1-hour TTL** - Automatic cache expiration
- **Parameter-based keys** - Different parameters cached separately

## Development

```bash
# Install dependencies
uv sync

# Set up environment
cp .env.example .env
# Edit .env with your configuration

# Run development server
python manage.py runserver

# Run with auto-reload
python manage.py runserver --noreload  # disable if background tasks cause issues

# Run tests (if available)
python manage.py test

# Create new migration
python manage.py makemigrations

# Apply migrations
python manage.py migrate
```

## Performance Considerations

- **Bulk uploads** - Use `/api/bulk-upload` instead of individual creates
- **Queue monitoring** - Check `/api/queue-status` to avoid overwhelming the system
- **Embedding cache** - Embeddings are generated once and stored permanently
- **API caching** - Visualization endpoint responses cached for 1 hour
- **HNSW indexes** - Provide fast approximate search (trade-off: accuracy vs speed)

## License

MIT

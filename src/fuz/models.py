"""
Django models for fuz
"""

from django.db import models
from django.contrib.postgres.indexes import GinIndex
from pgvector.django import VectorField, CosineDistance, HnswIndex


class Pattern(models.Model):
    """
    Pattern model for storing text logs with similarity search capabilities.
    Uses pg_trgm for efficient text similarity queries.
    """

    entity_id = models.IntegerField(help_text="Entity identifier")
    version = models.CharField(max_length=32, help_text="Version identifier for the pattern")
    text = models.TextField(help_text="Log text content (up to ~100 lines)")
    label = models.CharField(max_length=64, blank=True, null=True, help_text="User-defined label for grouping patterns")

    class Meta:
        db_table = 'patterns'
        indexes = [
            # GIN index for pg_trgm similarity search on text field
            GinIndex(name='patterns_text_trgm_idx', fields=['text'], opclasses=['gin_trgm_ops']),
            # Composite index for version + entity_id lookups
            models.Index(fields=['version', 'entity_id'], name='patterns_version_entity_idx'),
        ]
        constraints = [
            # Unique constraint on version + entity_id combination
            models.UniqueConstraint(fields=['version', 'entity_id'], name='unique_version_entity'),
        ]

    def __str__(self):
        return f"Pattern {self.id} ({self.version}, entity={self.entity_id})"


class MiniLMEmbedding(models.Model):
    """all-MiniLM-L6-v2 embeddings (384 dimensions)"""

    pattern = models.OneToOneField(Pattern, on_delete=models.CASCADE, related_name='minilm_embedding', primary_key=True)
    vector = VectorField(dimensions=384)

    class Meta:
        db_table = 'minilm_embeddings'
        indexes = [
            # HNSW index for fast cosine similarity  search
            HnswIndex(
                name='minilm_vector_idx', fields=['vector'], m=16, ef_construction=64, opclasses=['vector_cosine_ops']
            ),
        ]

    def __str__(self):
        return f"MiniLM({self.pattern_id})"


class MPNetEmbedding(models.Model):
    """all-mpnet-base-v2 embeddings (768 dimensions)"""

    pattern = models.OneToOneField(Pattern, on_delete=models.CASCADE, related_name='mpnet_embedding', primary_key=True)
    vector = VectorField(dimensions=768)

    class Meta:
        db_table = 'mpnet_embeddings'
        indexes = [
            HnswIndex(
                name='mpnet_vector_idx', fields=['vector'], m=16, ef_construction=64, opclasses=['vector_cosine_ops']
            ),
        ]

    def __str__(self):
        return f"MPNet({self.pattern_id})"


class JinaEmbedding(models.Model):
    """jina-embeddings-v2-base-code embeddings (768 dimensions)"""

    pattern = models.OneToOneField(Pattern, on_delete=models.CASCADE, related_name='jina_embedding', primary_key=True)
    vector = VectorField(dimensions=768)

    class Meta:
        db_table = 'jina_embeddings'
        indexes = [
            HnswIndex(
                name='jina_vector_idx', fields=['vector'], m=16, ef_construction=64, opclasses=['vector_cosine_ops']
            ),
        ]

    def __str__(self):
        return f"Jina({self.pattern_id})"

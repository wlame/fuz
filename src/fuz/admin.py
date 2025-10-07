"""
Django admin configuration for fuz
"""

from django.contrib import admin
from fuz.models import Pattern, MiniLMEmbedding, MPNetEmbedding, JinaEmbedding


class MiniLMEmbeddingInline(admin.StackedInline):
    """Inline admin for MiniLM embeddings"""

    model = MiniLMEmbedding
    extra = 0
    readonly_fields = ['vector_display']
    classes = ['collapse']

    def vector_display(self, obj):
        """Display vector as truncated string"""
        if obj.vector is None:
            return "None"
        vector_str = str(obj.vector)
        if len(vector_str) > 200:
            return vector_str[:200] + "..."
        return vector_str

    vector_display.short_description = 'Vector (384 dimensions)'


class MPNetEmbeddingInline(admin.StackedInline):
    """Inline admin for MPNet embeddings"""

    model = MPNetEmbedding
    extra = 0
    readonly_fields = ['vector_display']
    classes = ['collapse']

    def vector_display(self, obj):
        """Display vector as truncated string"""
        if obj.vector is None:
            return "None"
        vector_str = str(obj.vector)
        if len(vector_str) > 200:
            return vector_str[:200] + "..."
        return vector_str

    vector_display.short_description = 'Vector (768 dimensions)'


class JinaEmbeddingInline(admin.StackedInline):
    """Inline admin for Jina embeddings"""

    model = JinaEmbedding
    extra = 0
    readonly_fields = ['vector_display']
    classes = ['collapse']

    def vector_display(self, obj):
        """Display vector as truncated string"""
        if obj.vector is None:
            return "None"
        vector_str = str(obj.vector)
        if len(vector_str) > 200:
            return vector_str[:200] + "..."
        return vector_str

    vector_display.short_description = 'Vector (768 dimensions)'


@admin.register(Pattern)
class PatternAdmin(admin.ModelAdmin):
    """Admin interface for Pattern model"""

    list_display = [
        'id',
        'version',
        'entity_id',
        'label',
        'text_preview',
        'has_minilm',
        'has_mpnet',
        'has_jina',
    ]
    list_filter = ['version', 'label']
    search_fields = ['version', 'entity_id', 'text', 'label']

    fieldsets = (('Basic Information', {'fields': ('version', 'entity_id', 'text', 'label')}),)

    inlines = [MiniLMEmbeddingInline, MPNetEmbeddingInline, JinaEmbeddingInline]

    def text_preview(self, obj):
        """Show truncated text in list view"""
        return obj.text[:100] + '...' if len(obj.text) > 100 else obj.text

    text_preview.short_description = 'Text Preview'

    def has_minilm(self, obj):
        """Check if MiniLM embedding exists"""
        try:
            return '✓' if obj.minilm_embedding else '✗'
        except:
            return '✗'

    has_minilm.short_description = 'MiniLM'

    def has_mpnet(self, obj):
        """Check if MPNet embedding exists"""
        try:
            return '✓' if obj.mpnet_embedding else '✗'
        except:
            return '✗'

    has_mpnet.short_description = 'MPNet'

    def has_jina(self, obj):
        """Check if Jina embedding exists"""
        try:
            return '✓' if obj.jina_embedding else '✗'
        except:
            return '✗'

    has_jina.short_description = 'Jina'

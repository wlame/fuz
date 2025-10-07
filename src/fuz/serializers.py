"""
DRF Serializers for fuz models
"""

from rest_framework import serializers
from fuz.models import Pattern


class PatternBasicSerializer(serializers.ModelSerializer):
    """Serializer for Pattern model without embeddings (for clusters endpoint)"""

    class Meta:
        model = Pattern
        fields = [
            'id',
            'version',
            'entity_id',
            'text',
            'label',
        ]


class PatternSerializer(serializers.ModelSerializer):
    """Serializer for Pattern model with embedding vectors from related models"""

    minilm_l6_v2_384 = serializers.SerializerMethodField()
    mpnet_base_v2_768 = serializers.SerializerMethodField()
    jina_embeddings_v2_base_code = serializers.SerializerMethodField()

    class Meta:
        model = Pattern
        fields = [
            'id',
            'version',
            'entity_id',
            'text',
            'label',
            'minilm_l6_v2_384',
            'mpnet_base_v2_768',
            'jina_embeddings_v2_base_code',
        ]
        read_only_fields = [
            'id',
            'minilm_l6_v2_384',
            'mpnet_base_v2_768',
            'jina_embeddings_v2_base_code',
        ]

    def get_minilm_l6_v2_384(self, obj):
        """Get MiniLM embedding vector if exists"""
        try:
            return obj.minilm_embedding.vector
        except:
            return None

    def get_mpnet_base_v2_768(self, obj):
        """Get MPNet embedding vector if exists"""
        try:
            return obj.mpnet_embedding.vector
        except:
            return None

    def get_jina_embeddings_v2_base_code(self, obj):
        """Get Jina embedding vector if exists"""
        try:
            return obj.jina_embedding.vector
        except:
            return None

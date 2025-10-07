"""
REST API views for fuz
"""

from rest_framework import viewsets, filters, status, serializers
from rest_framework.decorators import action, api_view
from rest_framework.response import Response
from django.db.models import Q
from django.shortcuts import render
from pgvector.django import CosineDistance
from drf_spectacular.utils import extend_schema, OpenApiParameter
from drf_spectacular.types import OpenApiTypes
from fuz.models import Pattern
from fuz.serializers import PatternSerializer, PatternBasicSerializer
from fuz.background_tasks import task_queue

# from sentence_transformers import SentenceTransformer


class PatternViewSet(viewsets.ModelViewSet):
    """
    ViewSet for Pattern CRUD operations

    Endpoints:
    - GET    /api/patterns/          - List all patterns
    - POST   /api/patterns/          - Create new pattern
    - GET    /api/patterns/{id}/     - Retrieve pattern by ID
    - PUT    /api/patterns/{id}/     - Update pattern
    - PATCH  /api/patterns/{id}/     - Partial update
    - DELETE /api/patterns/{id}/     - Delete pattern
    """

    queryset = Pattern.objects.all()
    serializer_class = PatternSerializer
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ['version', 'text', 'entity_id']
    ordering_fields = ['version', 'entity_id']
    ordering = ['-id']

    def get_queryset(self):
        """
        Optionally filter by version and entity_id query params
        """
        queryset = super().get_queryset()

        version = self.request.query_params.get('version')
        entity_id = self.request.query_params.get('entity_id')

        if version:
            queryset = queryset.filter(version=version)
        if entity_id:
            queryset = queryset.filter(entity_id=entity_id)

        return queryset

    def _generate_embeddings(self, text):
        """
        Generate embeddings for all three models.
        Returns dict with embedding field names as keys.
        """
        from fuz.embeddings import (
            get_minilm_embedding,
            get_mpnet_embedding,
            get_jina_embedding,
        )

        return {
            'minilm_l6_v2_384': get_minilm_embedding(text),
            'mpnet_base_v2_768': get_mpnet_embedding(text),
            'jina_embeddings_v2_base_code': get_jina_embedding(text),
        }

    def perform_create(self, serializer):
        """
        Called before saving a new Pattern instance.
        Save pattern and enqueue for background embedding generation.
        """
        pattern = serializer.save()  # Save pattern first

        # Enqueue for background embedding generation
        task_queue.enqueue_pattern(pattern.id)

    def perform_update(self, serializer):
        """
        Called before saving an updated Pattern instance.
        Enqueue for embedding regeneration if text changed.
        """
        text = serializer.validated_data.get('text')
        text_changed = text and text != serializer.instance.text

        pattern = serializer.save()

        if text_changed:
            # Enqueue for background embedding regeneration
            task_queue.enqueue_pattern(pattern.id)


class NearestNeighborsInputSerializer(serializers.Serializer):
    """Input serializer for nearest neighbors endpoint"""

    text = serializers.CharField(required=True, help_text="Text to find similar patterns for")
    version = serializers.CharField(required=False, allow_blank=True, help_text="Optional: Filter patterns by version")


class NearestNeighborsOutputSerializer(serializers.Serializer):
    """Output serializer for nearest neighbors endpoint"""

    minilm_l6_v2_384 = serializers.ListField(
        child=serializers.ListField(), help_text="List of [score, id, version, entity_id, text] tuples"
    )
    mpnet_base_v2_768 = serializers.ListField(
        child=serializers.ListField(), help_text="List of [score, id, version, entity_id, text] tuples"
    )
    jina_embeddings_v2_base_code = serializers.ListField(
        child=serializers.ListField(), help_text="List of [score, id, version, entity_id, text] tuples"
    )


@extend_schema(
    request=NearestNeighborsInputSerializer,
    responses={200: NearestNeighborsOutputSerializer},
    description="Find 15 nearest neighbors for given text using all embedding models",
    parameters=[
        OpenApiParameter(
            name='text',
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            description='Text to find similar patterns for (for GET requests)',
            required=False,
        ),
        OpenApiParameter(
            name='version',
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            description='Optional: Filter patterns by version (for GET requests)',
            required=False,
        ),
    ],
)
@api_view(['GET', 'POST'])
def nearest_neighbors(request):
    """
    Find nearest neighbors for given text using all embedding models.

    Supports both GET and POST methods:
    - GET: /api/nn?text=...&version=...
    - POST: /api/nn with JSON body {"text": "...", "version": "..."}
    """
    # Get parameters from either POST body or GET query params
    if request.method == 'POST':
        text = request.data.get('text')
        version = request.data.get('version')
    else:  # GET
        text = request.query_params.get('text')
        version = request.query_params.get('version')

    if not text:
        return Response({'error': 'text field is required'}, status=status.HTTP_400_BAD_REQUEST)

    # Generate embeddings for the input text
    from fuz.embeddings import (
        get_minilm_embedding,
        get_mpnet_embedding,
        get_jina_embedding,
    )

    minilm_emb = get_minilm_embedding(text)
    mpnet_emb = get_mpnet_embedding(text)
    jina_emb = get_jina_embedding(text)

    # Build base queryset with optional version filter
    base_queryset = Pattern.objects.all()
    if version:
        base_queryset = base_queryset.filter(version=version)

    # Find nearest neighbors for each embedding model
    from fuz.models import MiniLMEmbedding, MPNetEmbedding, JinaEmbedding

    results = {}

    # MiniLM (384)
    if minilm_emb is not None:
        minilm_results = (
            MiniLMEmbedding.objects.filter(pattern__in=base_queryset)
            .select_related('pattern')
            .annotate(distance=CosineDistance('vector', minilm_emb))
            .order_by('distance')[:15]
        )

        results['minilm_l6_v2_384'] = [
            [float(e.distance), e.pattern.id, e.pattern.version, e.pattern.entity_id, e.pattern.text]
            for e in minilm_results
        ]
    else:
        results['minilm_l6_v2_384'] = []

    # MPNet (768)
    if mpnet_emb is not None:
        mpnet_results = (
            MPNetEmbedding.objects.filter(pattern__in=base_queryset)
            .select_related('pattern')
            .annotate(distance=CosineDistance('vector', mpnet_emb))
            .order_by('distance')[:15]
        )

        results['mpnet_base_v2_768'] = [
            [float(e.distance), e.pattern.id, e.pattern.version, e.pattern.entity_id, e.pattern.text]
            for e in mpnet_results
        ]
    else:
        results['mpnet_base_v2_768'] = []

    # Jina Code (768)
    if jina_emb is not None:
        jina_results = (
            JinaEmbedding.objects.filter(pattern__in=base_queryset)
            .select_related('pattern')
            .annotate(distance=CosineDistance('vector', jina_emb))
            .order_by('distance')[:15]
        )

        results['jina_embeddings_v2_base_code'] = [
            [float(e.distance), e.pattern.id, e.pattern.version, e.pattern.entity_id, e.pattern.text]
            for e in jina_results
        ]
    else:
        results['jina_embeddings_v2_base_code'] = []

    return Response(results)


@api_view(['GET'])
def embeddings_3d(request):
    """
    Get 3D projection of embeddings for visualization.

    Query params:
    - model: which embedding to visualize (minilm/mpnet/jina), default: minilm
    - version: optional version filter
    - method: reduction method (pca/tsne), default: pca
    - clustering: clustering algorithm (none/hdbscan/dbscan/kmeans), default: hdbscan
    - n_clusters: number of clusters for kmeans (default: 5)
    - min_cluster_size: HDBSCAN min cluster size (default: auto = len/20)
    - min_samples: HDBSCAN/DBSCAN min samples (default: auto for HDBSCAN, 2 for DBSCAN)
    - eps: DBSCAN epsilon distance (default: 0.5)
    """
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.cluster import DBSCAN, KMeans
    import hdbscan as hdb
    from django.core.cache import cache
    from fuz.models import MiniLMEmbedding, MPNetEmbedding, JinaEmbedding

    model_param = request.query_params.get('model', 'minilm')
    version = request.query_params.get('version', '')
    method = request.query_params.get('method', 'pca')
    clustering = request.query_params.get('clustering', 'hdbscan')
    n_clusters = int(request.query_params.get('n_clusters', 5))

    # Clustering parameters
    min_cluster_size_param = request.query_params.get('min_cluster_size', '')
    min_samples_param = request.query_params.get('min_samples', '')
    eps_param = request.query_params.get('eps', '')

    # Create cache key from all parameters
    cache_key = f'embeddings_3d:{model_param}:{version}:{method}:{clustering}:{n_clusters}:{min_cluster_size_param}:{min_samples_param}:{eps_param}'

    # Try to get from cache
    cached_response = cache.get(cache_key)
    if cached_response is not None:
        return Response(cached_response)

    # Map parameter to embedding model
    model_map = {
        'minilm': MiniLMEmbedding,
        'mpnet': MPNetEmbedding,
        'jina': JinaEmbedding,
    }

    embedding_model = model_map.get(model_param, MiniLMEmbedding)

    # Build base queryset with optional version filter
    base_queryset = Pattern.objects.all()
    if version:
        base_queryset = base_queryset.filter(version=version)

    # Get embeddings with related patterns
    embeddings_qs = embedding_model.objects.filter(pattern__in=base_queryset).select_related('pattern')
    embeddings_list = list(embeddings_qs)

    if len(embeddings_list) < 2:
        return Response(
            {'error': 'Need at least 2 patterns with embeddings for visualization'}, status=status.HTTP_400_BAD_REQUEST
        )

    # Extract embeddings and metadata
    embeddings = []
    metadata = []

    for emb_obj in embeddings_list:
        embeddings.append(emb_obj.vector)
        metadata.append(
            {
                'id': emb_obj.pattern.id,
                'version': emb_obj.pattern.version,
                'entity_id': emb_obj.pattern.entity_id,
                'label': emb_obj.pattern.label or '',
                'text_preview': emb_obj.pattern.text[:500] + '...'
                if len(emb_obj.pattern.text) > 500
                else emb_obj.pattern.text,
            }
        )

    if len(embeddings) < 2:
        return Response(
            {'error': 'Need at least 2 patterns with embeddings for visualization'}, status=status.HTTP_400_BAD_REQUEST
        )

    # Convert to numpy array
    X = np.array(embeddings)

    # Apply dimensionality reduction
    if method == 'tsne' and len(embeddings) >= 3:
        reducer = TSNE(n_components=3, random_state=42, perplexity=min(30, len(embeddings) - 1))
        coords_3d = reducer.fit_transform(X)
    else:  # PCA
        reducer = PCA(n_components=3)
        coords_3d = reducer.fit_transform(X)

    # Apply clustering algorithm
    cluster_labels = None
    if clustering == 'labels':
        # Use user-defined labels as clusters
        # Map unique labels to cluster numbers
        unique_labels = sorted(set(meta['label'] for meta in metadata if meta['label']))
        label_to_cluster = {label: idx for idx, label in enumerate(unique_labels)}
        # Assign cluster labels based on labels (-1 for no label)
        cluster_labels = np.array(
            [label_to_cluster.get(meta['label'], -1) if meta['label'] else -1 for meta in metadata]
        )
    elif clustering == 'hdbscan' and len(embeddings) >= 3:
        # Auto-calculate or use provided parameters
        if min_cluster_size_param:
            min_cluster_size = int(min_cluster_size_param)
        else:
            min_cluster_size = max(2, min(5, len(embeddings) // 40))

        if min_samples_param:
            min_samples = int(min_samples_param)
        else:
            min_samples = max(1, min(3, len(embeddings) // 50))

        clusterer = hdb.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=0.0,
            cluster_selection_method='eom',  # Excess of Mass - better for varying density
        )
        cluster_labels = clusterer.fit_predict(coords_3d)
    elif clustering == 'dbscan' and len(embeddings) >= 3:
        eps = float(eps_param) if eps_param else 0.5
        min_samples = int(min_samples_param) if min_samples_param else 2
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = clusterer.fit_predict(coords_3d)
    elif clustering == 'kmeans' and len(embeddings) >= n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(coords_3d)

    # Find representative (smallest entity_id) for each cluster
    cluster_representatives = {}
    if cluster_labels is not None:
        for i, label in enumerate(cluster_labels):
            if label >= 0:  # Ignore noise points (-1)
                entity_id = metadata[i]['entity_id']
                if label not in cluster_representatives or entity_id < cluster_representatives[label]['entity_id']:
                    cluster_representatives[label] = {'entity_id': entity_id, 'index': i}

    representative_indices = set(rep['index'] for rep in cluster_representatives.values())

    # Prepare response
    points = []
    for i, (coord, meta) in enumerate(zip(coords_3d, metadata)):
        point = {'x': float(coord[0]), 'y': float(coord[1]), 'z': float(coord[2]), **meta}
        if cluster_labels is not None:
            point['cluster'] = int(cluster_labels[i])
        point['is_representative'] = i in representative_indices
        points.append(point)

    response_data = {
        'points': points,
        'method': method,
        'model': model_param,
        'clustering': clustering,
        'total_points': len(points),
        'n_clusters': int(len(set(cluster_labels))) if cluster_labels is not None else 0,
    }

    # Cache the response for 1 hour (3600 seconds)
    cache.set(cache_key, response_data, 3600)

    return Response(response_data)


@extend_schema(
    request=PatternSerializer(many=True),
    responses={201: OpenApiTypes.OBJECT},
    description="Bulk upload multiple patterns. Patterns saved immediately, embeddings generated in background.",
)
@api_view(['POST'])
def bulk_upload(request):
    """
    Bulk upload multiple patterns at once.
    Patterns are saved immediately, embeddings generated in background.

    Request body: list of pattern objects
    [
        {"version": "1.0", "entity_id": 1, "text": "...", "label": "..."},
        {"version": "1.0", "entity_id": 2, "text": "...", "label": "..."},
        ...
    ]
    """
    if not isinstance(request.data, list):
        return Response({'error': 'Request body must be a list of pattern objects'}, status=status.HTTP_400_BAD_REQUEST)

    created_ids = []
    errors = []

    for idx, item in enumerate(request.data):
        serializer = PatternSerializer(data=item)
        if serializer.is_valid():
            pattern = serializer.save()
            created_ids.append(pattern.id)
        else:
            errors.append({'index': idx, 'errors': serializer.errors})

    # Enqueue all created patterns for embedding generation
    if created_ids:
        task_queue.enqueue_patterns(created_ids)

    return Response(
        {
            'created': len(created_ids),
            'created_ids': created_ids,
            'errors': errors,
            'queued_for_embeddings': len(created_ids),
        },
        status=status.HTTP_201_CREATED if created_ids else status.HTTP_400_BAD_REQUEST,
    )


@extend_schema(
    responses={200: OpenApiTypes.OBJECT},
    description="Get current embedding generation queue status",
)
@api_view(['GET'])
def queue_status(request):
    """Get current embedding generation queue status"""
    return Response(
        {
            'queue_size': task_queue.get_queue_size(),
            'worker_running': task_queue.running,
        }
    )


@extend_schema(
    parameters=[
        OpenApiParameter(
            name='model',
            type=str,
            enum=['minilm', 'mpnet', 'jina'],
            description='Embedding model: MiniLM (fast, 384d), MPNet (balanced, 768d), or Jina (code-specialized, 768d)',
            default='minilm',
        ),
        OpenApiParameter(name='version', type=str, description='Filter patterns by version identifier'),
        OpenApiParameter(
            name='method',
            type=str,
            enum=['pca', 'tsne'],
            description='Dimensionality reduction: PCA (fast) or t-SNE (better clusters, slower)',
            default='pca',
        ),
        OpenApiParameter(
            name='clustering',
            type=str,
            enum=['hdbscan', 'dbscan', 'kmeans', 'tags', 'none'],
            description='Clustering algorithm: HDBSCAN (auto-detect), DBSCAN (density-based), K-Means (partition-based), User labels, or None (by version)',
            default='hdbscan',
        ),
        OpenApiParameter(
            name='n_clusters', type=int, description='Number of clusters for K-Means algorithm', default=5
        ),
        OpenApiParameter(
            name='min_cluster_size', type=int, description='HDBSCAN: minimum cluster size (auto if not specified)'
        ),
        OpenApiParameter(
            name='min_samples',
            type=int,
            description='HDBSCAN/DBSCAN: minimum samples for core points (auto for HDBSCAN, 2 for DBSCAN)',
        ),
        OpenApiParameter(
            name='eps', type=float, description='DBSCAN: maximum distance between neighbors (default: 0.5)'
        ),
    ],
    responses={200: OpenApiTypes.OBJECT},
    description="Get patterns grouped by clusters with same parameters as visualization",
)
@api_view(['GET'])
def clusters(request):
    """
    Get patterns grouped by clusters.

    Query params (same as visualization):
    - model: which embedding to use (minilm/mpnet/jina), default: minilm
    - version: optional version filter
    - method: reduction method (pca/tsne), default: pca
    - clustering: clustering algorithm (none/hdbscan/dbscan/kmeans/tags), default: hdbscan
    - n_clusters: number of clusters for kmeans (default: 5)
    - min_cluster_size: HDBSCAN min cluster size (default: auto)
    - min_samples: HDBSCAN/DBSCAN min samples (default: auto/2)
    - eps: DBSCAN epsilon distance (default: 0.5)

    Returns:
    {
        "clusters": [
            {
                "cluster_id": 0,
                "size": 15,
                "representative": {...},  // Pattern with smallest entity_id
                "patterns": [...]  // List of Pattern objects
            },
            ...
        ],
        "noise": [...],  // Patterns not in any cluster (cluster_id = -1)
        "metadata": {
            "model": "minilm",
            "method": "pca",
            "clustering": "hdbscan",
            "n_clusters": 5,
            "total_patterns": 100
        }
    }
    """
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.cluster import DBSCAN, KMeans
    import hdbscan as hdb
    from django.core.cache import cache
    from fuz.models import MiniLMEmbedding, MPNetEmbedding, JinaEmbedding

    model_param = request.query_params.get('model', 'minilm')
    version = request.query_params.get('version', '')
    method = request.query_params.get('method', 'pca')
    clustering = request.query_params.get('clustering', 'hdbscan')
    n_clusters_param = int(request.query_params.get('n_clusters', 5))

    # Clustering parameters
    min_cluster_size_param = request.query_params.get('min_cluster_size', '')
    min_samples_param = request.query_params.get('min_samples', '')
    eps_param = request.query_params.get('eps', '')

    # Create cache key
    cache_key = f'clusters:{model_param}:{version}:{method}:{clustering}:{n_clusters_param}:{min_cluster_size_param}:{min_samples_param}:{eps_param}'

    # Try to get from cache
    cached_response = cache.get(cache_key)
    if cached_response is not None:
        return Response(cached_response)

    # Map parameter to embedding model
    model_map = {
        'minilm': MiniLMEmbedding,
        'mpnet': MPNetEmbedding,
        'jina': JinaEmbedding,
    }

    embedding_model = model_map.get(model_param, MiniLMEmbedding)

    # Build base queryset with optional version filter
    base_queryset = Pattern.objects.all()
    if version:
        base_queryset = base_queryset.filter(version=version)

    # Get embeddings with related patterns
    embeddings_qs = embedding_model.objects.filter(pattern__in=base_queryset).select_related('pattern')
    embeddings_list = list(embeddings_qs)

    if len(embeddings_list) < 2:
        return Response(
            {'error': 'Need at least 2 patterns with embeddings for clustering'}, status=status.HTTP_400_BAD_REQUEST
        )

    # Extract embeddings and keep pattern references
    embeddings = []
    patterns = []

    for emb_obj in embeddings_list:
        embeddings.append(emb_obj.vector)
        patterns.append(emb_obj.pattern)

    # Convert to numpy array
    X = np.array(embeddings)

    # Apply dimensionality reduction
    if method == 'tsne' and len(embeddings) >= 3:
        reducer = TSNE(n_components=3, random_state=42, perplexity=min(30, len(embeddings) - 1))
        coords_3d = reducer.fit_transform(X)
    else:  # PCA
        reducer = PCA(n_components=3)
        coords_3d = reducer.fit_transform(X)

    # Apply clustering algorithm
    cluster_labels = None
    if clustering == 'tags':
        # Use user-defined labels as clusters
        unique_labels = sorted(set(p.label for p in patterns if p.label))
        label_to_cluster = {label: idx for idx, label in enumerate(unique_labels)}
        cluster_labels = np.array([label_to_cluster.get(p.label, -1) if p.label else -1 for p in patterns])
    elif clustering == 'hdbscan' and len(embeddings) >= 3:
        if min_cluster_size_param:
            min_cluster_size = int(min_cluster_size_param)
        else:
            min_cluster_size = max(2, min(5, len(embeddings) // 20))

        if min_samples_param:
            min_samples = int(min_samples_param)
        else:
            min_samples = max(1, min(3, len(embeddings) // 50))

        clusterer = hdb.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=0.0,
            cluster_selection_method='eom',
        )
        cluster_labels = clusterer.fit_predict(coords_3d)
    elif clustering == 'dbscan' and len(embeddings) >= 3:
        eps = float(eps_param) if eps_param else 0.5
        min_samples = int(min_samples_param) if min_samples_param else 2
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = clusterer.fit_predict(coords_3d)
    elif clustering == 'kmeans' and len(embeddings) >= n_clusters_param:
        clusterer = KMeans(n_clusters=n_clusters_param, random_state=42)
        cluster_labels = clusterer.fit_predict(coords_3d)

    # Group patterns by cluster
    from collections import defaultdict

    cluster_groups = defaultdict(list)

    if cluster_labels is not None:
        for i, label in enumerate(cluster_labels):
            cluster_groups[int(label)].append(patterns[i])
    else:
        # No clustering, group by version
        for pattern in patterns:
            cluster_groups[pattern.version].append(pattern)

    # Build response
    clusters_data = []
    noise_data = []

    # Build mapping of pattern_id to embedding index for distance calculations
    pattern_to_index = {patterns[i].id: i for i in range(len(patterns))}

    # Process each cluster
    cluster_representatives_map = {}  # cluster_id -> representative pattern
    for cluster_id in sorted(cluster_groups.keys()):
        if cluster_id == -1:
            # Noise points
            noise_data = [PatternBasicSerializer(p).data for p in cluster_groups[cluster_id]]
        else:
            # Find representative (smallest entity_id)
            cluster_patterns = cluster_groups[cluster_id]
            representative = min(cluster_patterns, key=lambda p: p.entity_id)
            cluster_representatives_map[cluster_id] = representative

            # Calculate distances using embeddings from X array
            pattern_indices = [pattern_to_index[p.id] for p in cluster_patterns]
            cluster_embeddings = X[pattern_indices]

            # Calculate distance metrics for the cluster
            max_distance = {}
            median_distance = {}
            percentile_90_distance = {}

            if len(cluster_embeddings) > 1:
                from sklearn.metrics.pairwise import cosine_distances
                from scipy.spatial.distance import pdist, squareform

                # Calculate all distance metrics
                distance_metrics = {
                    'cosine': squareform(pdist(cluster_embeddings, metric='cosine')),
                    'euclidean': squareform(pdist(cluster_embeddings, metric='euclidean')),
                    'manhattan': squareform(pdist(cluster_embeddings, metric='cityblock')),
                    'minkowski': squareform(pdist(cluster_embeddings, metric='minkowski', p=3)),
                    'chebyshev': squareform(pdist(cluster_embeddings, metric='chebyshev')),
                }

                for metric_name, dist_matrix in distance_metrics.items():
                    # Get upper triangle (excludes diagonal and duplicates)
                    upper_triangle_indices = np.triu_indices_from(dist_matrix, k=1)
                    pairwise_distances = dist_matrix[upper_triangle_indices]

                    max_distance[metric_name] = float(np.max(pairwise_distances))
                    median_distance[metric_name] = float(np.median(pairwise_distances))
                    percentile_90_distance[metric_name] = float(np.percentile(pairwise_distances, 90))

            clusters_data.append(
                {
                    'cluster_id': cluster_id,
                    'size': len(cluster_patterns),
                    'representative': PatternBasicSerializer(representative).data,
                    'patterns': [PatternBasicSerializer(p).data for p in cluster_patterns],
                    'max_distance': max_distance,
                    'median_distance': median_distance,
                    'percentile_90_distance': percentile_90_distance,
                }
            )

    # Calculate closest_representative for each cluster
    # This is the minimum distance from each cluster's representative to all other cluster representatives
    if len(cluster_representatives_map) > 1:
        # Get embeddings of all representatives
        rep_pattern_ids = [rep.id for rep in cluster_representatives_map.values()]
        rep_indices = [pattern_to_index[pid] for pid in rep_pattern_ids]
        rep_embeddings = X[rep_indices]

        # Calculate pairwise distances between representatives for all metrics
        from scipy.spatial.distance import pdist, squareform

        rep_distance_metrics = {
            'cosine': squareform(pdist(rep_embeddings, metric='cosine')),
            'euclidean': squareform(pdist(rep_embeddings, metric='euclidean')),
            'manhattan': squareform(pdist(rep_embeddings, metric='cityblock')),
            'minkowski': squareform(pdist(rep_embeddings, metric='minkowski', p=3)),
            'chebyshev': squareform(pdist(rep_embeddings, metric='chebyshev')),
        }

        # For each cluster, find minimum distance to other representatives
        cluster_ids_ordered = sorted(cluster_representatives_map.keys())
        for i, cluster_id in enumerate(cluster_ids_ordered):
            closest_dist = {}

            for metric_name, rep_distances in rep_distance_metrics.items():
                # Get distances to all other representatives (exclude self with infinity)
                distances_to_others = rep_distances[i].copy()
                distances_to_others[i] = np.inf  # Exclude self
                closest_dist[metric_name] = float(np.min(distances_to_others))

            # Find the corresponding cluster_data entry and update it
            for cluster_data in clusters_data:
                if cluster_data['cluster_id'] == cluster_id:
                    cluster_data['closest_representative'] = closest_dist
                    break
    else:
        # Only one cluster or no clusters
        for cluster_data in clusters_data:
            cluster_data['closest_representative'] = None

    response_data = {
        'clusters': clusters_data,
        'noise': noise_data,
        'metadata': {
            'model': model_param,
            'method': method,
            'clustering': clustering,
            'n_clusters': len(clusters_data),
            'total_patterns': len(patterns),
            'noise_count': len(noise_data),
        },
    }

    # Cache for 1 hour
    cache.set(cache_key, response_data, 3600)

    return Response(response_data)


@extend_schema(
    responses={200: OpenApiTypes.OBJECT},
    description="Get list of available versions for filtering",
)
@api_view(['GET'])
def available_versions(request):
    """Get list of available versions for filtering"""
    versions = Pattern.objects.values_list('version', flat=True).distinct().order_by('version')
    return Response({'versions': list(versions)})


@extend_schema(
    parameters=[
        OpenApiParameter(name='id1', type=int, description='ID of first pattern'),
        OpenApiParameter(name='id2', type=int, description='ID of second pattern'),
        OpenApiParameter(name='version1', type=str, description='Version of first pattern (alternative to id1)'),
        OpenApiParameter(name='entity_id1', type=str, description='Entity ID of first pattern (alternative to id1)'),
        OpenApiParameter(name='version2', type=str, description='Version of second pattern (alternative to id2)'),
        OpenApiParameter(name='entity_id2', type=str, description='Entity ID of second pattern (alternative to id2)'),
    ],
    responses={200: OpenApiTypes.OBJECT},
    description="Calculate cosine distance between two patterns using all three embedding models",
)
@api_view(['GET'])
def pattern_distance(request):
    """
    Calculate distance between two patterns using all embedding models.

    Patterns can be specified by:
    - id1 and id2 (pattern IDs), or
    - version1+entity_id1 and version2+entity_id2

    Returns distances for all three models and pattern data (without embeddings).
    """
    from fuz.models import MiniLMEmbedding, MPNetEmbedding, JinaEmbedding
    from sklearn.metrics.pairwise import cosine_distances
    import numpy as np

    # Get pattern 1
    id1 = request.query_params.get('id1')
    version1 = request.query_params.get('version1')
    entity_id1 = request.query_params.get('entity_id1')

    if id1:
        try:
            pattern1 = Pattern.objects.get(id=id1)
        except Pattern.DoesNotExist:
            return Response({'error': f'Pattern with id={id1} not found'}, status=status.HTTP_404_NOT_FOUND)
    elif version1 and entity_id1:
        try:
            pattern1 = Pattern.objects.get(version=version1, entity_id=entity_id1)
        except Pattern.DoesNotExist:
            return Response(
                {'error': f'Pattern with version={version1}, entity_id={entity_id1} not found'},
                status=status.HTTP_404_NOT_FOUND,
            )
    else:
        return Response(
            {'error': 'Must specify either id1 or (version1 and entity_id1)'}, status=status.HTTP_400_BAD_REQUEST
        )

    # Get pattern 2
    id2 = request.query_params.get('id2')
    version2 = request.query_params.get('version2')
    entity_id2 = request.query_params.get('entity_id2')

    if id2:
        try:
            pattern2 = Pattern.objects.get(id=id2)
        except Pattern.DoesNotExist:
            return Response({'error': f'Pattern with id={id2} not found'}, status=status.HTTP_404_NOT_FOUND)
    elif version2 and entity_id2:
        try:
            pattern2 = Pattern.objects.get(version=version2, entity_id=entity_id2)
        except Pattern.DoesNotExist:
            return Response(
                {'error': f'Pattern with version={version2}, entity_id={entity_id2} not found'},
                status=status.HTTP_404_NOT_FOUND,
            )
    else:
        return Response(
            {'error': 'Must specify either id2 or (version2 and entity_id2)'}, status=status.HTTP_400_BAD_REQUEST
        )

    # Calculate distances for each embedding model
    from scipy.spatial.distance import cdist

    def calculate_distances(vec1, vec2):
        """Calculate multiple distance metrics between two vectors"""
        vec1_2d = np.array([vec1])
        vec2_2d = np.array([vec2])

        return {
            'cosine': float(cosine_distances(vec1_2d, vec2_2d)[0][0]),
            'euclidean': float(cdist(vec1_2d, vec2_2d, metric='euclidean')[0][0]),
            'manhattan': float(cdist(vec1_2d, vec2_2d, metric='cityblock')[0][0]),
            'minkowski': float(cdist(vec1_2d, vec2_2d, metric='minkowski', p=3)[0][0]),
            'chebyshev': float(cdist(vec1_2d, vec2_2d, metric='chebyshev')[0][0]),
        }

    distances = {}

    # MiniLM
    try:
        emb1 = MiniLMEmbedding.objects.get(pattern=pattern1)
        emb2 = MiniLMEmbedding.objects.get(pattern=pattern2)
        distances['minilm_l6_v2_384'] = calculate_distances(emb1.vector, emb2.vector)
    except MiniLMEmbedding.DoesNotExist:
        distances['minilm_l6_v2_384'] = None

    # MPNet
    try:
        emb1 = MPNetEmbedding.objects.get(pattern=pattern1)
        emb2 = MPNetEmbedding.objects.get(pattern=pattern2)
        distances['mpnet_base_v2_768'] = calculate_distances(emb1.vector, emb2.vector)
    except MPNetEmbedding.DoesNotExist:
        distances['mpnet_base_v2_768'] = None

    # Jina
    try:
        emb1 = JinaEmbedding.objects.get(pattern=pattern1)
        emb2 = JinaEmbedding.objects.get(pattern=pattern2)
        distances['jina_embeddings_v2_base_code'] = calculate_distances(emb1.vector, emb2.vector)
    except JinaEmbedding.DoesNotExist:
        distances['jina_embeddings_v2_base_code'] = None

    return Response(
        {
            'pattern1': PatternBasicSerializer(pattern1).data,
            'pattern2': PatternBasicSerializer(pattern2).data,
            'distances': distances,
        }
    )


def visualize_embeddings(request):
    """Render the 3D visualization page"""
    return render(request, 'visualize.html')

"""
URL configuration for fuz project
"""

from django.contrib import admin
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView, SpectacularRedocView
from fuz.views import (
    PatternViewSet,
    nearest_neighbors,
    embeddings_3d,
    available_versions,
    visualize_embeddings,
    bulk_upload,
    queue_status,
    clusters,
    pattern_distance,
)

# REST API router
router = DefaultRouter()
router.register(r'patterns', PatternViewSet, basename='pattern')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include(router.urls)),
    path('api/nn', nearest_neighbors, name='nearest-neighbors'),
    path('api/embeddings-3d', embeddings_3d, name='embeddings-3d'),
    path('api/clusters', clusters, name='clusters'),
    path('api/distance', pattern_distance, name='pattern-distance'),
    path('api/versions', available_versions, name='available-versions'),
    path('api/bulk-upload', bulk_upload, name='bulk-upload'),
    path('api/queue-status', queue_status, name='queue-status'),
    path('api-auth/', include('rest_framework.urls')),
    # Visualization
    path('visualize/', visualize_embeddings, name='visualize'),
    # Swagger/OpenAPI
    path('api/schema/', SpectacularAPIView.as_view(), name='schema'),
    path('api/docs/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
    path('api/redoc/', SpectacularRedocView.as_view(url_name='schema'), name='redoc'),
]

"""
Lightweight background task processor for embedding generation
"""

import threading
import queue
import time
import logging
from django.db import connection

logger = logging.getLogger(__name__)


class EmbeddingTaskQueue:
    """
    Simple background task queue for processing embeddings.
    Uses a single background thread to avoid overwhelming the system.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.task_queue = queue.Queue()
        self.worker_thread = None
        self.running = False
        self._initialized = True
        self.start_worker()

    def start_worker(self):
        """Start the background worker thread"""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker, daemon=True)
            self.worker_thread.start()
            logger.info("Embedding task worker started")

    def stop_worker(self):
        """Stop the background worker thread"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
            logger.info("Embedding task worker stopped")

    def _worker(self):
        """Background worker that processes tasks from the queue"""
        while self.running:
            try:
                # Get task with timeout to allow checking self.running
                try:
                    pattern_id = self.task_queue.get(timeout=1)
                except queue.Empty:
                    continue

                # Process the task
                try:
                    self._generate_embeddings_for_pattern(pattern_id)
                    logger.info(f"Generated embeddings for Pattern {pattern_id}")
                except Exception as e:
                    logger.error(f"Error generating embeddings for Pattern {pattern_id}: {e}")
                finally:
                    self.task_queue.task_done()

            except Exception as e:
                logger.error(f"Worker error: {e}")
                time.sleep(1)

    def _generate_embeddings_for_pattern(self, pattern_id):
        """Generate embeddings for a specific pattern"""
        # Import here to avoid circular imports
        from fuz.models import Pattern, MiniLMEmbedding, MPNetEmbedding, JinaEmbedding
        from fuz.embeddings import (
            get_minilm_embedding,
            get_mpnet_embedding,
            get_jina_embedding,
        )

        # Close old database connections (important for threads)
        connection.close()

        try:
            pattern = Pattern.objects.get(id=pattern_id)
            text = pattern.text

            # Generate embeddings
            minilm_emb = get_minilm_embedding(text)
            mpnet_emb = get_mpnet_embedding(text)
            jina_emb = get_jina_embedding(text)

            # Save to database
            if minilm_emb is not None:
                MiniLMEmbedding.objects.update_or_create(pattern=pattern, defaults={'vector': minilm_emb})

            if mpnet_emb is not None:
                MPNetEmbedding.objects.update_or_create(pattern=pattern, defaults={'vector': mpnet_emb})

            if jina_emb is not None:
                JinaEmbedding.objects.update_or_create(pattern=pattern, defaults={'vector': jina_emb})

        finally:
            # Clean up database connection
            connection.close()

    def enqueue_pattern(self, pattern_id):
        """Add a pattern to the queue for embedding generation"""
        self.task_queue.put(pattern_id)
        logger.debug(f"Enqueued Pattern {pattern_id} for embedding generation")

    def enqueue_patterns(self, pattern_ids):
        """Add multiple patterns to the queue"""
        for pattern_id in pattern_ids:
            self.enqueue_pattern(pattern_id)
        logger.info(f"Enqueued {len(pattern_ids)} patterns for embedding generation")

    def get_queue_size(self):
        """Get current queue size"""
        return self.task_queue.qsize()


# Global singleton instance
task_queue = EmbeddingTaskQueue()

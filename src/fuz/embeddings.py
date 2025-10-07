import numpy as np
import json
import torch
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel


minilm_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
mpnet_model_name = 'sentence-transformers/all-mpnet-base-v2'
jinacode_model_name = 'jinaai/jina-embeddings-v2-base-code'


# load models on startup
minilm_model = SentenceTransformer(minilm_model_name)  # 90M
mpnet_model = SentenceTransformer(mpnet_model_name)  # 438M
jina_model = SentenceTransformer(jinacode_model_name, trust_remote_code=True)  # 322M
jina_model.max_seq_length = 2048  # control your input sequence length up to 8192


def get_minilm_embedding(text):
    return minilm_model.encode([text], batch_size=64, normalize_embeddings=True)[0]


def get_mpnet_embedding(text):
    return mpnet_model.encode([text], batch_size=64, normalize_embeddings=True)[0]


def get_jina_embedding(text):
    return jina_model.encode([text], batch_size=64, normalize_embeddings=True)[0]

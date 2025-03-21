#!/usr/bin/env python3

import ollama
import redis
import numpy as np
import os
import fitz
import argparse
import time

REDIS_HOST = "localhost"
REDIS_PORT = 6380

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

CHUNK_SIZE = 300 # should this be a list? go through different chunk sizes ?

EMBEDDING_MODEL = "nomic-embed-text" # same Q as above - list of 3 diff embedding models

def get_redis_connection():
    pass

def clear_redis_db():
    pass

def create_vector_index():
    pass

def chunk():
    pass

def get_embedding():
    pass


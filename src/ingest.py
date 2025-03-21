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
REDIS_DB = 0

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

CHUNK_SIZE = 300 # should this be a list? go through different chunk sizes ?

EMBEDDING_MODEL = "nomic-embed-text" # same Q as above - list of 3 diff embedding models

def get_redis_connection():
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB
    )

def clear_redis_db(redis_client):
    print("Clearing Redis database...")
    redis_client.flushdb()
    print("Database cleared")

def create_vector_index():
    pass

def chunk():
    pass

def get_embedding():
    pass

def main():

    redis_client = get_redis_connection()

    # time how long it takes to query
    start_time = time.time()


    elapsed = time.time() - start_time


if __name__ == "__main__":
    main()
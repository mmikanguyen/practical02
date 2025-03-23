import subprocess

# List of file paths to execute
ingest_files = ['vector_dbs/chroma_ingest.py', 'vector_dbs/redis_ingest.py', 'vector_dbs/mongo_ingest.py']

for i in range(len(ingest_files)):
    print(f"Running {ingest_files[i]} - Execution {i + 1}")
    subprocess.run(['python', ingest_files[i]])
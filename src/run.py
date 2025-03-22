import subprocess

# List of file paths to execute
ingest_files = ['chroma_ingest.py', 'redis_ingest.py', 'mongo_ingest.py']
search_files = ['chroma_search.py', 'redis_search.py', 'mongo_search.py']

for i in range(len(ingest_files)):
    print(f"Running {ingest_files[i]} - Execution {i + 1}")
    subprocess.run(['python', ingest_files[i]])
    print(f"Running {search_files[i]} - Execution {i + 1}")
    subprocess.run(['python', search_files[i]])
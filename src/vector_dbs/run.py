import subprocess

# List of file paths to execute
#ingest_files = ['chroma_ingest.py', 'redis_ingest.py', 'mongo_ingest.py']


for i in range(10):
    print(f"Running {'mongo_ingest.py'} - Execution {i+1}")
    subprocess.run(['python', 'mongo_ingest.py'])
    print(f"Running {'search.py'} - Execution {i + 1}")
    subprocess.run(['python', 'search.py'])


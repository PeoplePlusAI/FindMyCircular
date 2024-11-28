## Overview
This project utilizes a Neo4j database within a Docker container and includes a setup script for installing Miniconda and managing Python dependencies.

## Prerequisites
- Docker installed on your system
- NVIDIA GPU and appropriate drivers (the container uses GPU support)
- Sufficient disk space for Neo4j data and Miniconda installation

## Docker Container Setup

### Container Command
```bash
docker run \
    -p 7474:7474 -p 7687:7687 \
    -v $PWD/data:/data \
    -v $PWD/plugins:/plugins \
    -v $PWD:/workspace \
    --name neo4jLangchain \
    -e NEO4J_apoc_export_file_enabled=true \
    -e NEO4J_apoc_import_file_enabled=true \
    -e NEO4J_apoc_import_file_useneo4jconfig=true \
    -itd --gpus all \
    -w /workspace \
    neo4j:5.24.1
```

### Command Breakdown
- Ports:
  - 7474: Neo4j browser interface
  - 7687: Bolt protocol for database access
- Volumes:
  - `$PWD/data:/data`: Persists Neo4j data
  - `$PWD/plugins:/plugins`: Custom Neo4j plugins
  - `$PWD:/workspace`: Mounts current directory as workspace
- Environment Variables:
  - Enables APOC file import/export functionality
- GPU Support:
  - `--gpus all`: Enables all available GPUs
- Working Directory:
  - `-w /workspace`: Sets the container's working directory

## Python Environment Setup

### Setup Script (setup.sh)
```bash
#!/bin/bash

# Update system packages
apt-get update && apt-get install -y wget curl && rm -rf /var/lib/apt/lists/*
apt update && apt upgrade -y

# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-py311_24.7.1-0-Linux-x86_64.sh
bash Miniconda3-py311_24.7.1-0-Linux-x86_64.sh -b

# Initialize conda
. ~/miniconda3/etc/profile.d/conda.sh
conda init

# Cleanup
rm Miniconda3-py311_24.7.1-0-Linux-x86_64.sh
rm ~/miniconda.sh
```

### Python Dependencies
Install required packages using:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Docker container:
   ```bash
   docker start neo4jLangchain
   ```

2. Execute the setup script:
   ```bash
   docker exec neo4jLangchain bash setup.sh
   ```

3. Install Python dependencies:
   ```bash
   docker exec neo4jLangchain pip install -r requirements.txt
   ```

## Accessing Neo4j

- Browser interface: http://localhost:7474
- Default credentials (if not changed):
  - Username: neo4j
  - Password: neo4j (you'll be prompted to change on first login)

## Best Practices

1. Data Persistence:
   - All Neo4j data is persisted in the `./data` directory
   - Back up this directory regularly

2. Plugin Management:
   - Place any Neo4j plugins in the `./plugins` directory
   - Restart the container after adding new plugins

3. Python Environment:
   - Use virtual environments for isolation
   - Keep requirements.txt updated

## Notes
- This setup uses Neo4j version 5.24.1
- Python version: 3.11 (via Miniconda)
- APOC core functionality is enabled for file operations

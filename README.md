# MARQO

## Step-by-Step Instructions

If Docker is not installed, follow the instructions for your operating system:

### 1. Install Docker

#### For Windows and macOS

1. **Download Docker Desktop**:
   - Visit the [Docker Desktop download page](https://www.docker.com/products/docker-desktop).
   - Download the installer for your operating system (Windows or macOS).

2. **Install Docker Desktop**:
   - Run the installer and follow the on-screen instructions.
   - For macOS, if you have an ARM-based Mac (like Apple M1/M2), ensure you download the version that supports ARM64.

3. **Start Docker Desktop**:
   - Launch Docker Desktop from your applications menu.
   - Follow any initial setup prompts.

#### For Linux

1. **Install Docker Engine**:
   - Follow the official [Docker Engine installation guide](https://docs.docker.com/engine/install/) for your Linux distribution.

2. **Start Docker**:
   - Start the Docker service using your system's service manager:

     ```sh
     sudo systemctl start docker
     ```

### 2. Enable QEMU Emulation (for ARM64 users)

If you are using an ARM64 architecture (e.g., Apple M1/M2), enable QEMU emulation to run AMD64 Docker images:

```sh
docker run --rm --privileged tonistiigi/binfmt --install all
```

### 3. Download MARQO Docker

1. **Download directly**:
  - Go to [Releases](https://github.com/igorafsouza/MARQO/releases)
  - Download `marqo-v1.0.0.tar.gz`

2. **Command line**:
  ```sh
  wget https://github.com/igorafsouza/MARQO/releases/download/v1.0.0/marqo-v1.0.0.tar.gz
  ```

### 4. Extract the tar.gz file (Browsers such as Safari may decompress it automatically)
  ```sh
  gunzip marqo-v1.0.0.tar.gz
  ```

### 5. Load the Docker Image
  ```sh
  docker load -i marqo-v1.0.0.tar
  ```
  If loading the Docker Image you encounter the error `read-only file system`, please refer to the [Troubleshooting Guide](https://github.com/igorafsouza/MARQO/blob/main/TROUBLESHOOTING.md) for macOS, Windows, and Linux.

### 6. Run the Docker
To run the Docker container and allow the Jupyter Notebook to access local files, you need to mount the directory containing your data into the container. Replace `/path/to/your/images` with the full path to the directory on your host machine that contains the data you want to analyze.

  ```sh
  docker run --platform linux/amd64 -p 8888:8888 -v /path/to/your/images:/mnt/data marqo-v1.0.0
  ```

Inside the Jupyter Notebook, you can access your data files in the `/mnt/data` directory. Use `/mnt/data` for both:
   - "Path to raw images"
   - "Path to output folder"
In Launch/Review Application text boxes. 

### 7. Access Jupyter Notebook
  In your web browser, navigate to `http://localhost:8888` to access the Jupyter Notebook 

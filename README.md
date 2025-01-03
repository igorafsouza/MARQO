# MARQO 

## Multiplex-imaging Analysis, Registration, Quantification, and Overlaying

### Step-by-Step Instructions

If Docker is not installed, follow the instructions for your operating system:

### 1. Install Docker

#### For Windows and macOS

1. **Download Docker Desktop**:
   - Visit the [Docker Desktop download page](https://www.docker.com/products/docker-desktop).
   - Download the installer for your operating system (Windows or macOS).

2. **Install Docker Desktop**:
   - Run the installer and follow the on-screen instructions.
   - For macOS, if you have an ARM-based Mac (like Apple M series), ensure you download the version that supports ARM64.

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

If you are using an ARM64 architecture (e.g., Apple M series), enable QEMU emulation to run AMD64 Docker images:

```sh
docker run --rm --privileged tonistiigi/binfmt --install all
```

You may need to select "QEMU (Legacy)" as your Virtual Machine Manager (VMM) in the Docker Desktop application settings, if not defaulted already.

### 3. Download MARQO Docker

1. **Download directly**:
  - Go to [Releases](https://github.com/igorafsouza/MARQO/releases)
  - Download `marqo-v1.0.0.tar.gz` (This compact version contains only StarDist2D as segmentation method - Recommended for machines up to 16 GB of RAM)

2. **Pull Image from Docker Hub**:
  - This image contains StarDist2D and Cellpose as segmentation methods
  - On Terminal/Command Prompt execute:
    
    ```sh
    docker pull igorafsouza/marqo:v1.0.0
    ``` 
   ! Attention: If you pull the image, you can skip to step **6. Run the Docker**
   
4. **Or use Command Line from Terminal/Command Prompt**:
  ```sh
  wget https://github.com/igorafsouza/MARQO/releases/download/v1.0.0/marqo-v1.0.0.tar.gz
  ```

### 4. Extract the tar.gz file (Browsers such as Safari may decompress it automatically)

1. **Run Command Line in Terminal (Recommended for Linux and macOS)**:
  ```sh
  gunzip marqo-v1.0.0.tar.gz
  ```
2. **Extract directly (Recommended for Windows)**:
   - Download [7-Zip](https://www.7-zip.org/)
   - Go to the folder where you downloaded marqo-v1.0.0.tar.gz, right-click on it
   - On 7-Zip choose "Extract Here"

### 5. Load the Docker Image
- Make sure Docker is running
- Open Terminal or Command Prompt
- Go to the directory where you extracted the docker using similar commands:
  Linux/macOS (Terminal)
  ```sh
  cd /path/to/folder/where/docker_tar_file_is
  ```
  
  Windows (Command Prompt)
  ```sh
  cd /d "C:\path\to\folder\where\docker_tar_file_is"
  ```
  
- Load the Docker file
  ```sh
  docker load -i marqo-v1.0.0.tar
  ```
  If loading the Docker Image you encounter the error `read-only file system`, please refer to the [Troubleshooting Guide](https://github.com/igorafsouza/MARQO/blob/main/TROUBLESHOOTING.md) for macOS, Windows, and Linux.

### 6. Run the Docker
To run the Docker container and allow the Jupyter Notebook to access the local files, you need to mount the directory containing your data/images. Replace `/path/to/your/data` with the full path of the directory on your local machine that contains the data/images you want to analyze.
   - Make sure Docker is running
   - From the Terminal (in Linux or macOS) run:
     
     If you pulled the image, use the fullname with the tag
     
     ```sh
     docker run --platform linux/amd64 -p 8888:8888 -v /path/to/your/data:/mnt/data igorafsouza/marqo:v1.0.0
     ```

     If you downloaded from Releases, and manually loaded the image, use

     ```sh
     docker run --platform linux/amd64 -p 8888:8888 -v /path/to/your/data:/mnt/data marqo-v1.0.0
     ```
     
   - From the Command Prompt (Windows) run:
     
     If you pulled the image, use the fullname with the tag
     ```sh
     docker run --platform linux/amd64 -p 8888:8888 -v "C:\path\to\your\data:/mnt/data" igorafsouza/marqo:v1.0.0
     ```
     
     If you downloaded from Releases, and manually loaded the image, use
     
     ```sh
     docker run --platform linux/amd64 -p 8888:8888 -v "C:\path\to\your\data:/mnt/data" marqo-v1.0.0
     ```
     
### 7. Access Jupyter Notebook
In your web browser, navigate to `http://localhost:8888` to access Jupyter Notebook. Once opened you will see the notebook `MARQO_master_application_v1.ipynb`. Once you click on it, a new web page will open. Please, locate the `Voila` button on the top menu bar. Click on `Voila` button to start the **Application**, in a new web page.

When you run the Docker container with the `-v /path/to/your/data:/mnt/data` option, it creates a mapping between a directory on your host machine and a directory inside the container. This allows the Docker container to access files and folders within the specified directory on your host machine.
   - Host Directory: `/path/to/your/data` (replace with your actual directory)
   - Container Directory: `/mnt/data`

**Using the Applications**

There are two applications you can use: ***Launch** and **Review***. 

**Launch Application**

In the ***Launch Application***, after you choose the technology, you must use the folder `/mnt/data/` as the root directory (currently set as default) and complete the path to your images.
   - "Path to raw images": Specify the path to your raw images starting from `/mnt/data`. If your images are directly in the root `/mnt/data`, you only need to pass it.
   - "Path to output folder": Specify the path where you want to create the output folder starting from `/mnt/data` (the directory must already exist). If you want to create the output folder directly in the root `/mnt/data`, you only need to pass it.

**Review Application**

In the Review Application, use `/mnt/data/` as the base path.
   - "Path to sample folder": Specify the path to your sample folder starting from `/mnt/data` until the sample output folder (e.g. `/mnt/data/sample_name`).

### 8. Demo Videos
Download the demo videos below on how to operate MARQO:
   - [Launch Application](https://github.com/igorafsouza/MARQO/releases/download/v1.0.0/Buckup-et-al-2024_sup-video-1.mov)
   - [Review Application](https://github.com/igorafsouza/MARQO/releases/download/v1.0.0/Buckup-et-al-2024_sup-video-2.mov)

### 9. Sample Images
   - MICSSS Single Sample
      - [Image Example 1](https://drive.google.com/uc?export=download&id=10F_yMlT0nzyhcGl4-_FeuKbazZTc53h0) (55MB)
      - [Image Example 2](https://drive.google.com/uc?export=download&id=16B_vNmer8nW_9G0KNkYzvx2WUqwanFFh) (49MB)
      - [Image Example 3](https://drive.google.com/uc?export=download&id=1BO1XOFqxLHwAEjvsWgvvT3tiPoSeriRh) (69MB)
    
   - MICSSS Multiple Samples
      - [Image Example 1](https://drive.google.com/uc?export=download&id=1wRsxVn2ariXU2Q_4n6_UDVuCbvplGXYZ) (247MB)
      - [Image Example 2](https://drive.google.com/uc?export=download&id=1TCaIq6xUFaw_EdeRIpAXrLhZOlbm7ig2) (309MB)
      - [Image Example 3](https://drive.google.com/uc?export=download&id=1Qhhy5hmdiGHtODozcZHAfHN56Z0am70b) (257MB)
    
   - Singleplex IHC
      - [Image Example 1](https://drive.google.com/uc?export=download&id=1FEjfMc1XVV-VGsM8hareU4OWxF8Kv5e7) (25MB)
    
   - COMET
      - [Image Example 1](https://drive.google.com/uc?export=download&id=1PsTXUakZ-SjzubqFynsMgRRP1rq12Slo) (12GB)


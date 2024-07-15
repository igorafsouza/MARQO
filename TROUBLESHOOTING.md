# Troubleshooting "Read-Only File System" Error

## For macOS
## 1. Restart Docker Desktop
1. Right-click on the Docker Desktop icon in the menu bar.
2. Select "Quit Docker Desktop".
3. Restart Docker Desktop from your Applications folder.

## 2. Ensure Docker Desktop Has Full Disk Access
1. Open `System Preferences`.
2. Go to `Security & Privacy`.
3. Select the `Privacy` tab.
4. Click the lock icon to make changes.
5. Select `Full Disk Access` from the list on the left.
6. Ensure Docker Desktop is checked in the list on the right. If it's not, add it by clicking the `+` button and selecting Docker Desktop from the Applications folder.

## 3. Load the Docker Image with Sufficient Permissions
1. Open a terminal with administrative privileges and run the commmand:
   ```sh
   docker load -i marqo-v1.0.0.tar
   ```
2. Open a terminal and load the Docker image with `sudo`:
   ```sh
   sudo docker load -i marqo-v1.0.0.tar
   ```
   
## 4. Check Disk Space
1. Click the Apple menu.
2. Select `About This Mac`.
3. Go to the `Storage` tab and ensure there is enough available space.

## 5. Increase Docker Disk Image Size
1. Open Docker Desktop.
2. Go to `Preferences`.
3. Select the `Resources` tab.
4. Increase the disk image size as needed.

## For Linux
## 1. Restart Docker Service
1. Open a terminal.
2. Run the following command to restart Docker:
   ```sh
   sudo systemctl restart docker
   ```
   
## 2. Ensure Docker Directory Permissions
1. Check and set the correct permissions for the Docker directory (**Attention!** The path `/var/lib/docker` can be slighly different in your system):
sudo chmod -R 755 /var/lib/docker
sudo chown -R root:root /var/lib/docker

## 3. Check Filesystem Status
1. Verify that the filesystem is not mounted as read-only:
   ```sh
   mount | grep /var/lib/docker
   ```
2. If necessary, remount the filesystem with read-write permissions:
   ```sh
   sudo mount -o remount,rw /var/lib/docker
   ```

## 4. Check Disk Space
1. Ensure there is sufficient disk space available:
   ```sh
   df -h /var/lib/docker
   ```

## 5. Load the Docker Image with Sufficient Permissions
1. Open a terminal.
2. Run the following command to load the Docker image with `sudo`:
   ```sh
   sudo docker load -i marqo-v1.0.0.tar
   ```

## For Windows
## 1. Restart Docker Desktop
1. Right-click on the Docker Desktop icon in the system tray.
2. Select "Quit Docker Desktop".
3. Restart Docker Desktop from your Start menu.

## 2. Run Docker Desktop as Administrator
1. Right-click on the Docker Desktop shortcut.
2. Select `Run as administrator`.

## 3. Check Disk Space
1. Open `File Explorer`.
2. Go to `This PC` and check the available space on your drive.

## 4. Increase Docker Disk Image Size
1. Open Docker Desktop.
2. Go to `Settings`.
3. Select the `Resources` tab.
4. Increase the disk image size as needed.

## 5. Load the Docker Image with Sufficient Permissions
1. Open a command prompt with administrative privileges.
2. Run the following command to load the Docker image:
   ```sh
   docker load -i marqo-v1.0.0.tar
   ```

# Frigate NVR: Shared Memory (SHM) Sizing & Configuration Guide

This document explains why Frigate NVR requires shared memory (`/dev/shm`), provides mathematical formulas to calculate the exact SHM size required for your specific camera setup, and lists configuration steps for Docker, Docker Compose, Kubernetes, and Home Assistant.

---

## 1. Why Frigate Uses Shared Memory

Frigate's multiprocessing architecture divides capture, object detection, recording, and database storage tasks into independent OS processes to bypass Python's Global Interpreter Lock (GIL).

To transfer video frame data between these processes without high latency or CPU-intensive data copies, Frigate implements a **zero-copy inter-process communication (IPC) pipeline**:
*   **FFmpeg capture processes** write raw YUV420p frame data directly into named shared memory buffers (e.g., `/dev/shm/{camera_name}_frame{index}`).
*   Only metadata pointers (frame name and timestamp) are sent over multiprocessing queues.
*   **Object detectors and encoders** read raw frames directly from the shared memory segment.
*   **Auxiliary writes**: `/dev/shm` is also used for runtime system logs (up to 40MB), Nginx API cache (10MB), and Birdseye restream frame buffers (8MB).

If the allocated `/dev/shm` space is depleted, the container will immediately exit with a **`SIGBUS` (Bus error)** or **`cannot allocate memory`** crash.

---

## 2. SHM Size Calculation Formulas

Use the formulas below to calculate your minimum required shared memory pool.

### A. Mathematical Formulas
1.  **Frame Size in Bytes (YUV420p Color Space)**:
    YUV420p uses exactly $1.5$ bytes per pixel.
    $$\text{Frame Size (Bytes)} = \text{detect.width} \times \text{detect.height} \times 1.5$$

2.  **Single Camera Requirement (Default 20-Frame Buffer & Detector Queue)**:
    Each camera maintains a rolling buffer of 20 frames, plus detector pipeline overhead (approx. $0.258$ MB):
    $$\text{Camera SHM (MB)} = \frac{(\text{detect.width} \times \text{detect.height} \times 1.5 \times 20) + 270,480}{1,048,576}$$

3.  **Total Recommended SHM Pool**:
    Includes static service overhead (58MB) and a padding buffer for 2 additional 720p streams to allow runtime camera additions:
    $$\text{Total Recommended SHM (MB)} \approx \sum_{c \in \text{cameras}} \left(\text{Camera SHM}_{c}\right) + 114 \text{ MB}$$

### B. Size Guidelines
*   **2 Cameras at 720p ($1280 \times 720$)**:
    *   *Camera SHM*: $\approx 26.6$ MB per camera.
    *   *Minimum Recommended*: $(2 \times 26.6\text{MB}) + 114\text{MB} \approx 167$ MB.
    *   **Minimum Target: 256MB**
*   **4 Cameras at 1080p ($1920 \times 1080$)**:
    *   *Camera SHM*: $\approx 59.6$ MB per camera.
    *   *Minimum Recommended*: $(4 \times 59.6\text{MB}) + 114\text{MB} \approx 352$ MB.
    *   **Minimum Target: 512MB**
*   **6 Cameras at 4K ($3840 \times 2160$)**:
    *   *Camera SHM*: $\approx 238.4$ MB per camera.
    *   *Minimum Recommended*: $(6 \times 238.4\text{MB}) + 114\text{MB} \approx 1544$ MB.
    *   **Minimum Target: 2GB**

---

## 3. Configuration Solutions per Environment

### A. Docker Compose
Specify the `shm_size` parameter inside your service definition:
```yaml
services:
  frigate:
    container_name: frigate
    image: ghcr.io/blakeblackshear/frigate:stable
    shm_size: "512mb"  # Align with calculated size
    volumes:
      - /path/to/config.yml:/config/config.yml
      - /path/to/media:/media/frigate
```

### B. Docker CLI
Pass the `--shm-size` flag directly:
```bash
docker run -d \
  --name=frigate \
  --shm-size=512m \
  -v /path/to/config.yml:/config/config.yml \
  ghcr.io/blakeblackshear/frigate:stable
```

### C. Kubernetes
Kubernetes Deployments do not have a native `shmSize` property. Instead, mount a RAM-backed `emptyDir` volume (medium `Memory`) directly to `/dev/shm`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frigate
spec:
  template:
    spec:
      containers:
      - name: frigate
        image: ghcr.io/blakeblackshear/frigate:stable
        volumeMounts:
        - name: shm
          mountPath: /dev/shm
      volumes:
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: "512Mi"  # Align with calculated size
```

### D. Home Assistant
*   **HA Add-on Store**: The Home Assistant Supervisor manages add-on container sizing dynamically. It automatically maps a `tmpfs` volume to `/dev/shm` sized at **50% of the host machine's total physical RAM**. No manual sizing is required.
*   **Virtualization / Proxmox LXC**:
    *   Ensure the VM/LXC container has sufficient allocated memory (since HA allocations scale with total RAM).
    *   If running under Proxmox LXC, verify that container **nesting** is enabled, and **ballooning** is disabled to prevent dynamic memory allocation lag from dropping shared pages.

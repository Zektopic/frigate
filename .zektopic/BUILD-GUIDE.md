# Custom Frigate Image Build Guide

Building the `frigate-ncnn-vulkan` Docker image with ncnn Vulkan support.

## Prerequisites

- Docker installed and working
- Vulkan-capable AMD GPU with Mesa RADV drivers
- Basic familiarity with Docker build

## Quick Build

```bash
cd docker/ncnn-vulkan
docker build -t frigate-ncnn-vulkan .
```

## What the Dockerfile Does

```dockerfile
FROM ghcr.io/blakeblackshear/frigate:stable

# Install ncnn with Vulkan support (Python bindings)
RUN pip install --break-system-packages ncnn

# Replace the ONNX detector with our ncnn Vulkan version
COPY onnx.py /opt/frigate/frigate/detectors/plugins/onnx.py

# Clear Python cache
RUN find /opt/frigate -name "__pycache__" -exec rm -rf {} + 2>/dev/null
```

The base image is `frigate:stable` (not `stable-rocm`). ROCm is not needed — ncnn uses Vulkan via RADV instead.

## Image Size

| Image | Size | Contents |
|---|---|---|
| `frigate:stable` | ~2GB | Frigate + ONNX Runtime CPU |
| `frigate:stable-rocm` | ~12.7GB | Frigate + full ROCm stack |
| `frigate-ncnn-vulkan` | ~2.1GB | Frigate + ncnn (~8MB) |

We save ~10GB by avoiding the ROCm image entirely.

## Deployment

After building the image, update your `docker-compose.yml`:

```yaml
services:
  frigate:
    image: frigate-ncnn-vulkan  # instead of ghcr.io/.../frigate:stable
```

Then redeploy:

```bash
docker compose down
docker compose up -d
```

## Pushing to a Registry

To push to GitHub Container Registry:

```bash
docker tag frigate-ncnn-vulkan ghcr.io/Zektopic/frigate-ncnn-vulkan:latest
docker push ghcr.io/Zektopic/frigate-ncnn-vulkan:latest
```

## Updating

When Frigate releases a new version:

```bash
# Rebuild with latest base
docker build --no-cache --pull -t frigate-ncnn-vulkan docker/ncnn-vulkan/
```

The ncnn version can be updated independently by changing the pip install line in the Dockerfile.

## CI/CD

For automated builds, add a GitHub Actions workflow:

```yaml
name: Build ncnn Vulkan Image
on:
  push:
    branches: [feature/ncnn-vulkan-detector]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          context: docker/ncnn-vulkan
          push: true
          tags: ghcr.io/Zektopic/frigate-ncnn-vulkan:latest
```

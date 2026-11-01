---
id: low_resource
title: Low-Resource Hosts (Experimental)
---

:::warning

Experimental and specific to this fork. The `low_resource` switch is off by default and nothing changes unless you enable it. Hardware marked **untested** below has not been run by the maintainers.

:::

Frigate can run on an old dual/quad-core PC, a 2–4 GB RAM box or a Raspberry Pi class board, but the defaults assume a mid-range machine. Most of the RAM goes to fixed costs that grow with the number of cameras, and most of the CPU goes to decoding video and running the detector. This page explains which settings lower each cost, and what the optional `low_resource` switch does for you.

## Where the resources go

| Cost                       | Default                                               | What lowers it                                                             |
| -------------------------- | ----------------------------------------------------- | -------------------------------------------------------------------------- |
| Decoded frames in /dev/shm | 50 frames per camera, at detect resolution            | `low_resource.shm_frames`, a lower detect resolution                       |
| Video decode (ffmpeg)      | One decoder per camera for the detect stream          | Use the camera's sub-stream for `detect`, hardware decode                  |
| Object detection           | Depends on the detector and model size                | A smaller model (320 instead of 640), an accelerator, fewer detect fps     |
| jsmpeg live encoders       | Started per camera while the jsmpeg live view is open | Started on demand and stopped when idle (always on in this fork)           |
| Birdseye                   | On                                                    | `birdseye.enabled: false`                                                  |
| Enrichments                | Off unless enabled                                    | Leave semantic search, face recognition, LPR and GenAI off on small hosts  |

### Shared memory budget

Every camera keeps a ring of decoded YUV420 frames in `/dev/shm`. The memory this takes is:

```
shm per camera = frames x detect_width x detect_height x 1.5 bytes
```

| Detect resolution | 50 frames (default) | 20 frames (low-resource) |
| ----------------- | ------------------- | ------------------------ |
| 1920x1080         | 148 MiB             | 59 MiB                   |
| 1280x720          | 66 MiB              | 26 MiB                   |
| 640x360           | 16.5 MiB            | 6.6 MiB                  |

For example, 8 cameras detecting at 720p use 527 MiB with the default 50 frames and 211 MiB with 20 frames, which saves 316 MiB. 4 cameras detecting on a 640x360 sub-stream with 20 frames need only 26 MiB. This memory counts against the container's RAM. Remember to also size `shm_size` for it (see [Calculating required shm-size](../frigate/installation.md#calculating-required-shm-size)).

The ring only has to buffer the frames that are waiting between processes. Cameras added at runtime already run with 10 frames, so any value from 10 up works. 20 frames gives about 4 seconds of slack at 5 detect fps.

## The `low_resource` switch

```yaml
low_resource:
  enabled: true
  # optional: decoded frames per camera in /dev/shm (10-500)
  # shm_frames: 20
```

With `enabled: true`, these defaults are filled in for any key that is **not** already set in your config. Values you set yourself always win, at the global level and per camera:

| Setting                       | Normal default | Low-resource default |
| ----------------------------- | -------------- | -------------------- |
| shm frames per camera         | 50             | 20                   |
| `birdseye.enabled`            | `true`         | `false`              |
| `live.height` (jsmpeg)        | 720            | 360                  |

Detection settings such as fps, resolution, zones and filters are never changed automatically, because they affect what Frigate detects.

`shm_frames` can also be set on its own without enabling low-resource mode. It takes precedence over the `SHM_MAX_FRAMES` environment variable. With neither set, the default is 50, or 20 in low-resource mode. If `/dev/shm` is too small for the requested count, Frigate still caps it and logs a warning.

## Recommended configuration for small hosts

```yaml
low_resource:
  enabled: true

detect:
  fps: 5 # 3-5 is enough for people and cars

ffmpeg:
  output_args:
    # copy video and audio; the default re-encodes audio to AAC
    record: preset-record-generic-audio-copy

record:
  enabled: true

motion:
  # optional: skip contrast improvement on very slow CPUs
  # improve_contrast: false

cameras:
  front:
    ffmpeg:
      inputs:
        # record the main stream (stream copy, almost free)
        - path: rtsp://user:pass@10.0.0.10:554/main
          roles:
            - record
        # detect on the sub-stream (decode is the expensive part)
        - path: rtsp://user:pass@10.0.0.10:554/sub
          roles:
            - detect
    detect:
      width: 640
      height: 360
```

A full example lives in `config/potato.yml.example`.

### Measured savings

These numbers were measured in the `frigate-ncnn-vulkan` image on an x86 host (Ryzen 7 class, `--cpus 4`) with synthetic input. An old dual-core CPU or a Raspberry Pi will be several times slower in absolute terms, but the ratios should hold.

| Change                                                         | Before                    | After                    |
| -------------------------------------------------------------- | ------------------------- | ------------------------ |
| Detect on a 640x360 sub-stream instead of the 1080p main stream | 7.4% of a core per camera | 1.0% of a core per camera |
| ncnn YOLO26n on CPU (no GPU), 640 to 320 input                 | 70 ms, 121 MB RSS         | 16 ms, 50 MB RSS         |
| jsmpeg encoders when nobody is watching (8 cameras)            | 8 ffmpeg processes, 185 MB RSS (60 MB PSS) | none                     |
| jsmpeg encode at 360p instead of 720p (while viewing)          | 2.3% of a core            | 1.3% of a core           |
| shm frames, 8 cameras at 720p                                  | 527 MiB                   | 211 MiB                  |

Motion detection is already cheap (Rust-accelerated, about 0.7 ms per frame whatever the detect resolution), so lowering the resolution saves RAM and decode time rather than motion CPU.

## Choosing a detector on weak hardware

| Hardware                                             | Suggested detector                                                                   | Status in this fork               |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------ | --------------------------------- |
| x86 without a GPU                                    | ncnn on CPU (this image) with a 320 model, or OpenVINO on CPU                        | ncnn CPU path tested              |
| Intel CPU with an iGPU (6th gen or newer)            | [OpenVINO](object_detectors.md#openvino-detector) `device: GPU` (upstream image)     | upstream, not tested here         |
| AMD/Intel GPU with Vulkan                            | ncnn with Vulkan (this image)                                                        | tested on Radeon 760M             |
| Rockchip RK3566/RK3588 boards                        | [RKNN](object_detectors.md#rockchip-platform) (upstream `-rk` image)                 | upstream, **untested** here       |
| Raspberry Pi 5 + Hailo-8L                            | [Hailo](object_detectors.md#hailo-8) (upstream image)                                | upstream, **untested** here       |
| Any host + Google Coral (USB/PCIe)                   | [EdgeTPU](object_detectors.md#edge-tpu-detector)                                     | upstream, **untested** here       |
| Raspberry Pi 4/5 without an accelerator              | ncnn on CPU with a 320 model                                                         | **untested** (no arm64 image yet) |

### ncnn on CPU

Without a usable GPU the ncnn plugin now runs on ncnn's native CPU kernels. A host without a GPU, or a container without `/dev/dri`, still shows Mesa's software Vulkan device (llvmpipe). The plugin ignores it, because ncnn on llvmpipe is slow and crashed the Rust detector worker.

- **Model size**: the plugin uses `model.width`/`model.height` as the network input size, which must be square and a multiple of 32. YOLO26 ncnn exports have their input size baked in, so export one at the size you want, for example `yolo export model=yolo26n.pt format=ncnn imgsz=320`, and set `model.width: 320` and `model.height: 320`.
- **Threads**: set the `NCNN_NUM_THREADS` environment variable on the container to change the ncnn CPU thread count (default 2 for the Rust worker). On a dual-core host, `1` leaves a core for ffmpeg. Measured with YOLO26n at 640 on 4 CPUs: 97 ms with 1 thread, 80 ms with 2 and 66 ms with 4. At 320, 1 thread was the fastest (23 ms).

## What is not changed

- The embeddings process (about 40 MB) always starts because other components talk to it. Leaving enrichments disabled keeps it idle.
- Recording always copies the video stream. The default record preset re-encodes audio to AAC, so use `preset-record-generic-audio-copy` (or `preset-record-generic` for no audio) on small hosts.
- Hardware video decoding (`ffmpeg.hwaccel_args`) is still auto-detected (VAAPI on Intel/AMD), which is the biggest single CPU saver when available.

variable "ROCM" {
  default = "6.4.3"
}
variable "HSA_OVERRIDE_GFX_VERSION" {
  default = ""
  description = "For Phoenix1 (gfx1103) leave empty — ROCm 6.x supports RDNA3 natively."
}
variable "HSA_OVERRIDE" {
  default = "1"
}

target wget {
  dockerfile = "docker/main/Dockerfile"
  platforms = ["linux/amd64"]
  target = "wget"
}

target deps {
  dockerfile = "docker/main/Dockerfile"
  platforms = ["linux/amd64"]
  target = "deps"
}

target rootfs {
  dockerfile = "docker/main/Dockerfile"
  platforms = ["linux/amd64"]
  target = "rootfs"
}

target xdnpu {
  dockerfile = "docker/xdnpu/Dockerfile"
  contexts = {
    deps = "target:deps",
    wget = "target:wget",
    rootfs = "target:rootfs"
  }
  platforms = ["linux/amd64"]
  args = {
    ROCM = ROCM,
    HSA_OVERRIDE_GFX_VERSION = HSA_OVERRIDE_GFX_VERSION,
    HSA_OVERRIDE = HSA_OVERRIDE
  }
}

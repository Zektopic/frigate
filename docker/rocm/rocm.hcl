variable "ROCM" {
  default = "6.4.3"
}
variable "HSA_OVERRIDE_GFX_VERSION" {
  default = ""
  description = "Set to '10.3.0' for Vega/Picasso APUs (gfx900/gfx909) which were dropped from official ROCm 7.x. For Phoenix1 APUs (gfx1103/Radeon 760M) leave empty — ROCm 6.x supports RDNA3 natively."
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

target rocm {
  dockerfile = "docker/rocm/Dockerfile"
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

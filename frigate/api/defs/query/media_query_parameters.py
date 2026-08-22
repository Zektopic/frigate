from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Extension(str, Enum):
    webp = "webp"
    png = "png"
    jpg = "jpg"
    jpeg = "jpeg"

    def get_mime_type(self) -> str:
        if self in (Extension.jpg, Extension.jpeg):
            return "image/jpeg"
        return f"image/{self.value}"


class MediaLatestFrameQueryParams(BaseModel):
    bbox: Optional[int] = None
    timestamp: Optional[int] = None
    zones: Optional[int] = None
    mask: Optional[int] = None
    motion: Optional[int] = None
    paths: Optional[int] = None
    regions: Optional[int] = None
    quality: Optional[int] = 70
    height: Optional[int] = None
    store: Optional[int] = None


class MediaEventsSnapshotQueryParams(BaseModel):
    download: Optional[bool] = False
    timestamp: Optional[int] = None
    bbox: Optional[int] = None
    crop: Optional[int] = None
    height: Optional[int] = None
    quality: Optional[int] = None


class MediaMjpegFeedQueryParams(BaseModel):
    # Lower bounds only: both values are divisors in the MJPEG stream
    # loop, so 0 raised ZeroDivisionError mid-response. No upper bound,
    # so existing callers requesting a high frame rate are unaffected.
    fps: int = Field(default=3, ge=1)
    height: int = Field(default=360, ge=1)
    bbox: Optional[int] = None
    timestamp: Optional[int] = None
    zones: Optional[int] = None
    mask: Optional[int] = None
    motion: Optional[int] = None
    regions: Optional[int] = None

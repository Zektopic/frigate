import { describe, it, expect } from "vitest";
import { calculateInpointOffset } from "../videoUtil";
import { Recording } from "@/types/record";

describe("calculateInpointOffset", () => {
  it("should return 0 when timeRangeStart is undefined", () => {
    const recording = {
      id: "1",
      camera: "test_camera",
      start_time: 100,
      end_time: 200,
      path: "/test/path.mp4",
      segment_size: 1024,
      duration: 100,
      motion: 0,
      objects: 0,
      dBFS: -40,
    } as Recording;

    expect(calculateInpointOffset(undefined, recording)).toBe(0);
  });

  it("should return 0 when firstRecordingSegment is undefined", () => {
    expect(calculateInpointOffset(150, undefined)).toBe(0);
  });

  it("should return 0 when the recording segment is entirely before timeRangeStart", () => {
    const recording = {
      id: "1",
      camera: "test_camera",
      start_time: 100,
      end_time: 200,
      path: "/test/path.mp4",
      segment_size: 1024,
      duration: 100,
      motion: 0,
      objects: 0,
      dBFS: -40,
    } as Recording;

    expect(calculateInpointOffset(250, recording)).toBe(0);
  });

  it("should return 0 when the recording segment is entirely after timeRangeStart", () => {
    const recording = {
      id: "1",
      camera: "test_camera",
      start_time: 100,
      end_time: 200,
      path: "/test/path.mp4",
      segment_size: 1024,
      duration: 100,
      motion: 0,
      objects: 0,
      dBFS: -40,
    } as Recording;

    expect(calculateInpointOffset(50, recording)).toBe(0);
  });

  it("should return the correct offset when the recording segment crosses timeRangeStart", () => {
    const recording = {
      id: "1",
      camera: "test_camera",
      start_time: 100, // starts before timeRangeStart
      end_time: 200,   // ends after timeRangeStart
      path: "/test/path.mp4",
      segment_size: 1024,
      duration: 100,
      motion: 0,
      objects: 0,
      dBFS: -40,
    } as Recording;

    // The function expects timeRangeStart - firstRecordingSegment.start_time
    // 150 - 100 = 50
    expect(calculateInpointOffset(150, recording)).toBe(50);
  });
});

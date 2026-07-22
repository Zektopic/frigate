import { describe, it, expect } from "vitest";
import { findChunkIndex } from "../timelineUtil";
import { TimeRange } from "@/types/timeline";

describe("findChunkIndex", () => {
  it("should return -1 for empty chunks", () => {
    expect(findChunkIndex([], 100)).toBe(-1);
  });

  const chunks: TimeRange[] = [
    { after: 100, before: 200 },
    { after: 200, before: 300 },
    { after: 350, before: 400 },
  ];

  it("should return correct index when timestamp falls inside a normal chunk", () => {
    expect(findChunkIndex(chunks, 150)).toBe(0);
    expect(findChunkIndex(chunks, 250)).toBe(1);
  });

  it("should return correct index when timestamp is exactly at 'after' of a chunk", () => {
    expect(findChunkIndex(chunks, 100)).toBe(0);
    expect(findChunkIndex(chunks, 200)).toBe(1);
    expect(findChunkIndex(chunks, 350)).toBe(2);
  });

  it("should not match when timestamp is exactly at 'before' of a non-last chunk", () => {
    // 300 is exactly 'before' of chunk 1. It shouldn't match chunk 1 because it's half-open [after, before).
    // It also shouldn't match chunk 2 because chunk 2 starts at 350.
    expect(findChunkIndex(chunks, 300)).toBe(-1);
  });

  it("should match when timestamp is exactly at 'before' of the last chunk", () => {
    // 400 is exactly 'before' of chunk 2 (the last chunk). It should match because it's closed [after, before].
    expect(findChunkIndex(chunks, 400)).toBe(2);
  });

  it("should return -1 when timestamp is outside all chunks", () => {
    expect(findChunkIndex(chunks, 50)).toBe(-1); // before first
    expect(findChunkIndex(chunks, 450)).toBe(-1); // after last
    expect(findChunkIndex(chunks, 320)).toBe(-1); // in a gap
  });
});

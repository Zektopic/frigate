import { describe, it, expect } from "vitest";
import { masksAreIdentical } from "./canvasUtil";

describe("masksAreIdentical", () => {
  it("should return true for identical string arrays", () => {
    const arr1 = ["a", "b", "c"];
    const arr2 = ["a", "b", "c"];
    expect(masksAreIdentical(arr1, arr2)).toBe(true);
  });

  it("should return false for arrays of different lengths", () => {
    const arr1 = ["a", "b", "c"];
    const arr2 = ["a", "b"];
    expect(masksAreIdentical(arr1, arr2)).toBe(false);
  });

  it("should return false for arrays with the same length but different values", () => {
    const arr1 = ["a", "b", "c"];
    const arr2 = ["a", "x", "c"];
    expect(masksAreIdentical(arr1, arr2)).toBe(false);
  });

  it("should return true for empty arrays", () => {
    const arr1: string[] = [];
    const arr2: string[] = [];
    expect(masksAreIdentical(arr1, arr2)).toBe(true);
  });

  it("should return false for arrays where order differs", () => {
    const arr1 = ["a", "b", "c"];
    const arr2 = ["a", "c", "b"];
    expect(masksAreIdentical(arr1, arr2)).toBe(false);
  });
});

import { describe, it, expect } from "vitest";
import { isSpecialCaseSection } from "../section-special-cases";

describe("isSpecialCaseSection", () => {
  it("returns true for special case sections at the global level", () => {
    expect(isSpecialCaseSection("motion", "global")).toBe(true);
    expect(isSpecialCaseSection("detectors", "global")).toBe(true);
    expect(isSpecialCaseSection("genai", "global")).toBe(true);
  });

  it("returns false for special case sections at non-global levels", () => {
    expect(isSpecialCaseSection("motion", "camera")).toBe(false);
    expect(isSpecialCaseSection("detectors", "camera")).toBe(false);
    expect(isSpecialCaseSection("genai", "camera")).toBe(false);
  });

  it("returns false for non-special case sections at any level", () => {
    expect(isSpecialCaseSection("objects", "global")).toBe(false);
    expect(isSpecialCaseSection("ffmpeg", "camera")).toBe(false);
    expect(isSpecialCaseSection("record", "global")).toBe(false);
  });
});

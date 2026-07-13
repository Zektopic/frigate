import { describe, it, expect } from "vitest";
import { isValidIconName } from "../iconUtil";

describe("iconUtil", () => {
  describe("isValidIconName", () => {
    it("should return true for valid Lu icons", () => {
      expect(isValidIconName("LuBox")).toBe(true);
      expect(isValidIconName("LuLassoSelect")).toBe(true);
      expect(isValidIconName("LuScanBarcode")).toBe(true);
      expect(isValidIconName("LuActivity")).toBe(true); // Activity is a standard Lucide icon
    });

    it("should return false for invalid icon names", () => {
      expect(isValidIconName("InvalidIconName")).toBe(false);
      expect(isValidIconName("")).toBe(false);
      expect(isValidIconName("FaBox")).toBe(false); // Fa is FontAwesome, not Lu
      expect(isValidIconName("luBox")).toBe(false); // case sensitivity check
    });
  });
});

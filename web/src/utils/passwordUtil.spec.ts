import { describe, it, expect } from "vitest";
import { calculatePasswordStrength } from "./passwordUtil";

describe("calculatePasswordStrength", () => {
  it("should return 0 for an empty string", () => {
    expect(calculatePasswordStrength("")).toBe(0);
  });

  it("should return 0 for a weak password", () => {
    expect(calculatePasswordStrength("abc")).toBe(0);
  });

  it("should add 1 point for length >= 8", () => {
    expect(calculatePasswordStrength("abcdefgh")).toBe(1);
  });

  it("should add 1 point for containing a digit", () => {
    expect(calculatePasswordStrength("abcdefg1")).toBe(2); // length >= 8 (1) + digit (1) = 2
    expect(calculatePasswordStrength("abc1")).toBe(1); // digit (1) = 1
  });

  it("should add 1 point for containing a special character", () => {
    expect(calculatePasswordStrength("abcdefg!")).toBe(2); // length >= 8 (1) + special char (1) = 2
    expect(calculatePasswordStrength("abc!")).toBe(1); // special char (1) = 1
  });

  it("should add 1 point for containing an uppercase letter", () => {
    expect(calculatePasswordStrength("Abcdefgh")).toBe(2); // length >= 8 (1) + uppercase (1) = 2
    expect(calculatePasswordStrength("Abcd")).toBe(1); // uppercase (1) = 1
  });

  it("should calculate correct strength for various combinations", () => {
    // length (1) + digit (1) + special char (1)
    expect(calculatePasswordStrength("abcdef1!")).toBe(3);

    // length (1) + uppercase (1) + digit (1)
    expect(calculatePasswordStrength("Abcdefg1")).toBe(3);

    // length (1) + uppercase (1) + special char (1)
    expect(calculatePasswordStrength("Abcdefg!")).toBe(3);

    // length (1) + digit (1) + special char (1) + uppercase (1) = 4
    expect(calculatePasswordStrength("Abcdef1!")).toBe(4);
  });
});

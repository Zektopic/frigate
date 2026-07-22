import { describe, it, expect } from "vitest";
import { stripRedactedCredentials } from "../configUtil";
import { REDACTED_CREDENTIAL_SENTINEL } from "@/lib/const";

describe("stripRedactedCredentials", () => {
  it("should return primitive values unchanged", () => {
    expect(stripRedactedCredentials(null)).toBeNull();
    expect(stripRedactedCredentials(undefined)).toBeUndefined();
    expect(stripRedactedCredentials("test")).toBe("test");
    expect(stripRedactedCredentials(123)).toBe(123);
    expect(stripRedactedCredentials(true)).toBe(true);
  });

  it("should not modify objects without redacted credentials", () => {
    const obj = { a: 1, b: "test", c: true };
    const result = stripRedactedCredentials(obj);
    expect(result).toEqual({ a: 1, b: "test", c: true });
    expect(result).toBe(obj); // Check mutation
  });

  it("should remove properties with redacted credentials in flat objects", () => {
    const obj = {
      user: "admin",
      password: REDACTED_CREDENTIAL_SENTINEL,
      token: "valid-token",
    };
    const result = stripRedactedCredentials(obj);
    expect(result).toEqual({ user: "admin", token: "valid-token" });
    expect(result).toBe(obj);
  });

  it("should recursively remove properties with redacted credentials in nested objects", () => {
    const obj = {
      level1: {
        normal: "value",
        secret: REDACTED_CREDENTIAL_SENTINEL,
        level2: {
          anotherSecret: REDACTED_CREDENTIAL_SENTINEL,
          keep: "this",
        },
      },
    };
    const result = stripRedactedCredentials(obj);
    expect(result).toEqual({
      level1: {
        normal: "value",
        level2: {
          keep: "this",
        },
      },
    });
    expect(result).toBe(obj);
  });

  it("should process arrays and remove redacted credentials within objects inside arrays", () => {
    const arr = [
      "string",
      { id: 1, pass: REDACTED_CREDENTIAL_SENTINEL },
      [{ nestedPass: REDACTED_CREDENTIAL_SENTINEL, keep: true }],
    ];
    const result = stripRedactedCredentials(arr);
    expect(result).toEqual([
      "string",
      { id: 1 },
      [{ keep: true }],
    ]);
    expect(result).toBe(arr);
  });

  it("should handle null or undefined properties in objects", () => {
    const obj = {
      a: null,
      b: undefined,
      c: {
        d: null,
        e: REDACTED_CREDENTIAL_SENTINEL
      }
    };
    const result = stripRedactedCredentials(obj);
    expect(result).toEqual({
      a: null,
      b: undefined,
      c: {
        d: null,
      }
    });
  });
});

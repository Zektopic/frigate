import { describe, it, expect } from "vitest";
import { getUTCOffset, epochToLong } from "../dateUtil";

describe("epochToLong", () => {
  it("should convert positive epoch time to long correctly", () => {
    expect(epochToLong(1700000000000)).toBe(1700000000);
    expect(epochToLong(1000)).toBe(1);
    expect(epochToLong(1500)).toBe(1.5);
  });

  it("should convert zero correctly", () => {
    expect(epochToLong(0)).toBe(0);
  });

  it("should convert negative epoch time correctly", () => {
    expect(epochToLong(-1000)).toBe(-1);
    expect(epochToLong(-1700000000000)).toBe(-1700000000);
  });
});

describe("getUTCOffset", () => {
  it("should calculate correct offset from a UTC string", () => {
    const date = new Date("2023-01-01T12:00:00Z");

    expect(getUTCOffset(date, "UTC+00:00")).toBe(0);
    expect(getUTCOffset(date, "UTC+02:30")).toBe(150);
    expect(getUTCOffset(date, "UTC-05:00")).toBe(-300);
    expect(getUTCOffset(date, "UTC-11:45")).toBe(-705);
  });

  it("should handle edge cases with UTC strings", () => {
    const date = new Date("2023-01-01T12:00:00Z");

    expect(getUTCOffset(date, "UTC+00:01")).toBe(1);
    expect(getUTCOffset(date, "UTC-00:01")).toBe(-1);
    expect(getUTCOffset(date, "UTC+14:00")).toBe(840);
  });

  it("should calculate correct offset from a timezone name (standard time)", () => {
    const date = new Date("2023-01-15T12:00:00Z"); // Jan = Standard time for most

    // Using explicit strings allows the Intl.DateTimeFormat to calculate properly based on Date object
    expect(getUTCOffset(date, "America/New_York")).toBe(-300); // EST is UTC-5
    expect(getUTCOffset(date, "Europe/London")).toBe(0); // GMT is UTC+0
    expect(getUTCOffset(date, "Asia/Tokyo")).toBe(540); // JST is UTC+9
    expect(getUTCOffset(date, "Australia/Sydney")).toBe(660); // AEDT is UTC+11
  });

  it("should calculate correct offset from a timezone name (daylight time)", () => {
    const date = new Date("2023-07-15T12:00:00Z"); // Jul = Daylight time for northern hemisphere

    expect(getUTCOffset(date, "America/New_York")).toBe(-240); // EDT is UTC-4
    expect(getUTCOffset(date, "Europe/London")).toBe(60); // BST is UTC+1
    expect(getUTCOffset(date, "Australia/Sydney")).toBe(600); // AEST is UTC+10 (Winter in Australia)
  });

  it("should correctly handle timezones with partial hour offsets", () => {
    const date = new Date("2023-01-15T12:00:00Z");

    expect(getUTCOffset(date, "Asia/Kolkata")).toBe(330); // IST is UTC+5:30
    expect(getUTCOffset(date, "Asia/Kathmandu")).toBe(345); // NPT is UTC+5:45
  });

  it("should fail gracefully or throw with invalid timezone strings", () => {
    const date = new Date("2023-01-01T12:00:00Z");

    // Intl.DateTimeFormat throws a RangeError for invalid time zone
    expect(() => getUTCOffset(date, "Invalid/Zone")).toThrow();
  });
});

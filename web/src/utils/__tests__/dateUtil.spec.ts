import { describe, it, expect } from "vitest";
import { dateToLong, epochToLong, getUTCOffset, longToDate } from "../dateUtil";

describe("longToDate", () => {
  it("should correctly convert a UNIX timestamp (seconds) to a Date object", () => {
    // 1672574400 is "2023-01-01T12:00:00.000Z"
    const timestamp = 1672574400;
    const date = longToDate(timestamp);

    expect(date).toBeInstanceOf(Date);
    expect(date.getTime()).toBe(timestamp * 1000);
    expect(date.toISOString()).toBe("2023-01-01T12:00:00.000Z");
  });

  it("should handle 0 timestamp", () => {
    const timestamp = 0;
    const date = longToDate(timestamp);
    expect(date.toISOString()).toBe("1970-01-01T00:00:00.000Z");
  });

  it("should handle negative timestamp (dates before 1970)", () => {
    // -1 day before epoch
    const timestamp = -86400;
    const date = longToDate(timestamp);
    expect(date.toISOString()).toBe("1969-12-31T00:00:00.000Z");
  });
});

describe("epochToLong", () => {
  it("should correctly convert milliseconds to seconds", () => {
    expect(epochToLong(1672574400000)).toBe(1672574400);
    expect(epochToLong(0)).toBe(0);
    expect(epochToLong(1000)).toBe(1);
    expect(epochToLong(-1000)).toBe(-1);
  });

  it("should convert positive epoch time to long correctly", () => {
    expect(epochToLong(1700000000000)).toBe(1700000000);
    expect(epochToLong(1500)).toBe(1.5);
  });

  it("should convert negative epoch time correctly", () => {
    expect(epochToLong(-1700000000000)).toBe(-1700000000);
  });
});

describe("dateToLong", () => {
  it("should correctly convert a Date object to a UNIX timestamp in seconds", () => {
    const date = new Date("2023-01-01T12:00:00.000Z");
    const timestamp = dateToLong(date);

    expect(timestamp).toBe(1672574400);
  });

  it("should handle the epoch Date correctly", () => {
    const date = new Date("1970-01-01T00:00:00.000Z");
    expect(dateToLong(date)).toBe(0);
  });

  it("should handle pre-epoch Dates correctly", () => {
    const date = new Date("1969-12-31T00:00:00.000Z");
    expect(dateToLong(date)).toBe(-86400);
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

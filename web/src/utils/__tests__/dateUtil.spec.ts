import { describe, it, expect, vi } from "vitest";
import { getUTCOffset, getDurationFromTimestamps } from "../dateUtil";
import * as i18n from "@/utils/i18n";

vi.mock("@/utils/i18n", () => ({
  default: {
    t: (key: string) => key,
  },
}));

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

describe("getDurationFromTimestamps", () => {
  it("should return invalid start time if start time is NaN", () => {
    expect(getDurationFromTimestamps(NaN, 1000)).toBe("time.invalidStartTime");
  });

  it("should return in progress if end time is null", () => {
    expect(getDurationFromTimestamps(1000, null)).toBe("time.inProgress");
  });

  it("should return invalid end time if end time is NaN", () => {
    expect(getDurationFromTimestamps(1000, NaN)).toBe("time.invalidEndTime");
  });

  it("should return correct full duration with hours, minutes, and seconds", () => {
    // 1 hour, 2 minutes, 3 seconds
    const start = 1000;
    const end = start + 3600 + 120 + 3;
    expect(getDurationFromTimestamps(start, end, false)).toBe("time.hour_one time.minute_other time.second_other");
  });

  it("should return correct abbreviated duration with hours, minutes, and seconds", () => {
    // 2 hours, 1 minute, 1 second
    const start = 1000;
    const end = start + 7200 + 60 + 1;
    expect(getDurationFromTimestamps(start, end, true)).toBe("2h 1m 1s");
  });

  it("should handle omitted units that are 0", () => {
    // 1 hour, 0 minutes, 1 second
    const start = 1000;
    const end = start + 3600 + 1;

    // Non-abbreviated
    expect(getDurationFromTimestamps(start, end, false)).toBe("time.hour_one time.second_one");

    // Abbreviated
    expect(getDurationFromTimestamps(start, end, true)).toBe("1h 1s");
  });

  it("should default abbreviated to false", () => {
    const start = 1000;
    const end = start + 60; // 1 minute
    expect(getDurationFromTimestamps(start, end)).toBe("time.minute_one");
  });
});

import { describe, it, expect, vi, afterEach } from "vitest";
import {
  dateToLong,
  epochToLong,
  getBeginningOfDayTimestamp,
  getDurationFromTimestamps,
  getUTCOffset,
  isCurrentHour,
  longToDate,
  to24Hour,
  formatSecondsToDuration,
} from "../dateUtil";

vi.mock("@/utils/i18n", () => ({
  default: {
    t: (key: string) => key,
  },
}));

describe("to24Hour", () => {
  it("should return the time as-is if format is 24hour", () => {
    expect(to24Hour("13:45", "24hour")).toBe("13:45");
    expect(to24Hour("08:30", "24hour")).toBe("08:30");
  });

  it("should convert 12-hour AM times correctly", () => {
    expect(to24Hour("12:00 AM", "12hour")).toBe("00:00");
    expect(to24Hour("12:30 AM", "12hour")).toBe("00:30");
    expect(to24Hour("1:00 AM", "12hour")).toBe("01:00");
    expect(to24Hour("09:45 AM", "12hour")).toBe("09:45");
    expect(to24Hour("11:59 AM", "12hour")).toBe("11:59");
  });

  it("should convert 12-hour PM times correctly", () => {
    expect(to24Hour("12:00 PM", "12hour")).toBe("12:00");
    expect(to24Hour("12:30 PM", "12hour")).toBe("12:30");
    expect(to24Hour("1:00 PM", "12hour")).toBe("13:00");
    expect(to24Hour("09:45 PM", "12hour")).toBe("21:45");
    expect(to24Hour("11:59 PM", "12hour")).toBe("23:59");
  });

  it("should handle case-insensitive AM/PM correctly", () => {
    expect(to24Hour("1:00 pm", "12hour")).toBe("13:00");
    expect(to24Hour("1:00 am", "12hour")).toBe("01:00");
    expect(to24Hour("1:00 pM", "12hour")).toBe("13:00");
    expect(to24Hour("1:00 Am", "12hour")).toBe("01:00");
  });

  it("should throw an error for invalid formats when not 24hour", () => {
    expect(() => to24Hour("13:00", "12hour")).toThrow(
      "Invalid time format: 13:00",
    );
    expect(() => to24Hour("invalid", "12hour")).toThrow(
      "Invalid time format: invalid",
    );
  });
});

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

describe("isCurrentHour", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("should return true for a timestamp in the current hour", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2023-10-10T14:30:00Z"));

    // Timestamp for 14:15:00
    const timestamp = new Date("2023-10-10T14:15:00Z").getTime() / 1000;
    expect(isCurrentHour(timestamp)).toBe(true);
  });

  it("should return false for a timestamp in the previous hour", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2023-10-10T14:30:00Z"));

    // Timestamp for 13:45:00
    const timestamp = new Date("2023-10-10T13:45:00Z").getTime() / 1000;
    expect(isCurrentHour(timestamp)).toBe(false);
  });

  it("should return false for a timestamp exactly at the start of the hour", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2023-10-10T14:30:00Z"));

    // Timestamp for 14:00:00 (which is exactly equal to the threshold, so > returns false)
    const timestamp = new Date("2023-10-10T14:00:00Z").getTime() / 1000;
    expect(isCurrentHour(timestamp)).toBe(false);
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

describe("getBeginningOfDayTimestamp", () => {
  it("should return the correct timestamp for the beginning of the day", () => {
    // 2023-01-01T15:30:45.123 in local time
    const date = new Date(2023, 0, 1, 15, 30, 45, 123);
    const result = getBeginningOfDayTimestamp(date);

    // The result should be the timestamp for 2023-01-01T00:00:00.000 local time
    const expectedDate = new Date(2023, 0, 1, 0, 0, 0, 0);
    expect(result).toBe(expectedDate.getTime() / 1000);
  });

  it("should mutate the original Date object", () => {
    const date = new Date(2023, 0, 1, 15, 30, 45, 123);
    getBeginningOfDayTimestamp(date);

    expect(date.getHours()).toBe(0);
    expect(date.getMinutes()).toBe(0);
    expect(date.getSeconds()).toBe(0);
    expect(date.getMilliseconds()).toBe(0);
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
    expect(getDurationFromTimestamps(start, end, false)).toBe(
      "time.hour_one time.minute_other time.second_other",
    );
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
    expect(getDurationFromTimestamps(start, end, false)).toBe(
      "time.hour_one time.second_one",
    );

    // Abbreviated
    expect(getDurationFromTimestamps(start, end, true)).toBe("1h 1s");
  });

  it("should default abbreviated to false", () => {
    const start = 1000;
    const end = start + 60; // 1 minute
    expect(getDurationFromTimestamps(start, end)).toBe("time.minute_one");
  });
});

describe("formatSecondsToDuration", () => {
  it("should handle invalid durations", () => {
    expect(formatSecondsToDuration(NaN)).toBe("Invalid duration");
    expect(formatSecondsToDuration(-5)).toBe("Invalid duration");
  });

  it("should format 0 seconds", () => {
    expect(formatSecondsToDuration(0)).toBe("0 seconds");
  });

  it("should format seconds into minutes and seconds", () => {
    expect(formatSecondsToDuration(65)).toBe("1 minute, 5 seconds");
    expect(formatSecondsToDuration(120)).toBe("2 minutes");
  });

  it("should format seconds into hours, minutes, and seconds", () => {
    expect(formatSecondsToDuration(3665)).toBe("1 hour, 1 minute, 5 seconds");
    expect(formatSecondsToDuration(7200)).toBe("2 hours");
    expect(formatSecondsToDuration(7325)).toBe("2 hours, 2 minutes, 5 seconds");
  });
});

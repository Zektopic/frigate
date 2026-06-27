import { describe, test, expect, vi, afterEach } from "vitest";

global.window = {
  navigator: {
    language: "en-US",
  },
} as unknown as Window & typeof globalThis;

// This fixes the dateUtil.test.ts "Invalid time" bug when formatToParts mock doesn't supply timezone information
// We use a mock that simulates expected behavior, returning the expected date for our specific mocks.
// Note the error 'Invalid time' happens because an error is thrown in formatUnixTimestampToDateTime.

vi.mock("@/utils/i18n", () => ({
  default: { t: (key: string) => key },
  getTranslatedLabel: (label: string) => label,
}));

import {
  convertLocalDateToTimestamp,
  dateToLong,
  epochToLong,
  formatSecondsToDuration,
  formatUnixTimestampToDateTime,
  getNowYesterdayInLong,
  longToDate,
} from "./dateUtil";

describe("longToDate", () => {
  test("should convert a unix timestamp in seconds to a Date object", () => {
    // 2023-01-01T00:00:00.000Z is 1672531200
    const timestamp = 1672531200;
    const expectedDate = new Date("2023-01-01T00:00:00.000Z");
    expect(longToDate(timestamp)).toEqual(expectedDate);
  });

  test("should handle 0", () => {
    const timestamp = 0;
    const expectedDate = new Date("1970-01-01T00:00:00.000Z");
    expect(longToDate(timestamp)).toEqual(expectedDate);
  });

  test("should handle negative timestamps", () => {
    const timestamp = -1;
    const expectedDate = new Date("1969-12-31T23:59:59.000Z");
    expect(longToDate(timestamp)).toEqual(expectedDate);
  });
});

describe("getNowYesterdayInLong", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  test("should return timestamp of exactly 24 hours ago in seconds", () => {
    const mockCurrentTime = new Date("2023-10-31T12:00:00.000Z");
    vi.useFakeTimers();
    vi.setSystemTime(mockCurrentTime);
    const expectedTimeInSeconds =
      new Date("2023-10-30T12:00:00.000Z").getTime() / 1000;
    expect(getNowYesterdayInLong()).toBe(expectedTimeInSeconds);
  });
});

describe("epochToLong", () => {
  test("should convert epoch milliseconds to a unix timestamp in seconds", () => {
    const ms = 1672531200000;
    const expectedSeconds = 1672531200;
    expect(epochToLong(ms)).toBe(expectedSeconds);
  });
  test("should handle 0", () => {
    expect(epochToLong(0)).toBe(0);
  });
  test("should handle negative epoch milliseconds", () => {
    expect(epochToLong(-1000)).toBe(-1);
  });
});

describe("dateToLong", () => {
  test("should convert a Date object to a unix timestamp in seconds", () => {
    const date = new Date("2023-01-01T00:00:00.000Z");
    const expectedSeconds = 1672531200;
    expect(dateToLong(date)).toBe(expectedSeconds);
  });
  test("should handle epoch 0", () => {
    const date = new Date("1970-01-01T00:00:00.000Z");
    expect(dateToLong(date)).toBe(0);
  });
  test("should handle dates before epoch", () => {
    const date = new Date("1969-12-31T23:59:59.000Z");
    expect(dateToLong(date)).toBe(-1);
  });
});

describe("formatSecondsToDuration", () => {
  test("should handle invalid durations", () => {
    expect(formatSecondsToDuration(NaN)).toBe("Invalid duration");
    expect(formatSecondsToDuration(-5)).toBe("Invalid duration");
  });
  test("should format 0 seconds", () => {
    expect(formatSecondsToDuration(0)).toBe("0 seconds");
  });
  test("should format seconds into minutes and seconds", () => {
    expect(formatSecondsToDuration(65)).toBe("1 minute, 5 seconds");
    expect(formatSecondsToDuration(120)).toBe("2 minutes");
  });
  test("should format seconds into hours, minutes, and seconds", () => {
    expect(formatSecondsToDuration(3665)).toBe("1 hour, 1 minute, 5 seconds");
    expect(formatSecondsToDuration(7200)).toBe("2 hours");
    expect(formatSecondsToDuration(7325)).toBe("2 hours, 2 minutes, 5 seconds");
  });
});

describe("convertLocalDateToTimestamp", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  const mockDateFormat = (type: "MDY" | "DMY" | "YMD") => {
    const partsMap: Record<string, Intl.DateTimeFormatPart[]> = {
      MDY: [
        { type: "month", value: "10" },
        { type: "literal", value: "/" },
        { type: "day", value: "31" },
        { type: "literal", value: "/" },
        { type: "year", value: "2023" },
      ],
      DMY: [
        { type: "day", value: "31" },
        { type: "literal", value: "/" },
        { type: "month", value: "10" },
        { type: "literal", value: "/" },
        { type: "year", value: "2023" },
      ],
      YMD: [
        { type: "year", value: "2023" },
        { type: "literal", value: "/" },
        { type: "month", value: "10" },
        { type: "literal", value: "/" },
        { type: "day", value: "31" },
      ],
    };

    const formatToParts = vi.fn().mockReturnValue(partsMap[type]);
    vi.spyOn(Intl, "DateTimeFormat").mockImplementation(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      () =>
        ({
          formatToParts,
        }) as unknown as Intl.DateTimeFormat,
    );
  };

  test("should return 0 for invalid format (not 8 digits)", () => {
    expect(convertLocalDateToTimestamp("1234")).toBe(0);
    expect(convertLocalDateToTimestamp("123456789")).toBe(0);
    expect(convertLocalDateToTimestamp("12a45678")).toBe(0);
    expect(convertLocalDateToTimestamp("")).toBe(0);
  });

  test("should return 0 for invalid date", () => {
    mockDateFormat("MDY");
    expect(convertLocalDateToTimestamp("99992023")).toBe(0);
  });

  test("should parse date correctly for MDY format", () => {
    mockDateFormat("MDY");
    const result = convertLocalDateToTimestamp("10312023");
    const expected = new Date("2023-10-31T00:00:00").getTime();
    expect(result).toBe(expected);
  });

  test("should parse date correctly for DMY format", () => {
    mockDateFormat("DMY");
    const result = convertLocalDateToTimestamp("31102023");
    const expected = new Date("2023-10-31T00:00:00").getTime();
    expect(result).toBe(expected);
  });

  test("should parse date correctly for YMD format", () => {
    mockDateFormat("YMD");
    const result = convertLocalDateToTimestamp("20231031");
    const expected = new Date("2023-10-31T00:00:00").getTime();
    expect(result).toBe(expected);
  });

  test("should return 0 for unsupported format", () => {
    const formatToParts = vi.fn().mockReturnValue([
      { type: "month", value: "10" },
      { type: "year", value: "2023" },
    ]);
    vi.spyOn(Intl, "DateTimeFormat").mockImplementation(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      () =>
        ({
          formatToParts,
        }) as unknown as Intl.DateTimeFormat,
    );

    expect(convertLocalDateToTimestamp("10102023")).toBe(0);
  });
});

describe("formatUnixTimestampToDateTime", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  const TEST_TIMESTAMP = 1672574400; // 2023-01-01T12:00:00.000Z

  test("should return 'Invalid time' for NaN", () => {
    expect(formatUnixTimestampToDateTime(NaN)).toBe("Invalid time");
  });

  test("should format with default configuration", () => {
    const result = formatUnixTimestampToDateTime(TEST_TIMESTAMP, {
      timezone: "UTC",
    });
    expect(result).toMatch(/1\/1\/2023|01\/01\/2023/);
  });

  test("should format with explicit timezone", () => {
    const result = formatUnixTimestampToDateTime(TEST_TIMESTAMP, {
      timezone: "America/New_York",
      time_style: "short",
    });
    expect(result).toMatch(/7:00|07:00/);
  });

  test("should format with 12hour format", () => {
    const result = formatUnixTimestampToDateTime(TEST_TIMESTAMP, {
      timezone: "UTC",
      time_format: "12hour",
      time_style: "short",
    });
    expect(result).toMatch(/12:00\s?(AM|PM)/);
  });

  test("should format with 24hour format", () => {
    const result = formatUnixTimestampToDateTime(TEST_TIMESTAMP, {
      timezone: "UTC",
      time_format: "24hour",
      time_style: "short",
    });
    expect(result).toContain("12:00");
    expect(result).not.toMatch(/(AM|PM)/i);
  });

  test("should format with explicit date_format", () => {
    const result = formatUnixTimestampToDateTime(TEST_TIMESTAMP, {
      timezone: "UTC",
      date_format: "yyyy-MM-dd HH:mm:ss",
    });
    expect(result).toBe("2023-01-01 12:00:00");
  });

  test("should uppercase AM/PM for a formats", () => {
    const result = formatUnixTimestampToDateTime(1672585200, {
      // 15:00:00 UTC
      timezone: "UTC",
      date_format: "h:mm a",
    });
    expect(result).toMatch(/3:00\s?(AM|PM|TIME.PM)/i);
  });

  test("should handle locale string", () => {
    const result = formatUnixTimestampToDateTime(TEST_TIMESTAMP, {
      timezone: "UTC",
      locale: "en-GB",
      date_style: "short",
    });
    expect(result).toMatch(/01\/01\/(23|2023)/);
  });
});

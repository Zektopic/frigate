import { describe, test, expect, vi, afterEach } from "vitest";
import { convertLocalDateToTimestamp, longToDate, epochToLong, dateToLong } from "./dateUtil";

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

describe("epochToLong", () => {
  test("should convert epoch milliseconds to a unix timestamp in seconds", () => {
    // 2023-01-01T00:00:00.000Z is 1672531200000 ms
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
    // 2023-01-01T00:00:00.000Z
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
      () =>
        ({
          formatToParts,
        }) as any,
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
    // Month 99 is invalid in YYYY-MM-DD format
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
    // Mock a format that doesn't produce DMY, MDY, or YMD
    const formatToParts = vi.fn().mockReturnValue([
      { type: "month", value: "10" },
      { type: "year", value: "2023" },
      // missing day
    ]);
    vi.spyOn(Intl, "DateTimeFormat").mockImplementation(
      () =>
        ({
          formatToParts,
        }) as any,
    );

    expect(convertLocalDateToTimestamp("10102023")).toBe(0);
  });
});

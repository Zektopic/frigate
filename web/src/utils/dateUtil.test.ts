import { describe, test, expect, vi, afterEach } from "vitest";
import { convertLocalDateToTimestamp, getNowYesterdayInLong } from "./dateUtil";

describe("getNowYesterdayInLong", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  test("should return timestamp of exactly 24 hours ago in seconds", () => {
    // Set a fixed time: 2023-10-31T12:00:00.000Z
    const mockCurrentTime = new Date("2023-10-31T12:00:00.000Z");
    vi.useFakeTimers();
    vi.setSystemTime(mockCurrentTime);

    // Expected is 24 hours earlier: 2023-10-30T12:00:00.000Z in seconds
    const expectedTimeInSeconds = new Date("2023-10-30T12:00:00.000Z").getTime() / 1000;

    expect(getNowYesterdayInLong()).toBe(expectedTimeInSeconds);
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
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      () =>
        ({
          formatToParts,
        }) as unknown as Intl.DateTimeFormat,
    );

    expect(convertLocalDateToTimestamp("10102023")).toBe(0);
  });
});

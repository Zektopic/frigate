import { describe, test, expect, vi, afterEach } from "vitest";
import { convertLocalDateToTimestamp, formatUnixTimestampToDateTime } from "./dateUtil";

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
    // Default Intl formatter string can depend on Node/Vitest version,
    // but typically looks like "1/1/2023" for en-US without time options
    expect(result).toContain("1/1/2023");
  });

  test("should format with explicit timezone", () => {
    const result = formatUnixTimestampToDateTime(TEST_TIMESTAMP, {
      timezone: "America/New_York",
      time_style: "short",
    });
    // 12:00:00 UTC -> 07:00:00 EST
    expect(result).toContain("7:00");
  });

  test("should format with 12hour format", () => {
    const result = formatUnixTimestampToDateTime(TEST_TIMESTAMP, {
      timezone: "UTC",
      time_format: "12hour",
      time_style: "short",
    });
    // Should have AM/PM uppercased
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
    const result = formatUnixTimestampToDateTime(1672585200, { // 15:00:00 UTC
      timezone: "UTC",
      date_format: "h:mm a",
    });
    // i18n should uppercase pm
    expect(result).toMatch(/3:00\s?(AM|PM)/);
  });

  test("should handle locale string", () => {
    const result = formatUnixTimestampToDateTime(TEST_TIMESTAMP, {
      timezone: "UTC",
      locale: "en-GB",
      date_style: "short",
    });
    // en-GB uses DD/MM/YYYY
    expect(result).toContain("01/01/23");
  });
});

import { describe, test, expect, vi, afterEach, beforeEach } from "vitest";
import { formatTimeAgo } from "../formatTimeAgo";

describe("formatTimeAgo", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2024-01-01T12:00:00Z"));
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  test("should format time in the past correctly", () => {
    const timestamp = new Date("2024-01-01T11:59:30Z").getTime() / 1000;
    expect(formatTimeAgo(new Date(timestamp * 1000))).toBe("30 seconds ago");
    // minutes
    expect(formatTimeAgo(new Date("2024-01-01T11:50:00Z"))).toBe(
      "10 minutes ago",
    );
  });

  test("should format 0 seconds ago correctly", () => {
    const date = new Date("2024-01-01T12:00:00Z"); // 0 seconds ago
    expect(formatTimeAgo(date)).toMatch(/in\s0\sseconds/);
  });

  test("should format past times correctly", () => {
    // seconds
    expect(formatTimeAgo(new Date("2024-01-01T11:59:30Z"))).toBe(
      "30 seconds ago",
    );
    // minutes
    expect(formatTimeAgo(new Date("2024-01-01T11:50:00Z"))).toBe(
      "10 minutes ago",
    );
    expect(formatTimeAgo(new Date("2024-01-01T11:59:30Z"))).toMatch(
      /30\sseconds\sago/,
    );
    // minutes
    expect(formatTimeAgo(new Date("2024-01-01T11:50:00Z"))).toMatch(
      /10\sminutes\sago/,
    );
    // hours
    expect(formatTimeAgo(new Date("2024-01-01T07:00:00Z"))).toMatch(
      /5\shours\sago/,
    );
    // days
    expect(formatTimeAgo(new Date("2023-12-29T12:00:00Z"))).toMatch(
      /3\sdays\sago/,
    );
    // weeks
    expect(formatTimeAgo(new Date("2023-12-18T12:00:00Z"))).toMatch(
      /2\sweeks\sago/,
    );
    // months
    expect(formatTimeAgo(new Date("2023-10-01T12:00:00Z"))).toBe(
      "3 months ago",
    );
    expect(formatTimeAgo(new Date("2023-10-01T12:00:00Z"))).toMatch(
      /3\smonths\sago/,
    );
    // years
    expect(formatTimeAgo(new Date("2022-01-01T12:00:00Z"))).toMatch(
      /2\syears\sago/,
    );
  });

  test("should format future times correctly", () => {
    // seconds
    expect(formatTimeAgo(new Date("2024-01-01T12:00:30Z"))).toBe(
      "in 30 seconds",
    );
    // minutes
    expect(formatTimeAgo(new Date("2024-01-01T12:10:00Z"))).toBe(
      "in 10 minutes",
    );
    expect(formatTimeAgo(new Date("2024-01-01T12:00:30Z"))).toMatch(
      /in\s30\sseconds/,
    );
    // minutes
    expect(formatTimeAgo(new Date("2024-01-01T12:10:00Z"))).toMatch(
      /in\s10\sminutes/,
    );
    // hours
    expect(formatTimeAgo(new Date("2024-01-01T17:00:00Z"))).toMatch(
      /in\s5\shours/,
    );
    // days
    expect(formatTimeAgo(new Date("2024-01-04T12:00:00Z"))).toMatch(
      /in\s3\sdays/,
    );
    // weeks
    expect(formatTimeAgo(new Date("2024-01-15T12:00:00Z"))).toMatch(
      /in\s2\sweeks/,
    );
    // months
    expect(formatTimeAgo(new Date("2024-04-01T12:00:00Z"))).toMatch(
      /in\s3\smonths/,
    );
    // years
    expect(formatTimeAgo(new Date("2026-01-01T12:00:00Z"))).toMatch(
      /in\s2\syears/,
    );
  });

  test("should round the duration appropriately", () => {
    // 59 seconds ago -> seconds
    expect(formatTimeAgo(new Date("2024-01-01T11:59:01Z"))).toBe(
      "59 seconds ago",
    );
    // 60 seconds ago -> 1 minute ago
    expect(formatTimeAgo(new Date("2024-01-01T11:59:00Z"))).toBe(
      "1 minute ago",
    );
    expect(formatTimeAgo(new Date("2024-01-01T11:59:01Z"))).toMatch(
      /59\sseconds\sago/,
    );
    // 60 seconds ago -> 1 minute ago
    expect(formatTimeAgo(new Date("2024-01-01T11:59:00Z"))).toMatch(
      /1\sminute\sago/,
    );
    // 1 hour and 29 minutes ago -> 1 hour ago
    expect(formatTimeAgo(new Date("2024-01-01T10:31:00Z"))).toMatch(
      /1\shour\sago/,
    );
    // 1 hour and 30 minutes ago -> duration=-5400s -> -90m -> -1.5h. Math.round(-1.5) = -1
    expect(formatTimeAgo(new Date("2024-01-01T10:30:00Z"))).toMatch(
      /1\shour\sago/,
    );
    // 1 hour and 31 minutes ago -> duration=-5460s -> -91m -> -1.5166h. Math.round(-1.5166) = -2
    expect(formatTimeAgo(new Date("2024-01-01T10:29:00Z"))).toMatch(
      /2\shours\sago/,
    );
  });

  test("should format exact edge cases", () => {
    const d = new Date("2024-01-01T12:00:00Z");
    expect(formatTimeAgo(d)).toBe("in 0 seconds");
  });
});

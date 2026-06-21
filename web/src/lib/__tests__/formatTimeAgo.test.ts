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
    expect(formatTimeAgo(new Date("2024-01-01T11:50:00Z"))).toBe(
      "10 minutes ago",
    );
  });

  it("should format past times correctly", () => {
    // seconds
    expect(formatTimeAgo(new Date("2024-01-01T11:59:30Z"))).toBe(
      "30 seconds ago",
    );
    // minutes
    expect(formatTimeAgo(new Date("2024-01-01T11:50:00Z"))).toBe(
      "10 minutes ago",
    );
    // hours
    expect(formatTimeAgo(new Date("2024-01-01T07:00:00Z"))).toBe("5 hours ago");
    // days
    expect(formatTimeAgo(new Date("2023-12-29T12:00:00Z"))).toBe("3 days ago");
    // weeks
    expect(formatTimeAgo(new Date("2023-12-18T12:00:00Z"))).toBe("2 weeks ago");
    // months
    expect(formatTimeAgo(new Date("2023-10-01T12:00:00Z"))).toBe(
      "3 months ago",
    );
    // years
    expect(formatTimeAgo(new Date("2022-01-01T12:00:00Z"))).toBe("2 years ago");
  });

  it("should format future times correctly", () => {
    // seconds
    expect(formatTimeAgo(new Date("2024-01-01T12:00:30Z"))).toBe(
      "in 30 seconds",
    );
    // minutes
    expect(formatTimeAgo(new Date("2024-01-01T12:10:00Z"))).toBe(
      "in 10 minutes",
    );
    // hours
    expect(formatTimeAgo(new Date("2024-01-01T17:00:00Z"))).toBe("in 5 hours");
    // days
    expect(formatTimeAgo(new Date("2024-01-04T12:00:00Z"))).toBe("in 3 days");
    // weeks
    expect(formatTimeAgo(new Date("2024-01-15T12:00:00Z"))).toBe("in 2 weeks");
    // months
    expect(formatTimeAgo(new Date("2024-04-01T12:00:00Z"))).toBe("in 3 months");
    // years
    expect(formatTimeAgo(new Date("2026-01-01T12:00:00Z"))).toBe("in 2 years");
  });

  it("should round the duration appropriately", () => {
    // 59 seconds ago -> seconds
    expect(formatTimeAgo(new Date("2024-01-01T11:59:01Z"))).toBe(
      "59 seconds ago",
    );
    // 60 seconds ago -> 1 minute ago
    expect(formatTimeAgo(new Date("2024-01-01T11:59:00Z"))).toBe(
      "1 minute ago",
    );
    // 1 hour and 29 minutes ago -> 1 hour ago
    expect(formatTimeAgo(new Date("2024-01-01T10:31:00Z"))).toBe("1 hour ago");
    // 1 hour and 30 minutes ago -> duration=-5400s -> -90m -> -1.5h. Math.round(-1.5) = -1
    expect(formatTimeAgo(new Date("2024-01-01T10:30:00Z"))).toBe("1 hour ago");
    // 1 hour and 31 minutes ago -> duration=-5460s -> -91m -> -1.5166h. Math.round(-1.5166) = -2
    expect(formatTimeAgo(new Date("2024-01-01T10:29:00Z"))).toBe("2 hours ago");
  });

  test("should format exact edge cases", () => {
    const d = new Date("2024-01-01T12:00:00Z");
    expect(formatTimeAgo(d)).toBe("in 0 seconds");
  });
});

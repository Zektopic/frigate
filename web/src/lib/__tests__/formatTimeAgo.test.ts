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

  test("should format time in the future correctly", () => {
    const futureDate = new Date("2024-04-01T12:00:00Z");
    expect(formatTimeAgo(futureDate)).toBe("in 3 months");

    expect(formatTimeAgo(new Date("2024-01-01T12:00:30Z"))).toBe(
      "in 30 seconds",
    );
    expect(formatTimeAgo(new Date("2024-01-01T12:10:00Z"))).toBe(
      "in 10 minutes",
    );
  });

  test("should format exact edge cases", () => {
    const d = new Date("2024-01-01T12:00:00Z");
    expect(formatTimeAgo(d)).toBe("in 0 seconds");
  });
});

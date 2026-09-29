import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { AxiosError, AxiosHeaders } from "axios";
import { SWRConfig } from "swr";
import { swrErrorRetry } from "../swr-retry";

function axiosError(status?: number): AxiosError {
  const error = new AxiosError("request failed");
  if (status !== undefined) {
    error.response = {
      status,
      statusText: "",
      data: null,
      headers: {},
      config: { headers: new AxiosHeaders() },
    };
  }
  return error;
}

describe("swrErrorRetry", () => {
  const config = SWRConfig.defaultValue;

  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  test.each([undefined, 502, 503, 504])(
    "retries transient error %s within the cap even after many attempts",
    (status) => {
      const revalidate = vi.fn();
      swrErrorRetry(axiosError(status), "config", config, revalidate, {
        retryCount: 20,
        dedupe: true,
      });

      vi.advanceTimersByTime(15_000);
      expect(revalidate).toHaveBeenCalledTimes(1);
    },
  );

  test("respects errorRetryCount for transient errors", () => {
    const revalidate = vi.fn();
    swrErrorRetry(
      axiosError(502),
      "config",
      { ...config, errorRetryCount: 2 },
      revalidate,
      { retryCount: 3, dedupe: true },
    );

    vi.advanceTimersByTime(60_000);
    expect(revalidate).not.toHaveBeenCalled();
  });

  test.each([401, 403, 404])(
    "defers %s to the default SWR backoff",
    (status) => {
      const defaultRetry = vi
        .spyOn(SWRConfig.defaultValue, "onErrorRetry")
        .mockImplementation(() => {});
      const revalidate = vi.fn();
      swrErrorRetry(axiosError(status), "config", config, revalidate, {
        retryCount: 1,
        dedupe: true,
      });

      expect(defaultRetry).toHaveBeenCalledTimes(1);
      vi.advanceTimersByTime(60_000);
      expect(revalidate).not.toHaveBeenCalled();
    },
  );
});

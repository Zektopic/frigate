import axios from "axios";
import { SWRConfig } from "swr";
import type { SWRConfiguration } from "swr";

// SWR's default backoff doubles errorRetryInterval (5s) on every attempt,
// up to 2^8, so the gap between retries grows to roughly 21 minutes, and
// refreshInterval polling is suspended while a key is in error. After a
// backend restart that left views on a spinner or on stale data long
// after the backend was back, until the page was reloaded by hand.
const TRANSIENT_RETRY_BASE_MS = 1000;
const TRANSIENT_RETRY_MAX_MS = 15_000;

// A restarting backend shows up as a network error (connection refused
// or reset) or a 5xx from the reverse proxy; those clear on their own
function isTransientError(error: unknown): boolean {
  if (!axios.isAxiosError(error)) {
    return false;
  }

  const status = error.response?.status;
  return status === undefined || status >= 500;
}

export const swrErrorRetry: NonNullable<SWRConfiguration["onErrorRetry"]> = (
  error,
  key,
  config,
  revalidate,
  opts,
) => {
  if (!isTransientError(error)) {
    SWRConfig.defaultValue.onErrorRetry(error, key, config, revalidate, opts);
    return;
  }

  if (
    config.errorRetryCount !== undefined &&
    opts.retryCount > config.errorRetryCount
  ) {
    return;
  }

  const delay = Math.min(
    TRANSIENT_RETRY_BASE_MS * 2 ** opts.retryCount * (0.5 + Math.random()),
    TRANSIENT_RETRY_MAX_MS,
  );
  setTimeout(() => revalidate(opts), delay);
};

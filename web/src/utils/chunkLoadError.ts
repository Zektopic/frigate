// Minimum gap between two automatic reloads, so a chunk that is genuinely
// broken cannot put the page in a reload loop.
const AUTO_RELOAD_GUARD_MS = 30_000;
const AUTO_RELOAD_KEY = "frigate-chunk-error-reload";

/**
 * True for the errors browsers raise when a lazily loaded route chunk cannot
 * be fetched: the server was restarting, or a new build replaced the hashed
 * file names this page still references.
 */
export function isChunkLoadError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return /Failed to fetch dynamically imported module|error loading dynamically imported module|Importing a module script failed|Unable to preload CSS/i.test(
    message,
  );
}

export function reloadIfNotRecentlyReloaded(): void {
  try {
    const last = Number(sessionStorage.getItem(AUTO_RELOAD_KEY) ?? 0);

    if (Date.now() - last < AUTO_RELOAD_GUARD_MS) {
      return;
    }

    sessionStorage.setItem(AUTO_RELOAD_KEY, String(Date.now()));
  } catch {
    // storage unavailable, reload anyway
  }

  window.location.reload();
}

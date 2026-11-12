import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  isChunkLoadError,
  reloadIfNotRecentlyReloaded,
} from "../chunkLoadError";

describe("isChunkLoadError", () => {
  it("matches the dynamic import failures of each browser engine", () => {
    expect(
      isChunkLoadError(
        new TypeError(
          "Failed to fetch dynamically imported module: http://frigate/assets/Live-abc.js",
        ),
      ),
    ).toBe(true);
    expect(
      isChunkLoadError(
        new TypeError(
          "error loading dynamically imported module: http://frigate/assets/Live-abc.js",
        ),
      ),
    ).toBe(true);
    expect(
      isChunkLoadError(new TypeError("Importing a module script failed.")),
    ).toBe(true);
  });

  it("ignores ordinary render errors", () => {
    expect(
      isChunkLoadError(new TypeError("Cannot read properties of undefined")),
    ).toBe(false);
  });
});

describe("reloadIfNotRecentlyReloaded", () => {
  const reload = vi.fn();

  beforeEach(() => {
    sessionStorage.clear();
    reload.mockClear();
    vi.stubGlobal("location", { ...window.location, reload });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("reloads once and then refuses to loop", () => {
    reloadIfNotRecentlyReloaded();
    reloadIfNotRecentlyReloaded();
    expect(reload).toHaveBeenCalledTimes(1);
  });
});

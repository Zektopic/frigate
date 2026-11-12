import { act } from "react";
import { createRoot, Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import ErrorBoundary from "../ErrorBoundary";

function Thrower({ message }: { message: string }): never {
  throw new TypeError(message);
}

describe("ErrorBoundary", () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    (
      globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }
    ).IS_REACT_ACT_ENVIRONMENT = true;
    // React reports caught errors to the console; keep the output clean
    vi.spyOn(console, "error").mockImplementation(() => {});
    container = document.createElement("div");
    root = createRoot(container);
  });

  afterEach(() => {
    act(() => root.unmount());
    vi.restoreAllMocks();
  });

  it("renders a fallback instead of unmounting the app", () => {
    act(() => {
      root.render(
        <ErrorBoundary resetKey="/">
          <Thrower message="Failed to fetch dynamically imported module: /assets/Live-abc.js" />
        </ErrorBoundary>,
      );
    });

    expect(container.textContent).toContain("errorBoundary.chunkDescription");
  });

  it("clears the error when the reset key changes", () => {
    act(() => {
      root.render(
        <ErrorBoundary resetKey="/review">
          <Thrower message="boom" />
        </ErrorBoundary>,
      );
    });
    expect(container.textContent).toContain("errorBoundary.description");

    act(() => {
      root.render(
        <ErrorBoundary resetKey="/">
          <span>recovered</span>
        </ErrorBoundary>,
      );
    });
    expect(container.textContent).toBe("recovered");
  });
});

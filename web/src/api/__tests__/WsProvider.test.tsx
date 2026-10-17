import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { act } from "react";
import { createRoot, Root } from "react-dom/client";
import useSWR, { SWRConfig } from "swr";
import { WsProvider } from "../WsProvider";

class FakeWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;
  static instances: FakeWebSocket[] = [];

  readyState = FakeWebSocket.CONNECTING;
  sent: string[] = [];
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;

  constructor(public url: string) {
    FakeWebSocket.instances.push(this);
  }

  send(data: string) {
    this.sent.push(data);
  }

  close() {
    this.readyState = FakeWebSocket.CLOSED;
  }

  serverOpen() {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen?.();
  }

  serverDrop() {
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.();
  }
}

describe("WsProvider", () => {
  let container: HTMLDivElement;
  let root: Root;
  const fetcher = vi.fn();

  function Consumer() {
    useSWR("config", fetcher);
    return null;
  }

  async function mount() {
    await act(async () => {
      root.render(
        <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
          <WsProvider>
            <Consumer />
          </WsProvider>
        </SWRConfig>,
      );
    });
  }

  beforeEach(() => {
    (
      globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }
    ).IS_REACT_ACT_ENVIRONMENT = true;
    vi.useFakeTimers();
    vi.stubGlobal("WebSocket", FakeWebSocket);
    FakeWebSocket.instances = [];
    fetcher.mockResolvedValue({ ok: true });
    container = document.createElement("div");
    root = createRoot(container);
  });

  afterEach(() => {
    act(() => root.unmount());
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  test("does not refetch SWR data on the first connection", async () => {
    await mount();
    expect(fetcher).toHaveBeenCalledTimes(1);

    await act(async () => FakeWebSocket.instances[0].serverOpen());
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  test("refetches SWR data after reconnecting", async () => {
    await mount();
    await act(async () => FakeWebSocket.instances[0].serverOpen());
    expect(fetcher).toHaveBeenCalledTimes(1);

    await act(async () => FakeWebSocket.instances[0].serverDrop());
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000);
    });
    expect(FakeWebSocket.instances).toHaveLength(2);

    await act(async () => FakeWebSocket.instances[1].serverOpen());
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  test("reconnects a silent socket without waiting for onclose", async () => {
    await mount();
    await act(async () => FakeWebSocket.instances[0].serverOpen());

    // close() on a half-open socket never fires onclose here, as with a
    // peer that does not answer the closing handshake
    await act(async () => {
      await vi.advanceTimersByTimeAsync(120_000);
    });

    expect(FakeWebSocket.instances.length).toBeGreaterThanOrEqual(2);
    await act(async () => FakeWebSocket.instances[1].serverOpen());
    expect(fetcher).toHaveBeenCalledTimes(2);
  });
});

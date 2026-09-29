import { baseUrl } from "./baseUrl";
import { ReactNode, useCallback, useEffect, useRef } from "react";
import { useSWRConfig } from "swr";
import { WsSendContext } from "./wsContext";
import type { Update } from "./wsContext";
import {
  invalidateCameraActivityCache,
  processWsMessage,
  resetWsStore,
} from "./ws";

// A silently half-open TCP connection (NAT/conntrack expiry, Wi-Fi
// roam, VPN re-key, container network hiccup) never fires onclose, so
// the reconnect logic below would never run and the whole UI would sit
// frozen on stale data while still looking connected. Browsers do not
// expose protocol-level ping/pong to JS, so we do an application-level
// round trip: send a ping on an interval and treat a long silence as a
// dead socket.
const WS_PING_INTERVAL_MS = 20_000;
// Must clear two independent floors: several ping intervals, so a
// dropped ping is not treated as a dead socket, and the 60s default
// mqtt.stats_interval, which is the slowest source of unsolicited
// traffic. That second floor matters if the backend predates the pong
// handler — natural traffic alone must then keep the connection
// considered alive rather than driving a reconnect loop.
const WS_IDLE_TIMEOUT_MS = 90_000;

export function WsProvider({ children }: { children: ReactNode }) {
  const wsUrl = `${baseUrl.replace(/^http/, "ws")}ws`;
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectAttempt = useRef(0);
  const unmounted = useRef(false);
  const pendingSends = useRef<Map<string, unknown>>(new Map());
  const lastMessageAt = useRef(0);
  const livenessTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  // Kept in a ref so a new mutate identity can never re-run the connect
  // effect below, whose cleanup tears down the socket and the ws store
  const { mutate } = useSWRConfig();
  const mutateRef = useRef(mutate);

  useEffect(() => {
    mutateRef.current = mutate;
  }, [mutate]);

  const sendJsonMessage = useCallback((msg: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    } else if (msg && typeof msg === "object" && "topic" in msg) {
      // Sends issued before the socket reaches OPEN (or during a reconnect
      // window) are buffered here and flushed in onopen
      pendingSends.current.set(String((msg as { topic: unknown }).topic), msg);
    }
  }, []);

  useEffect(() => {
    unmounted.current = false;
    const queue = pendingSends.current;

    function connect() {
      if (unmounted.current) return;

      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;
      lastMessageAt.current = Date.now();

      ws.onopen = () => {
        // Any failed attempt or dropped socket since the last open means
        // the backend may have restarted. HTTP data (config, profile,
        // reviews, ...) is then stale or parked in SWR's error backoff,
        // which can take many minutes to clear without a manual reload,
        // so refresh every mounted SWR hook as a reload would.
        const recovered = reconnectAttempt.current > 0;
        reconnectAttempt.current = 0;
        lastMessageAt.current = Date.now();
        // events may have been missed while disconnected — the snapshot
        // requested below must fully apply even if byte-identical
        invalidateCameraActivityCache();
        ws.send(
          JSON.stringify({ topic: "onConnect", message: "", retain: false }),
        );
        for (const queued of queue.values()) {
          ws.send(JSON.stringify(queued));
        }
        queue.clear();

        if (recovered) {
          mutateRef.current(() => true);
        }
      };

      ws.onmessage = (event: MessageEvent) => {
        // Any inbound traffic proves the socket is alive, not just pong.
        lastMessageAt.current = Date.now();
        processWsMessage(event.data as string);
      };

      ws.onclose = () => {
        stopLivenessCheck();
        scheduleReconnect();
      };

      ws.onerror = () => {
        ws.close();
      };

      startLivenessCheck(ws);
    }

    function scheduleReconnect() {
      if (unmounted.current) return;
      const delay = Math.min(1000 * 2 ** reconnectAttempt.current, 30000);
      reconnectAttempt.current++;
      reconnectTimer.current = setTimeout(connect, delay);
    }

    function startLivenessCheck(ws: WebSocket) {
      stopLivenessCheck();
      livenessTimer.current = setInterval(() => {
        if (unmounted.current || wsRef.current !== ws) {
          return;
        }

        if (Date.now() - lastMessageAt.current > WS_IDLE_TIMEOUT_MS) {
          // Nothing has arrived in far longer than the ping interval:
          // the connection is half-open. close() on it waits for a
          // closing handshake the peer will never answer (up to 60s in
          // Chromium) before onclose fires, so detach the socket and
          // reconnect right away instead of waiting on it.
          stopLivenessCheck();
          ws.onopen = null;
          ws.onmessage = null;
          ws.onclose = null;
          ws.onerror = null;
          ws.close();
          scheduleReconnect();
          return;
        }

        if (ws.readyState === WebSocket.OPEN) {
          try {
            ws.send(JSON.stringify({ topic: "ping", payload: "" }));
          } catch {
            // A failed send means the socket is already gone; let the
            // idle timeout above drive the reconnect.
          }
        }
      }, WS_PING_INTERVAL_MS);
    }

    function stopLivenessCheck() {
      if (livenessTimer.current) {
        clearInterval(livenessTimer.current);
        livenessTimer.current = null;
      }
    }

    connect();

    return () => {
      unmounted.current = true;
      stopLivenessCheck();
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current);
      }
      const ws = wsRef.current;
      if (ws) {
        ws.onopen = null;
        ws.onmessage = null;
        ws.onclose = null;
        ws.onerror = null;
        ws.close();
      }
      queue.clear();
      resetWsStore();
    };
  }, [wsUrl]);

  const send = useCallback(
    (message: Update) => {
      sendJsonMessage({
        topic: message.topic,
        payload: message.payload,
        retain: message.retain,
      });
    },
    [sendJsonMessage],
  );

  return (
    <WsSendContext.Provider value={send}>{children}</WsSendContext.Provider>
  );
}

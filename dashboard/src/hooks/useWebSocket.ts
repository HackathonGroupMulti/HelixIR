import { useEffect, useRef, useCallback } from "react";
import type { WsMessage } from "../types";

export function useWebSocket(
  url: string,
  onMessage: (msg: WsMessage) => void,
) {
  const wsRef = useRef<WebSocket | null>(null);
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;

  const connect = useCallback(() => {
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data) as WsMessage;
        onMessageRef.current(msg);
      } catch {
        /* ignore malformed frames */
      }
    };

    ws.onclose = () => {
      // Reconnect after 2 s
      setTimeout(connect, 2000);
    };
  }, [url]);

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
    };
  }, [connect]);
}

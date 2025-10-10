import { useEffect } from 'react';

export function usePolling(callback, interval, enabled = true) {
  useEffect(() => {
    if (!enabled) return;
    callback();
    const id = setInterval(callback, interval);
    return () => clearInterval(id);
  }, [callback, interval, enabled]);
}

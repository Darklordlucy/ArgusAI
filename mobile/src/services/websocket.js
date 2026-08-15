/**
 * Asphr Mobile WebSocket Service
 * Handles live hazard alerts, weather updates, and crash events.
 */

const BASE_URL = import.meta.env.VITE_API_URL || 'https://darklord11-asphr-backend.hf.space';
const WS_URL = BASE_URL.replace(/^http/, 'ws') + '/ws';

class MobileWebSocketService {
  constructor() {
    this.ws = null;
    this.listeners = new Set();
    this.isConnected = false;
    this.reconnectTimer = null;
  }

  connect() {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    try {
      this.ws = new WebSocket(WS_URL);

      this.ws.onopen = () => {
        console.log('[Mobile WS] Connected to live alert stream at', WS_URL);
        this.isConnected = true;
        this.notifyListeners({ type: 'status_change', isConnected: true });
      };

      this.ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          this.notifyListeners(message);
        } catch (e) {
          console.error('[Mobile WS] Failed to parse message:', e);
        }
      };

      this.ws.onclose = () => {
        console.log('[Mobile WS] Disconnected. Reconnecting in 5s...');
        this.isConnected = false;
        this.notifyListeners({ type: 'status_change', isConnected: false });
        this.scheduleReconnect();
      };

      this.ws.onerror = (err) => {
        console.warn('[Mobile WS] Socket error:', err);
        this.ws.close();
      };
    } catch (e) {
      console.warn('[Mobile WS] Connection initialization failed:', e);
      this.scheduleReconnect();
    }
  }

  scheduleReconnect() {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.reconnectTimer = setTimeout(() => this.connect(), 5000);
  }

  subscribe(callback) {
    this.listeners.add(callback);
    return () => this.listeners.delete(callback);
  }

  notifyListeners(data) {
    this.listeners.forEach((callback) => callback(data));
  }

  reportHazard(lat, lon, hazardType = 'pothole', hazardScore = 0.8) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'report_hazard',
        latitude: lat,
        longitude: lon,
        hazard_type: hazardType,
        hazard_score: hazardScore,
        expires_in_sec: 7200
      }));
      return true;
    }
    return false;
  }
}

export const wsService = new MobileWebSocketService();

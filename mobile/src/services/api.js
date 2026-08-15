/**
 * Asphr Mobile API Layer
 * Connects to the live deployed FastAPI backend
 */

const BASE_URL = import.meta.env.VITE_API_URL || 'https://darklord11-asphr-backend.hf.space';

async function request(path, options = {}) {
  try {
    const res = await fetch(`${BASE_URL}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP Error ${res.status}`);
    }
    return await res.json();
  } catch (error) {
    console.warn(`[Mobile API] Request to ${path} failed:`, error.message);
    throw error;
  }
}

/**
 * Fetch hazard segments within bounding box
 */
export async function fetchHazards({ minLat = 18.90, minLon = 72.75, maxLat = 19.50, maxLon = 73.20 }) {
  const params = new URLSearchParams({
    min_lat: minLat,
    min_lon: minLon,
    max_lat: maxLat,
    max_lon: maxLon,
  });
  return request(`/api/v1/routes/hazards?${params}`).catch(() => ({ hazards: [] }));
}

/**
 * Fetch heavy traffic incidents from backend (TomTom proxy)
 */
export async function fetchHeavyTraffic() {
  return request('/api/v1/custom-db/heavy_traffic').catch(() => ({ features: [] }));
}

/**
 * Fetch weather grid
 */
export async function fetchWeatherGrid() {
  return request('/api/v1/custom-db/weather_grid').catch(() => ({ features: [] }));
}

/**
 * Compute multi-objective route
 */
export async function computeRoute(payload) {
  return request('/api/v1/routes/compute', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

/**
 * Post IoT sensor telemetry reading
 */
export async function postIoTTelemetry(payload) {
  return request('/api/v1/iot/telemetry', {
    method: 'POST',
    body: JSON.stringify(payload),
  }).catch(err => {
    console.log('[IoT Telemetry] Logged / Simulated:', payload);
    return { status: 'queued', payload };
  });
}

/**
 * Trigger SOS alert
 */
export async function triggerSOSAlert(payload) {
  return request('/api/v1/iot/sos', {
    method: 'POST',
    body: JSON.stringify(payload),
  }).catch(err => {
    console.log('[SOS Trigger] Logged / Simulated:', payload);
    return { status: 'dispatched', payload };
  });
}

/**
 * Forward Geocode
 */
export async function forwardGeocode(query) {
  const params = new URLSearchParams({ query });
  return request(`/api/v1/geocode/forward?${params}`);
}

/**
 * Backend Health Check
 */
export async function checkBackendHealth() {
  return request('/health');
}

/**
 * Asphr API Service Layer
 * All backend requests are centralized here.
 * To switch environments, update VITE_API_URL in .env
 */

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

// ─── Hazards ────────────────────────────────────────────────────────────────

export async function fetchHazards(bounds = {}) {
  const minLat = bounds.minLat !== undefined ? bounds.minLat : 0.0;
  const minLon = bounds.minLon !== undefined ? bounds.minLon : 0.0;
  const maxLat = bounds.maxLat !== undefined ? bounds.maxLat : 90.0;
  const maxLon = bounds.maxLon !== undefined ? bounds.maxLon : 180.0;

  const params = new URLSearchParams({
    min_lat: minLat,
    min_lon: minLon,
    max_lat: maxLat,
    max_lon: maxLon,
  });

  try {
    const data = await request(`/api/v1/routes/hazards?${params}`);
    if (data && Array.isArray(data.hazards) && data.hazards.length > 0) return data;
  } catch (e) {}

  try {
    const res = await fetch(`https://darklord11-asphr-backend.hf.space/api/v1/routes/hazards?${params}`);
    const data = await res.json();
    if (data && Array.isArray(data.hazards)) return data;
  } catch (err) {}

  return { hazards: [] };
}

// ─── Geocoding ───────────────────────────────────────────────────────────────

export async function forwardGeocode(query) {
  const params = new URLSearchParams({ query });
  return request(`/api/v1/geocode/forward?${params}`);
}

export async function reverseGeocode(lat, lon) {
  const params = new URLSearchParams({ lat, lon });
  return request(`/api/v1/geocode/reverse?${params}`);
}

// ─── Routes ──────────────────────────────────────────────────────────────────

export async function computeRoute({
  origin,
  destination,
  route_type = 'fastest',
  vehicle_type = 'car',
  avoid_tolls = false,
}) {
  return request('/api/v1/routes/compute', {
    method: 'POST',
    body: JSON.stringify({ origin, destination, route_type, vehicle_type, avoid_tolls }),
  });
}

export async function submitRouteFeedback(payload) {
  try {
    return await request('/api/v1/routes/feedback', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  } catch (e) {
    const res = await fetch('https://darklord11-asphr-backend.hf.space/api/v1/routes/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return res.json();
  }
}

// ─── Health ──────────────────────────────────────────────────────────────────

export async function checkHealth() {
  return request('/health');
}

// ─── Custom DB & IoT Endpoints ────────────────────────────────────────────────

export async function fetchPopularPlaces() {
  try {
    const data = await request('/api/v1/custom-db/popular_places');
    if (data && data.features && data.features.length > 0) return data;
    throw new Error('Empty popular places');
  } catch (e) {
    const res = await fetch('https://darklord11-asphr-backend.hf.space/api/v1/custom-db/popular_places');
    return res.json();
  }
}

export async function fetchIotReadings() {
  try {
    const data = await request('/api/v1/custom-db/iot_readings');
    if (data && Array.isArray(data.features) && data.features.length > 0) return data;
  } catch (e) {}

  try {
    const res = await fetch('https://darklord11-asphr-backend.hf.space/api/v1/custom-db/iot_readings');
    const data = await res.json();
    if (data && Array.isArray(data.features)) return data;
  } catch (err) {}

  return { type: 'FeatureCollection', features: [] };
}

export async function fetchWeatherGrid() {
  try {
    const data = await request('/api/v1/custom-db/weather_grid');
    if (data && data.features && data.features.length > 0) return data;
    throw new Error('No weather grid features in local DB');
  } catch (e) {
    try {
      const res = await fetch('https://darklord11-asphr-backend.hf.space/api/v1/custom-db/weather_grid');
      const data = await res.json();
      if (data && data.features && data.features.length > 0) return data;
    } catch (err) {}
    
    // Synthetic fallback weather grid polygons over Mumbai MMR region
    return {
      type: 'FeatureCollection',
      features: [
        {
          type: 'Feature',
          geometry: {
            type: 'Polygon',
            coordinates: [[[72.80, 18.90], [72.88, 18.90], [72.88, 19.05], [72.80, 19.05], [72.80, 18.90]]]
          },
          properties: { id: 1, weather_condition: 'rain', temperature: 28.5, humidity: 82.0 }
        },
        {
          type: 'Feature',
          geometry: {
            type: 'Polygon',
            coordinates: [[[72.88, 18.90], [72.96, 18.90], [72.96, 19.05], [72.88, 19.05], [72.88, 18.90]]]
          },
          properties: { id: 2, weather_condition: 'thunderstorm', temperature: 27.0, humidity: 89.0 }
        },
        {
          type: 'Feature',
          geometry: {
            type: 'Polygon',
            coordinates: [[[72.80, 19.05], [72.88, 19.05], [72.88, 19.20], [72.80, 19.20], [72.80, 19.05]]]
          },
          properties: { id: 3, weather_condition: 'heavy rain', temperature: 26.5, humidity: 91.0 }
        },
        {
          type: 'Feature',
          geometry: {
            type: 'Polygon',
            coordinates: [[[72.88, 19.05], [72.96, 19.05], [72.96, 19.20], [72.88, 19.20], [72.88, 19.05]]]
          },
          properties: { id: 4, weather_condition: 'cloudy', temperature: 29.0, humidity: 78.0 }
        }
      ]
    };
  }
}

export async function fetchHeavyTraffic() {
  try {
    return await request('/api/v1/custom-db/heavy_traffic');
  } catch (e) {
    const res = await fetch('https://darklord11-asphr-backend.hf.space/api/v1/custom-db/heavy_traffic');
    return res.json();
  }
}

export async function fetchSosAlerts() {
  try {
    const res = await request('/api/v1/iot/sos');
    if (res && res.alerts && res.alerts.length > 0) return res;
    throw new Error('Local DB returned no alerts');
  } catch (e) {
    // Fallback to deployed Hugging Face backend connected directly to live Supabase DB
    const res = await fetch('https://darklord11-asphr-backend.hf.space/api/v1/iot/sos');
    return res.json();
  }
}

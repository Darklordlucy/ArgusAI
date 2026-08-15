import React, { useEffect, useRef, useState } from 'react';
import mapboxgl from 'mapbox-gl';

const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN || 'pk.eyJ1IjoibWFwYm94IiwiYSI6ImNpejY4M29iazA2Z2gycXA4N2pmbDZmangifQ.-g_vE53SD2WrJ6tFX7QHmA';
mapboxgl.accessToken = MAPBOX_TOKEN;

// Mumbai default viewport
const DEFAULT_CENTER = [72.8777, 19.0760];
const DEFAULT_ZOOM = 12;

const HomeMapView = ({
  activeOverlay = 'traffic', // 'traffic' | 'hazards' | 'crashes'
  hazardsData = [],
  trafficData = [],
  crashesData = [],
  onSelectMarker = () => { },
}) => {
  const mapContainer = useRef(null);
  const map = useRef(null);
  const markersRef = useRef([]);
  const [mapLoaded, setMapLoaded] = useState(false);

  // Initialize Map in Light Mode
  useEffect(() => {
    if (map.current) return;

    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/light-v11',
      center: DEFAULT_CENTER,
      zoom: DEFAULT_ZOOM,
      attributionControl: false,
    });

    map.current.addControl(
      new mapboxgl.NavigationControl({ showCompass: false }),
      'top-right'
    );

    map.current.on('load', () => {
      setMapLoaded(true);

      // Source for live traffic line features
      map.current.addSource('traffic-lines', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] },
      });

      // Traffic line layer - Green (#8f9d68) for smooth flow
      map.current.addLayer({
        id: 'traffic-layer',
        type: 'line',
        source: 'traffic-lines',
        layout: {
          'line-join': 'round',
          'line-cap': 'round',
        },
        paint: {
          'line-color': ['get', 'color'],
          'line-width': 5,
          'line-opacity': 0.85,
        },
      });

      // Source for hazard GeoJSON lines
      map.current.addSource('hazard-lines', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] },
      });

      map.current.addLayer({
        id: 'hazard-layer',
        type: 'line',
        source: 'hazard-lines',
        layout: {
          'line-join': 'round',
          'line-cap': 'round',
        },
        paint: {
          'line-color': [
            'interpolate',
            ['linear'],
            ['get', 'hazard_score'],
            0.0, '#8f9d68',
            0.4, '#eab308',
            0.7, '#f97316',
            1.0, '#ef4444'
          ],
          'line-width': 6,
          'line-opacity': 0.9,
        },
      });
    });

    return () => {
      if (map.current) {
        map.current.remove();
        map.current = null;
      }
    };
  }, []);

  // Update Markers & Layers based on activeOverlay
  useEffect(() => {
    if (!map.current || !mapLoaded) return;

    markersRef.current.forEach((m) => m.remove());
    markersRef.current = [];

    if (activeOverlay === 'traffic') {
      const trafficFeatures = trafficData.map((inc) => ({
        type: 'Feature',
        geometry: inc.geometry,
        properties: {
          color: inc.properties?.color || '#8f9d68',
          name: inc.properties?.name || 'Live Traffic Segment',
        },
      }));

      if (map.current.getSource('traffic-lines')) {
        map.current.getSource('traffic-lines').setData({
          type: 'FeatureCollection',
          features: trafficFeatures,
        });
      }

      trafficData.forEach((inc) => {
        if (!inc.geometry?.coordinates) return;
        const el = document.createElement('div');
        el.className = 'w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold text-white shadow-md border-2 border-white cursor-pointer transition-transform hover:scale-125';
        el.style.backgroundColor = inc.properties?.color || '#8f9d68';
        el.innerText = inc.properties?.speed_kmh ? `${Math.round(inc.properties.speed_kmh)}` : 'kmh';

        el.addEventListener('click', () => onSelectMarker(inc));

        const marker = new mapboxgl.Marker(el)
          .setLngLat(inc.geometry.coordinates)
          .addTo(map.current);

        markersRef.current.push(marker);
      });
    } else if (activeOverlay === 'hazards') {
      const hazardFeatures = hazardsData.map((h) => ({
        type: 'Feature',
        geometry: h.geometry,
        properties: {
          hazard_score: h.hazard_score,
          hazard_type: h.hazard_type,
        },
      }));

      if (map.current.getSource('hazard-lines')) {
        map.current.getSource('hazard-lines').setData({
          type: 'FeatureCollection',
          features: hazardFeatures,
        });
      }

      hazardsData.forEach((h) => {
        if (!h.geometry?.coordinates) return;
        const coords = h.geometry.type === 'LineString' ? h.geometry.coordinates[0] : h.geometry.coordinates;
        if (!coords || coords.length < 2) return;

        const el = document.createElement('div');
        el.className = 'w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold text-white shadow-lg border-2 border-white cursor-pointer animate-pulse';
        el.style.backgroundColor = h.hazard_score > 0.7 ? '#ef4444' : '#8f9d68';
        el.innerHTML = '⚠️';

        el.addEventListener('click', () => onSelectMarker(h));

        const marker = new mapboxgl.Marker(el)
          .setLngLat(coords)
          .addTo(map.current);

        markersRef.current.push(marker);
      });
    } else if (activeOverlay === 'crashes') {
      crashesData.forEach((crash) => {
        if (!crash.latitude || !crash.longitude) return;

        const el = document.createElement('div');
        el.className = 'relative flex items-center justify-center cursor-pointer';
        el.innerHTML = `
          <span class="absolute w-8 h-8 rounded-full bg-red-500/40 animate-ping"></span>
          <div class="w-8 h-8 rounded-full bg-red-600 border-2 border-white flex items-center justify-center text-white text-xs font-bold shadow-xl">
            🚨
          </div>
        `;

        el.addEventListener('click', () => onSelectMarker(crash));

        const marker = new mapboxgl.Marker(el)
          .setLngLat([crash.longitude, crash.latitude])
          .addTo(map.current);

        markersRef.current.push(marker);
      });
    }
  }, [activeOverlay, hazardsData, trafficData, crashesData, mapLoaded]);

  return (
    <div className="relative w-full h-[360px] rounded-2xl overflow-hidden border border-slate-200 shadow-md bg-white">
      <div ref={mapContainer} className="w-full h-full" />
      <div className="absolute top-3 left-3 bg-white/90 backdrop-blur-md px-3 py-1.5 rounded-lg border border-slate-200 text-xs font-medium text-slate-700 flex items-center gap-2 shadow-sm">
        <span className="w-2 h-2 rounded-full bg-[#8f9d68]" />
        <span>Mumbai Interactive Map</span>
      </div>
    </div>
  );
};

export default HomeMapView;

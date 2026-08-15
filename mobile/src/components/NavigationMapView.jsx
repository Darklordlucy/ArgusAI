import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

const NavigationMapView = ({
  origin = { lat: 19.0657, lon: 72.8687, name: 'Bandra Kurla Complex (BKC)' },
  destination = { lat: 18.9400, lon: 72.8353, name: 'CSMT Station, Mumbai' },
  routeType = 'fastest',
  height = 'h-[260px]'
}) => {
  const mapContainer = useRef(null);
  const mapInstance = useRef(null);
  const polylineRef = useRef(null);
  const polylineOutlineRef = useRef(null);
  const originMarkerRef = useRef(null);
  const destMarkerRef = useRef(null);

  const createMarkerIcon = (letter, bgColor, ringColor) =>
    L.divIcon({
      className: 'custom-leaflet-marker',
      html: `<div style="
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background-color: ${bgColor};
        border: 2.5px solid white;
        color: white;
        font-weight: 900;
        font-size: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        outline: 3px solid ${ringColor};
      ">${letter}</div>`,
      iconSize: [30, 30],
      iconAnchor: [15, 15],
    });

  useEffect(() => {
    if (!mapContainer.current || mapInstance.current) return;

    const map = L.map(mapContainer.current, {
      center: [19.02, 72.85],
      zoom: 12,
      zoomControl: true,
      attributionControl: false
    });

    L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
      maxZoom: 19,
      subdomains: 'abcd',
    }).addTo(map);

    const initialCoords = [
      [origin.lat, origin.lon],
      [(origin.lat + destination.lat) / 2 + 0.004, (origin.lon + destination.lon) / 2 - 0.005],
      [destination.lat, destination.lon]
    ];

    polylineOutlineRef.current = L.polyline(initialCoords, {
      color: '#ffffff',
      weight: 8,
      opacity: 0.9,
      lineCap: 'round',
      lineJoin: 'round'
    }).addTo(map);

    polylineRef.current = L.polyline(initialCoords, {
      color: routeType === 'fastest' ? '#ffd700' : '#8f9d68',
      weight: 5,
      opacity: 1,
      lineCap: 'round',
      lineJoin: 'round'
    }).addTo(map);

    originMarkerRef.current = L.marker([origin.lat, origin.lon], {
      icon: createMarkerIcon('A', '#8f9d68', 'rgba(143, 157, 104, 0.35)')
    }).bindPopup(`<b>Start</b>: ${origin.name}`).addTo(map);

    destMarkerRef.current = L.marker([destination.lat, destination.lon], {
      icon: createMarkerIcon('B', '#ef4444', 'rgba(239, 68, 68, 0.35)')
    }).bindPopup(`<b>Destination</b>: ${destination.name}`).addTo(map);

    map.fitBounds(initialCoords, { padding: [35, 35] });
    mapInstance.current = map;

    const timer = setTimeout(() => {
      if (mapInstance.current) mapInstance.current.invalidateSize();
    }, 300);

    return () => {
      clearTimeout(timer);
      if (mapInstance.current) {
        mapInstance.current.remove();
        mapInstance.current = null;
      }
    };
  }, []);

  // Update route polyline and markers on location/type change
  useEffect(() => {
    if (!mapInstance.current) return;

    const coords = [
      [origin.lat, origin.lon],
      [(origin.lat + destination.lat) / 2 + 0.004, (origin.lon + destination.lon) / 2 - 0.005],
      [destination.lat, destination.lon]
    ];

    if (polylineRef.current) {
      polylineRef.current.setLatLngs(coords);
      polylineRef.current.setStyle({
        color: routeType === 'fastest' ? '#ffd700' : '#8f9d68'
      });
    }

    if (polylineOutlineRef.current) {
      polylineOutlineRef.current.setLatLngs(coords);
    }

    if (originMarkerRef.current) {
      originMarkerRef.current.setLatLng([origin.lat, origin.lon]);
    }

    if (destMarkerRef.current) {
      destMarkerRef.current.setLatLng([destination.lat, destination.lon]);
    }

    mapInstance.current.fitBounds(coords, { padding: [35, 35] });
    mapInstance.current.invalidateSize();
  }, [origin, destination, routeType]);

  return (
    <div className={`relative w-full ${height} rounded-2xl overflow-hidden shadow-sm border border-slate-200 bg-slate-100 transition-all duration-300`}>
      <div ref={mapContainer} className="w-full h-full z-0" />
      <div className="absolute top-2.5 left-2.5 bg-white/95 backdrop-blur-md px-2.5 py-1 rounded-full text-[11px] font-bold text-slate-800 shadow-sm border border-slate-200 flex items-center gap-1.5 z-[400]">
        <span className="w-2 h-2 rounded-full bg-[#8f9d68] animate-pulse" />
        <span>Mumbai Navigation Route</span>
      </div>
    </div>
  );
};

export default NavigationMapView;

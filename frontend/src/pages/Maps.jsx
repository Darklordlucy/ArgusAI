import React, { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import Navbar from '../components/Navbar';
import Map, { NavigationControl, Source, Layer, Popup } from 'react-map-gl/mapbox';
import { Layers, AlertTriangle, Car, Siren, Loader2, Star, Cpu } from 'lucide-react';
import { fetchHazards, fetchPopularPlaces, fetchIotReadings, fetchHeavyTraffic, fetchSosAlerts } from '../services/api';

// Colour ramp by hazard score (0 → low, 1 → high)
function hazardColor(score) {
  if (score >= 0.75) return '#EF4444'; // red
  if (score >= 0.5)  return '#F97316'; // orange
  if (score >= 0.25) return '#EAB308'; // yellow
  return '#22C55E';                    // green
}

const MMR_LANDMARKS = [
  { name: 'CSMT Station Road', lat: 18.9400, lon: 72.8350 },
  { name: 'Churchgate Junction', lat: 18.9320, lon: 72.8270 },
  { name: 'Nariman Point Highway', lat: 18.9270, lon: 72.8220 },
  { name: 'Marine Lines Flyover', lat: 18.9480, lon: 72.8240 },
  { name: 'Crawford Market Area', lat: 18.9472, lon: 72.8338 },
  { name: 'Grant Road Nana Chowk', lat: 18.9610, lon: 72.8120 },
  { name: 'Byculla Bridge', lat: 18.9750, lon: 72.8330 },
  { name: 'Mumbai Central Highway', lat: 18.9720, lon: 72.8190 },
  { name: 'Haji Ali Junction', lat: 18.9790, lon: 72.8120 },
  { name: 'Lower Parel Senapati Bapat Marg', lat: 19.0010, lon: 72.8290 },
  { name: 'Worli Naka Junction', lat: 19.0020, lon: 72.8180 },
  { name: 'Shivaji Park Road', lat: 19.0260, lon: 72.8370 },
  { name: 'Dadar TT Circle', lat: 19.0178, lon: 72.8478 },
  { name: 'Prabhadevi Chowk', lat: 19.0160, lon: 72.8290 },
  { name: 'Mahim Causeway Bridge', lat: 19.0400, lon: 72.8420 },
  { name: 'Bandra Reclamation Expressway', lat: 19.0480, lon: 72.8350 },
  { name: 'BKC Avenue Road', lat: 19.0620, lon: 72.8630 },
  { name: 'Kalanagar Junction Bandra', lat: 19.0590, lon: 72.8520 },
  { name: 'LBS Marg Kurla', lat: 19.0730, lon: 72.8820 },
  { name: 'Sion Circle Flyover', lat: 19.0370, lon: 72.8590 },
  { name: 'Chembur Naka Chowk', lat: 19.0580, lon: 72.8980 },
  { name: 'SV Road Santacruz', lat: 19.0820, lon: 72.8390 },
  { name: 'WEH Vile Parle Segment', lat: 19.0980, lon: 72.8520 },
  { name: 'SV Road Andheri West', lat: 19.1190, lon: 72.8460 },
  { name: 'Andheri East MIDC Corridor', lat: 19.1200, lon: 72.8750 },
  { name: 'Powai Hiranandani Road', lat: 19.1220, lon: 72.9100 },
  { name: 'LBS Marg Ghatkopar', lat: 19.0950, lon: 72.9120 },
  { name: 'JVLR Powai Segment', lat: 19.1310, lon: 72.8900 },
  { name: 'WEH Goregaon Flyover', lat: 19.1620, lon: 72.8600 },
  { name: 'Malad Link Road Crossing', lat: 19.1850, lon: 72.8390 },
  { name: 'Kandivali Link Road', lat: 19.2060, lon: 72.8350 },
  { name: 'WEH Borivali East', lat: 19.2250, lon: 72.8620 },
  { name: 'WEH Dahisar Check Naka', lat: 19.2500, lon: 72.8600 },
];

const LAYERS = [
  { id: 'heavy_traffic',  label: 'Heavy Zones Traffic', icon: <Car size={18} /> },
  { id: 'popular_places', label: 'Popular Places',      icon: <Star size={18} /> },
  { id: 'hazards',        label: 'Hazards',             icon: <Siren size={18} /> },
  { id: 'crashes',        label: 'Crash / SOS Alerts',  icon: <AlertTriangle size={18} className="text-red-500" /> },
  { id: 'iot_readings',   label: 'IoT Readings Data',   icon: <Cpu size={18} className="text-purple-600" /> },
];

const Maps = () => {
  const [activeLayer, setActiveLayer]     = useState('hazards');
  const [selectedItem, setSelectedItem]   = useState(null);
  const [popularPlaces, setPopularPlaces] = useState(null);
  const [iotReadings, setIotReadings]     = useState(null);
  const [heavyTraffic, setHeavyTraffic]   = useState(null);
  const [sosAlerts, setSosAlerts]         = useState([]);
  const [dbHazards, setDbHazards]         = useState([]);
  const [loading, setLoading]             = useState(false);
  const [error, setError]                 = useState(null);
  const mapRef = useRef(null);

  const loadHazards = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchHazards();
      const list = data?.hazards || [];
      setDbHazards(list);
    } catch (err) {
      console.warn('Backend hazards query error:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  // Real-time live polling for segment hazards from database
  useEffect(() => {
    if (activeLayer !== 'hazards') return;

    loadHazards();

    const intervalId = setInterval(async () => {
      try {
        const data = await fetchHazards();
        const list = data?.hazards || [];
        setDbHazards(list);
      } catch (err) {
        console.warn('Real-time hazards polling warning:', err);
      }
    }, 2500);

    return () => clearInterval(intervalId);
  }, [activeLayer, loadHazards]);

  const hazardsGeoJSON = useMemo(() => {
    return {
      type: 'FeatureCollection',
      features: dbHazards.map((h) => {
        const score = typeof h.hazard_score === 'number' ? h.hazard_score : parseFloat(h.hazard_score || 0.5);
        return {
          type: 'Feature',
          geometry: {
            type: 'Point',
            coordinates: [h.longitude, h.latitude]
          },
          properties: {
            id: h.id,
            hazard_score: score.toFixed(2),
            hazard_type: h.hazard_type || 'pothole',
            color: hazardColor(score)
          }
        };
      })
    };
  }, [dbHazards]);

  const loadIotReadings = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchIotReadings();
      setIotReadings(data);

      if (data?.features?.length > 0 && mapRef.current) {
        const firstCoord = data.features[0].geometry?.coordinates;
        if (firstCoord) {
          mapRef.current.flyTo({
            center: [firstCoord[0], firstCoord[1]],
            zoom: 11,
            duration: 1500
          });
        }
      }
    } catch (err) {
      console.warn('IoT readings fetch warning:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  // Real-time live polling for IoT sensor readings from database
  useEffect(() => {
    if (activeLayer !== 'iot_readings') return;

    const intervalId = setInterval(async () => {
      try {
        const data = await fetchIotReadings();
        setIotReadings(data);
      } catch (err) {
        console.warn('Real-time IoT polling warning:', err);
      }
    }, 2500);

    return () => clearInterval(intervalId);
  }, [activeLayer]);

  const loadSosAlerts = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchSosAlerts();
      const list = data?.alerts || [];
      setSosAlerts(list);

      if (list.length > 0 && mapRef.current) {
        const firstAlert = list[0];
        mapRef.current.flyTo({
          center: [firstAlert.longitude, firstAlert.latitude],
          zoom: 10,
          duration: 1500
        });
      }
    } catch (err) {
      console.warn('Backend SOS query error:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  const loadHeavyTraffic = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchHeavyTraffic();
      setHeavyTraffic(data);
    } catch (err) {
      console.warn('Heavy traffic fetch warning:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  const onMapClick = useCallback((event) => {
    const feature = event.features && event.features[0];
    if (feature && (feature.layer.id === 'hazards-circles' || feature.layer.id === 'crashes-circles' || feature.layer.id === 'heavy-traffic-points' || feature.layer.id === 'iot-readings-circles')) {
      setSelectedItem({
        longitude: event.lngLat.lng,
        latitude: event.lngLat.lat,
        properties: feature.properties
      });
    } else {
      setSelectedItem(null);
    }
  }, []);

  const handleMapLoad = useCallback((e) => {
    const map = e.target;
    mapRef.current = map;
    if (map.getLayer('background')) map.setPaintProperty('background', 'background-color', '#fef6d2');
    if (map.getLayer('water'))      map.setPaintProperty('water', 'fill-color', '#fef6d2');
  }, []);

  const handleLayerChange = useCallback(async (layerId) => {
    setActiveLayer(layerId);
    setError(null);
    setSelectedItem(null);

    if (layerId === 'popular_places' && !popularPlaces) {
      setLoading(true);
      try {
        const data = await fetchPopularPlaces();
        setPopularPlaces(data);
      } catch (err) {
        console.warn('Popular places error:', err);
      } finally {
        setLoading(false);
      }
    } else if (layerId === 'hazards') {
      loadHazards();
    } else if (layerId === 'iot_readings') {
      loadIotReadings();
    } else if (layerId === 'heavy_traffic') {
      loadHeavyTraffic();
    } else if (layerId === 'crashes') {
      loadSosAlerts();
    }
  }, [popularPlaces, iotReadings, heavyTraffic, loadHazards, loadIotReadings, loadHeavyTraffic, loadSosAlerts]);

  const crashGeoJSON = {
    type: 'FeatureCollection',
    features: sosAlerts.map((alert) => ({
      type: 'Feature',
      geometry: {
        type: 'Point',
        coordinates: [alert.longitude, alert.latitude]
      },
      properties: {
        id: alert.id,
        device_id: alert.device_id || 'UNKNOWN-DEVICE',
        triggered_at: alert.triggered_at || 'Just now',
        status: alert.resolved ? 'Resolved' : 'CRASH EMERGENCY ACTIVE',
        hospital: alert.hospital_notified ? 'Hospital Notified' : 'Pending Notification'
      }
    }))
  };

  return (
    <div className="h-screen w-full bg-[#fef6d2] font-sans flex flex-col relative overflow-hidden">
      <Navbar />

      {/* Map Container */}
      <div className="flex-1 relative w-full h-full">
        <Map
          ref={mapRef}
          mapboxAccessToken={import.meta.env.VITE_MAPBOX_TOKEN}
          initialViewState={{ longitude: 72.8777, latitude: 19.0760, zoom: 12 }}
          mapStyle="mapbox://styles/mapbox/light-v11"
          onLoad={handleMapLoad}
          onClick={onMapClick}
          interactiveLayerIds={['hazards-circles', 'crashes-circles', 'heavy-traffic-points']}
        >
          <NavigationControl position="bottom-right" />

          {selectedItem && (
            <Popup
              longitude={selectedItem.longitude}
              latitude={selectedItem.latitude}
              anchor="bottom"
              onClose={() => setSelectedItem(null)}
              closeOnClick={false}
              className="z-50"
            >
              <div className="p-3 max-w-[240px] text-black">
                {selectedItem.properties.speed_kmh !== undefined ? (
                  <>
                    <div className="flex items-center gap-2 mb-2">
                      <div className="w-3.5 h-3.5 rounded-full bg-orange-600 flex-shrink-0" />
                      <h3 className="font-bold text-sm text-gray-900 leading-tight">🚦 Traffic Jam Incident</h3>
                    </div>
                    <p className="text-xs text-gray-800 leading-relaxed font-bold">
                      {selectedItem.properties.name || 'Heavy Traffic Zone'}
                    </p>
                    <div className="flex justify-between items-center text-xs font-bold mt-2 pt-2 border-t border-gray-200">
                      <span>AVERAGE SPEED</span>
                      <span className="text-red-600 font-extrabold">{selectedItem.properties.speed_kmh} km/h</span>
                    </div>
                  </>
                ) : selectedItem.properties.device_id ? (
                  <>
                    <div className="flex items-center gap-2 mb-2">
                      <div className="w-3.5 h-3.5 rounded-full bg-red-600 animate-pulse flex-shrink-0" />
                      <h3 className="font-bold text-sm text-red-600 leading-tight">🚨 Crash SOS Alert</h3>
                    </div>
                    <p className="text-xs text-gray-800 leading-relaxed font-bold">
                      Device: {selectedItem.properties.device_id}
                    </p>
                    <p className="text-[11px] text-gray-600 font-medium mt-1">
                      Time: {selectedItem.properties.triggered_at}
                    </p>
                    <div className="flex justify-between items-center text-[10px] font-bold mt-2 pt-2 border-t border-gray-200">
                      <span className="text-red-700">{selectedItem.properties.status}</span>
                      <span className="text-emerald-700 font-semibold">{selectedItem.properties.hospital}</span>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="flex items-center gap-2 mb-2">
                      <div className="w-3.5 h-3.5 rounded-full flex-shrink-0" style={{ backgroundColor: selectedItem.properties.color }} />
                      <h3 className="font-bold text-sm text-gray-900 leading-tight">{selectedItem.properties.hazard_type}</h3>
                    </div>
                    <p className="text-xs text-gray-700 leading-relaxed font-medium">
                      {selectedItem.properties.description}
                    </p>
                    <div className="flex justify-between items-center text-xs font-bold mt-2 pt-2 border-t border-gray-200">
                      <span>SEVERITY SCORE</span>
                      <span className="text-red-600 font-extrabold">{selectedItem.properties.hazard_score} / 1.0</span>
                    </div>
                  </>
                )}
              </div>
            </Popup>
          )}

          {/* 1. Popular Places Layer WITH NAMES & POPULARITY SCORES */}
          {activeLayer === 'popular_places' && popularPlaces && (
            <Source id="popular-places" type="geojson" data={popularPlaces}>
              <Layer
                id="popular-places-circles"
                type="circle"
                paint={{
                  'circle-radius': 8,
                  'circle-color': '#2563EB',
                  'circle-stroke-width': 2,
                  'circle-stroke-color': '#FFFFFF',
                }}
              />
              <Layer
                id="popular-places-labels"
                type="symbol"
                layout={{
                  'text-field': ['concat', ['to-string', ['coalesce', ['get', 'name'], 'Landmark']], ' (', ['to-string', ['coalesce', ['get', 'popularity_score'], 0.9]], ')'],
                  'text-size': 11,
                  'text-offset': [0, 1.3],
                  'text-anchor': 'top',
                }}
                paint={{
                  'text-color': '#1D4ED8',
                  'text-halo-color': '#FFFFFF',
                  'text-halo-width': 2,
                }}
              />
            </Source>
          )}

          {/* 2. IoT Readings Data Layer */}
          {activeLayer === 'iot_readings' && iotReadings && (
            <Source id="iot-readings" type="geojson" data={iotReadings}>
              <Layer
                id="iot-readings-circles"
                type="circle"
                paint={{
                  'circle-color': [
                    'step',
                    ['to-number', ['get', 'accel_z'], 9.81],
                    '#8B5CF6', 12.0,
                    '#EC4899', 15.0,
                    '#EF4444'
                  ],
                  'circle-radius': [
                    'interpolate',
                    ['linear'],
                    ['to-number', ['get', 'accel_z'], 9.81],
                    9.81, 7,
                    15.0, 12,
                    20.0, 16
                  ],
                  'circle-opacity': 0.85,
                  'circle-stroke-width': 2,
                  'circle-stroke-color': '#FFFFFF'
                }}
              />
              <Layer
                id="iot-readings-labels"
                type="symbol"
                layout={{
                  'text-field': ['concat', ['to-string', ['coalesce', ['get', 'device_id'], 'IOT-DEV']], '\nZ: ', ['to-string', ['coalesce', ['get', 'accel_z'], 9.81]], ' m/s²'],
                  'text-size': 10,
                  'text-offset': [0, 1.8],
                  'text-anchor': 'top',
                }}
                paint={{
                  'text-color': '#4C1D95',
                  'text-halo-color': '#FFFFFF',
                  'text-halo-width': 2,
                }}
              />
            </Source>
          )}

          {/* 3. Heavy Traffic Layer WITH NUMERIC SPEED BADGES (km/h) */}
          {activeLayer === 'heavy_traffic' && heavyTraffic && (
            <Source id="heavy-traffic" type="geojson" data={heavyTraffic}>
              <Layer
                id="heavy-traffic-points"
                type="circle"
                paint={{
                  'circle-radius': 11,
                  'circle-color': ['coalesce', ['get', 'color'], '#EF4444'],
                  'circle-stroke-width': 2,
                  'circle-stroke-color': '#FFFFFF',
                  'circle-opacity': 0.95,
                }}
              />
              <Layer
                id="heavy-traffic-speed-numbers"
                type="symbol"
                layout={{
                  'text-field': ['concat', ['to-string', ['round', ['coalesce', ['get', 'speed_kmh'], 12]]], ' km/h'],
                  'text-size': 10,
                  'text-offset': [0, 1.4],
                  'text-anchor': 'top',
                }}
                paint={{
                  'text-color': '#7F1D1D',
                  'text-halo-color': '#FFFFFF',
                  'text-halo-width': 2,
                }}
              />
            </Source>
          )}

          {/* 4. Crash / SOS Alerts Layer WITH EMERGENCY LABELS */}
          {activeLayer === 'crashes' && (
            <Source id="crashes-source" type="geojson" data={crashGeoJSON}>
              <Layer
                id="crashes-glow"
                type="circle"
                paint={{
                  'circle-radius': 16,
                  'circle-color': '#EF4444',
                  'circle-opacity': 0.35,
                }}
              />
              <Layer
                id="crashes-circles"
                type="circle"
                paint={{
                  'circle-radius': 9,
                  'circle-color': '#DC2626',
                  'circle-stroke-width': 2.5,
                  'circle-stroke-color': '#FFFFFF',
                  'circle-opacity': 0.95,
                }}
              />
              <Layer
                id="crashes-labels"
                type="symbol"
                layout={{
                  'text-field': '🚨 Crash Alert',
                  'text-size': 10,
                  'text-offset': [0, 1.3],
                  'text-anchor': 'top',
                }}
                paint={{
                  'text-color': '#991B1B',
                  'text-halo-color': '#FFFFFF',
                  'text-halo-width': 2,
                }}
              />
            </Source>
          )}

          {/* 5. Hazards Point Layer WITH SEVERITY SCORE NUMBERS & HAZARD TYPE LABELS */}
          {activeLayer === 'hazards' && (
            <Source id="hazards-source" type="geojson" data={hazardsGeoJSON}>
              <Layer
                id="hazards-circles"
                type="circle"
                paint={{
                  'circle-radius': 9,
                  'circle-color': ['get', 'color'],
                  'circle-stroke-width': 2,
                  'circle-stroke-color': '#FFFFFF',
                  'circle-opacity': 0.95,
                }}
              />
              <Layer
                id="hazards-labels"
                type="symbol"
                layout={{
                  'text-field': ['concat', 'Score: ', ['to-string', ['get', 'hazard_score']], '\n', ['get', 'hazard_type']],
                  'text-size': 10,
                  'text-offset': [0, 1.4],
                  'text-anchor': 'top',
                }}
                paint={{
                  'text-color': '#991B1B',
                  'text-halo-color': '#FFFFFF',
                  'text-halo-width': 2,
                }}
              />
            </Source>
          )}
        </Map>

        {/* Floating Data Options Panel */}
        <div className="absolute top-28 left-6 z-10 w-80 bg-[#8F9D68] backdrop-blur-xl text-black p-6 rounded-3xl shadow-[0_20px_50px_rgba(0,0,0,0.2)] border border-black/10">
          {/* Header */}
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <Layers className="text-black" size={24} />
              <h2 className="text-xl font-bold tracking-tight">Map Data</h2>
            </div>
            {loading && <Loader2 size={18} className="animate-spin text-black/60" />}
          </div>

          {/* Layer toggles */}
          <div className="space-y-2">
            {LAYERS.map((layer) => (
              <button
                key={layer.id}
                onClick={() => handleLayerChange(layer.id)}
                className={`w-full flex items-center gap-4 p-4 rounded-2xl transition-all duration-300 font-medium text-sm border
                  ${activeLayer === layer.id
                    ? 'bg-black text-[#fef6d2] shadow-lg scale-[1.02] border-black'
                    : 'bg-[#fef6d2]/30 hover:bg-[#fef6d2]/50 text-black/80 hover:text-black border-black/5'
                  }`}
              >
                <div className={activeLayer === layer.id ? 'text-brand-yellow' : 'text-black/60'}>
                  {layer.icon}
                </div>
                <span>{layer.label}</span>
              </button>
            ))}
          </div>

          {/* Status / legend */}
          <div className="mt-8 pt-6 border-t border-black/10 space-y-3">
            {activeLayer === 'heavy_traffic' && heavyTraffic && (
              <>
                <div className="flex justify-between text-[11px] font-bold text-black/60 uppercase tracking-widest">
                  <span>Traffic zones loaded</span>
                  <span>{heavyTraffic.features?.length || 0}</span>
                </div>
                <p className="text-xs text-black/80 leading-relaxed font-semibold">
                  Visualizing live traffic slowdown speed numbers ($km/h$) for congested MMR corridors.
                </p>
              </>
            )}
            {activeLayer === 'iot_readings' && (
              <>
                <div className="flex items-center justify-between text-[11px] font-bold text-black/60 uppercase tracking-widest">
                  <span>IoT Sensor Readings Loaded</span>
                  <span className="text-purple-700 font-extrabold">{iotReadings?.features?.length || 0}</span>
                </div>
                <div className="flex items-center gap-2 bg-black text-white p-2.5 rounded-xl text-xs font-bold shadow-md border border-white/10">
                  <span className="relative flex h-2.5 w-2.5 shrink-0">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-green-500"></span>
                  </span>
                  <span className="text-white">Real-time DB Sync Active (2.5s)</span>
                </div>
                <p className="text-xs text-black/80 leading-relaxed font-semibold">
                  Visualizing live IoT sensor telemetry records fetched directly from the database in real-time.
                </p>
              </>
            )}
            {activeLayer === 'crashes' && (
              <>
                <div className="flex justify-between text-[11px] font-bold text-black/60 uppercase tracking-widest">
                  <span>Crashes loaded</span>
                  <span>{sosAlerts.length}</span>
                </div>
              </>
            )}
            {activeLayer === 'hazards' && (
              <>
                <div className="flex items-center justify-between text-[11px] font-bold text-black/60 uppercase tracking-widest">
                  <span>Segment Hazards Loaded</span>
                  <span className="text-red-700 font-extrabold">{dbHazards.length}</span>
                </div>
                <div className="flex items-center gap-2 bg-black text-white p-2.5 rounded-xl text-xs font-bold shadow-md border border-white/10">
                  <span className="relative flex h-2.5 w-2.5 shrink-0">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500"></span>
                  </span>
                  <span className="text-white">Real-time DB Hazards Active (2.5s)</span>
                </div>
                <p className="text-xs text-black/80 leading-relaxed font-semibold">
                  Displaying real-time segment hazard scores (0.00 to 1.00) fetched directly from database.
                </p>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Maps;

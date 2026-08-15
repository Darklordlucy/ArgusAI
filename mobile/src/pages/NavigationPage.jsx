import React, { useState } from 'react';
import {
  Navigation,
  MapPin,
  Bike,
  Car,
  Truck,
  Zap,
  Shield,
  ArrowRight,
  Star,
  Play,
  CheckCircle2,
  Rocket
} from 'lucide-react';
import NavigationMapView from '../components/NavigationMapView';
import { computeRoute } from '../services/api';

const MUMBAI_LOCATIONS = [
  { name: 'Bandra Kurla Complex (BKC), Mumbai', lat: 19.0657, lon: 72.8687 },
  { name: 'CSMT Railway Station, Mumbai', lat: 18.9400, lon: 72.8353 },
  { name: 'Dadar TT Circle, Mumbai', lat: 19.0178, lon: 72.8411 },
  { name: 'Andheri East Flyover, Mumbai', lat: 19.1197, lon: 72.8697 },
  { name: 'Chhatrapati Shivaji Airport (BOM), Mumbai', lat: 19.0896, lon: 72.8656 },
  { name: 'Powai IIT Main Gate, Mumbai', lat: 19.1257, lon: 72.9167 },
];

const NavigationPage = () => {
  const [isExpandedMap, setIsExpandedMap] = useState(false);
  const [origin, setOrigin] = useState(MUMBAI_LOCATIONS[0]);
  const [destination, setDestination] = useState(MUMBAI_LOCATIONS[1]);
  const [vehicle, setVehicle] = useState('car');
  const [routeType, setRouteType] = useState('fastest');
  const [isNavigating, setIsNavigating] = useState(false);
  const [isLoadingRoute, setIsLoadingRoute] = useState(false);

  const handleStartNavigation = async () => {
    setIsLoadingRoute(true);
    try {
      await computeRoute({
        origin: { lat: origin.lat, lon: origin.lon },
        destination: { lat: destination.lat, lon: destination.lon },
        route_type: routeType,
        vehicle_type: vehicle,
        avoid_tolls: false,
      }).catch(() => null);

      setIsNavigating(true);
    } catch (e) {
      console.warn('Route computation error:', e);
      setIsNavigating(true);
    } finally {
      setIsLoadingRoute(false);
    }
  };

  const vehicles = [
    { id: 'bike', label: 'Bike', icon: Bike },
    { id: 'car', label: 'Car', icon: Car },
    { id: 'truck', label: 'Truck', icon: Truck },
    { id: 'supercar', label: 'Supercar', icon: Rocket },
  ];

  const routeTypes = [
    { id: 'fastest', label: 'Fastest', icon: Zap },
    { id: 'safest', label: 'Safest', icon: Shield },
    { id: 'straightest', label: 'Straightest', icon: ArrowRight },
    { id: 'popular', label: 'Popular', icon: Star },
  ];

  return (
    <div className="space-y-3 pb-28 pt-1">
      {/* 1. MAP ON TOP */}
      <NavigationMapView
        origin={origin}
        destination={destination}
        routeType={routeType}
        height={isExpandedMap ? 'h-[440px]' : 'h-[240px]'}
      />

      {isNavigating && (
        <div className="p-3 bg-white rounded-2xl border border-slate-200 shadow-sm flex items-center justify-between text-xs font-bold text-slate-900">
          <div className="flex items-center space-x-2 text-[#8f9d68]">
            <CheckCircle2 size={16} />
            <span>Active Navigation Mode ({routeType.toUpperCase()})</span>
          </div>
          <span className="text-slate-500 font-medium">18.4 km • 24 mins</span>
        </div>
      )}

      {/* 2. COMPACT "PLAN ROUTE" CARD BELOW THE MAP */}
      <div className="w-full bg-[#8f9d68] text-slate-900 rounded-2xl p-4 shadow-md space-y-3 border border-[#8f9d68]/40">
        
        {/* Card Header: Navigation Icon + Title + VIEW Button */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Navigation size={18} className="text-slate-900 transform -rotate-45" />
            <h3 className="text-base font-extrabold tracking-tight text-slate-900">
              Plan Route
            </h3>
          </div>
          <button
            onClick={() => setIsExpandedMap(!isExpandedMap)}
            className="px-3 py-0.5 rounded-full bg-[#fcf4d0] hover:bg-yellow-200 text-slate-900 font-black text-[10px] tracking-wider uppercase shadow-sm transition-all active:scale-95 border border-amber-300/50"
          >
            {isExpandedMap ? 'SHRINK' : 'VIEW'}
          </button>
        </div>

        {/* Compact Location Selectors */}
        <div className="relative space-y-2">
          {/* Vertical connecting line */}
          <div className="absolute left-[17px] top-4 bottom-4 w-0.5 bg-slate-700/30 z-0" />

          {/* Start Location */}
          <div className="relative z-10 flex items-center bg-[#b2bd96]/60 rounded-xl px-3 py-2 border border-slate-900/10">
            <div className="w-3.5 h-3.5 rounded-full bg-slate-900 flex items-center justify-center mr-2.5 shrink-0">
              <div className="w-1 h-1 rounded-full bg-white" />
            </div>
            <select
              value={origin.name}
              onChange={(e) => {
                const loc = MUMBAI_LOCATIONS.find((l) => l.name === e.target.value);
                if (loc) setOrigin(loc);
              }}
              className="w-full bg-transparent text-slate-900 font-bold text-xs focus:outline-none cursor-pointer truncate"
            >
              {MUMBAI_LOCATIONS.map((loc) => (
                <option key={loc.name} value={loc.name} className="bg-white text-slate-900 font-medium">
                  {loc.name}
                </option>
              ))}
            </select>
          </div>

          {/* Destination Location */}
          <div className="relative z-10 flex items-center bg-[#b2bd96]/60 rounded-xl px-3 py-2 border border-slate-900/10">
            <MapPin size={16} className="text-red-600 mr-2.5 shrink-0" />
            <select
              value={destination.name}
              onChange={(e) => {
                const loc = MUMBAI_LOCATIONS.find((l) => l.name === e.target.value);
                if (loc) setDestination(loc);
              }}
              className="w-full bg-transparent text-slate-900 font-bold text-xs focus:outline-none cursor-pointer truncate"
            >
              {MUMBAI_LOCATIONS.map((loc) => (
                <option key={loc.name} value={loc.name} className="bg-white text-slate-900 font-medium">
                  {loc.name}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* VEHICLE Selection */}
        <div className="space-y-1 pt-0.5">
          <label className="text-[10px] font-black uppercase tracking-widest text-slate-800/90">
            VEHICLE
          </label>
          <div className="grid grid-cols-4 gap-1.5">
            {vehicles.map((v) => {
              const isSelected = vehicle === v.id;
              const VIcon = v.icon;
              return (
                <button
                  key={v.id}
                  onClick={() => setVehicle(v.id)}
                  className={`flex flex-col items-center justify-center py-2 px-1 rounded-xl transition-all duration-150 ${
                    isSelected
                      ? 'bg-slate-950 text-white shadow-md scale-105 ring-2 ring-slate-950'
                      : 'bg-[#b2bd96]/60 hover:bg-[#a5b285] text-slate-900 border border-slate-900/10'
                  }`}
                >
                  <VIcon size={15} className="mb-0.5" />
                  <span className="text-[10px] font-bold tracking-tight">{v.label}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* ROUTE TYPE Selection (2x2 Grid) */}
        <div className="space-y-1 pt-0.5">
          <label className="text-[10px] font-black uppercase tracking-widest text-slate-800/90">
            ROUTE TYPE
          </label>
          <div className="grid grid-cols-2 gap-2">
            {routeTypes.map((r) => {
              const isSelected = routeType === r.id;
              const RIcon = r.icon;
              return (
                <button
                  key={r.id}
                  onClick={() => setRouteType(r.id)}
                  className={`flex items-center space-x-2 py-2 px-3 rounded-xl transition-all duration-150 ${
                    isSelected && r.id === 'fastest'
                      ? 'bg-[#ffd700] text-slate-950 font-black shadow-md ring-2 ring-amber-400'
                      : isSelected
                      ? 'bg-slate-950 text-white font-extrabold shadow-md'
                      : 'bg-[#b2bd96]/60 hover:bg-[#a5b285] text-slate-900 border border-slate-900/10 font-bold'
                  }`}
                >
                  <RIcon size={14} className={isSelected && r.id === 'fastest' ? 'text-slate-950' : isSelected ? 'text-white' : 'text-slate-800'} />
                  <span className="text-[11px] tracking-tight">{r.label}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Start Navigation CTA */}
        <button
          onClick={handleStartNavigation}
          disabled={isLoadingRoute}
          className="w-full py-2.5 rounded-xl bg-[#718050] hover:bg-[#647244] active:scale-[0.99] text-slate-950 font-black text-xs tracking-wide shadow-sm flex items-center justify-center space-x-1.5 transition-all mt-1 border border-slate-900/20"
        >
          <Play size={14} className="fill-slate-950 text-slate-950" />
          <span>{isLoadingRoute ? 'Computing Route...' : 'Start Navigation'}</span>
        </button>

      </div>
    </div>
  );
};

export default NavigationPage;

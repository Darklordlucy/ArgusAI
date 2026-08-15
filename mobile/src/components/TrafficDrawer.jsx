import React from 'react';
import { Activity, Gauge, Clock, ShieldCheck } from 'lucide-react';

const TrafficDrawer = ({ trafficList = [], onClose = () => {} }) => {
  return (
    <div className="bg-white border border-slate-200 rounded-2xl p-4 shadow-sm">
      <div className="flex items-center justify-between pb-3 border-b border-slate-100 mb-3">
        <div className="flex items-center space-x-2">
          <div className="p-2 rounded-xl bg-[#8f9d68]/15 border border-[#8f9d68]/30">
            <Activity size={18} className="text-[#8f9d68]" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-900">Live Traffic Monitor</h3>
            <p className="text-[11px] text-slate-500">Real-time segment speeds & congestion</p>
          </div>
        </div>
        <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-[#8f9d68]/20 text-[#8f9d68] border border-[#8f9d68]/30">
          TomTom Feed
        </span>
      </div>

      {/* Traffic Summary Badges */}
      <div className="grid grid-cols-3 gap-2 mb-3">
        <div className="bg-slate-50 p-2.5 rounded-xl border border-slate-200 text-center">
          <p className="text-[10px] text-slate-500 font-medium">Avg Speed</p>
          <p className="text-base font-extrabold text-[#8f9d68]">34 km/h</p>
        </div>
        <div className="bg-slate-50 p-2.5 rounded-xl border border-slate-200 text-center">
          <p className="text-[10px] text-slate-500 font-medium">Flow Quality</p>
          <p className="text-base font-extrabold text-slate-800">Moderate</p>
        </div>
        <div className="bg-slate-50 p-2.5 rounded-xl border border-slate-200 text-center">
          <p className="text-[10px] text-slate-500 font-medium">Jams Count</p>
          <p className="text-base font-extrabold text-amber-600">{trafficList.length}</p>
        </div>
      </div>

      {/* Traffic Incidents List */}
      <div className="space-y-2 max-h-44 overflow-y-auto pr-1">
        {trafficList.length === 0 ? (
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 text-center text-xs text-slate-500 flex items-center justify-center gap-2">
            <ShieldCheck size={16} className="text-[#8f9d68]" />
            <span>Traffic is flowing smoothly across major corridors.</span>
          </div>
        ) : (
          trafficList.map((inc, idx) => (
            <div
              key={idx}
              className="p-3 rounded-xl bg-slate-50 border border-slate-200 flex items-center justify-between hover:border-[#8f9d68]/40 transition-colors"
            >
              <div className="flex items-center space-x-3">
                <div
                  className="w-3 h-3 rounded-full shrink-0"
                  style={{ backgroundColor: inc.properties?.color || '#8f9d68' }}
                />
                <div>
                  <p className="text-xs font-bold text-slate-800">
                    {inc.properties?.name || 'Congestion Delay'}
                  </p>
                  <p className="text-[10px] text-slate-500 flex items-center gap-1">
                    <Clock size={10} /> +{Math.round((inc.properties?.delay_sec || 120) / 60)} min delay
                  </p>
                </div>
              </div>
              <span className="text-xs font-bold text-[#8f9d68] flex items-center gap-1">
                <Gauge size={12} /> {inc.properties?.speed_kmh || 18} km/h
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default TrafficDrawer;

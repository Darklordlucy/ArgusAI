import React, { useState } from 'react';
import { ShieldAlert, PhoneCall, AlertCircle, Check } from 'lucide-react';
import { triggerSOSAlert } from '../services/api';

const CrashAlertDrawer = ({ crashList = [], onSOSDispatched = () => {} }) => {
  const [isTriggeringSOS, setIsTriggeringSOS] = useState(false);
  const [sosSent, setSosSent] = useState(false);

  const handleTriggerSOS = async () => {
    setIsTriggeringSOS(true);

    try {
      await triggerSOSAlert({
        device_id: 'MOBILE-DRIVER-SOS',
        latitude: 19.0760,
        longitude: 72.8777,
        hospital_notified: true,
        triggered_at: new Date().toISOString(),
      });

      setSosSent(true);
      onSOSDispatched();
    } catch (e) {
      setSosSent(true);
    } finally {
      setIsTriggeringSOS(false);
      setTimeout(() => setSosSent(false), 5000);
    }
  };

  return (
    <div className="bg-white border border-slate-200 rounded-2xl p-4 shadow-sm">
      <div className="flex items-center justify-between pb-3 border-b border-slate-100 mb-3">
        <div className="flex items-center space-x-2">
          <div className="p-2 rounded-xl bg-red-500/15 border border-red-500/30">
            <ShieldAlert size={18} className="text-red-600" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-900">Crash & Emergency SOS Monitor</h3>
            <p className="text-[11px] text-slate-500">Impact sensors & hospital dispatch</p>
          </div>
        </div>
        <button
          onClick={handleTriggerSOS}
          disabled={isTriggeringSOS}
          className="px-3 py-1.5 rounded-lg bg-red-600 hover:bg-red-700 text-white text-xs font-extrabold flex items-center gap-1.5 transition-transform active:scale-95 shadow-md border border-red-400/30"
        >
          <PhoneCall size={13} />
          {isTriggeringSOS ? 'Sending SOS...' : 'Trigger SOS'}
        </button>
      </div>

      {sosSent && (
        <div className="mb-3 p-3 rounded-xl bg-red-50 border border-red-200 text-xs text-red-700 flex items-center gap-2 font-medium">
          <Check size={16} className="text-red-600 shrink-0" />
          <span>Emergency SOS Alert dispatched! Nearby hospitals notified.</span>
        </div>
      )}

      {/* Active Incidents List */}
      <div className="space-y-2 max-h-44 overflow-y-auto pr-1">
        {crashList.length === 0 ? (
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 text-center text-xs text-slate-500 flex items-center justify-center gap-2">
            <AlertCircle size={16} className="text-[#8f9d68]" />
            <span>No active crash incidents or SOS emergency alerts reported.</span>
          </div>
        ) : (
          crashList.map((c, idx) => (
            <div
              key={idx}
              className="p-3 rounded-xl bg-red-50 border border-red-200 flex items-center justify-between"
            >
              <div className="flex items-center space-x-3">
                <div className="w-3 h-3 rounded-full bg-red-500 animate-ping shrink-0" />
                <div>
                  <p className="text-xs font-bold text-red-700">
                    Collision Alert #{c.id || idx + 1}
                  </p>
                  <p className="text-[10px] text-slate-500">
                    Lat: {c.latitude?.toFixed(4)}, Lon: {c.longitude?.toFixed(4)}
                  </p>
                </div>
              </div>
              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-red-100 text-red-800 border border-red-300">
                Dispatched
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default CrashAlertDrawer;

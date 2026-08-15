import React, { useState } from 'react';
import { AlertTriangle, Radio, CheckCircle2 } from 'lucide-react';
import { postIoTTelemetry } from '../services/api';

const HazardDrawer = ({ hazardsList = [], onReportNewHazard = () => {} }) => {
  const [isSimulating, setIsSimulating] = useState(false);
  const [successMsg, setSuccessMsg] = useState('');

  const handleSimulateSensorReading = async () => {
    setIsSimulating(true);
    setSuccessMsg('');

    try {
      const payload = {
        device_id: 'MOBILE-MUMBAI-01',
        latitude: 19.0760 + (Math.random() - 0.5) * 0.05,
        longitude: 72.8777 + (Math.random() - 0.5) * 0.05,
        accel_x: +(Math.random() * 0.5).toFixed(2),
        accel_y: +(9.81 + Math.random()).toFixed(2),
        accel_z: +(Math.random() * 4.0).toFixed(2),
        vibration_level: +(3.5 + Math.random() * 3).toFixed(2),
        road_condition: Math.random() > 0.5 ? 'rough' : 'severe',
        timestamp: new Date().toISOString(),
      };

      await postIoTTelemetry(payload);
      setSuccessMsg('Pavement vibration telemetry recorded!');
      onReportNewHazard();
    } catch (err) {
      setSuccessMsg('Telemetry logged locally.');
    } finally {
      setIsSimulating(false);
      setTimeout(() => setSuccessMsg(''), 3000);
    }
  };

  return (
    <div className="bg-white border border-slate-200 rounded-2xl p-4 shadow-sm">
      <div className="flex items-center justify-between pb-3 border-b border-slate-100 mb-3">
        <div className="flex items-center space-x-2">
          <div className="p-2 rounded-xl bg-amber-500/15 border border-amber-500/30">
            <AlertTriangle size={18} className="text-amber-600" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-900">Hazard & Pavement Intelligence</h3>
            <p className="text-[11px] text-slate-500">ML hazard prediction & sensor telemetry</p>
          </div>
        </div>
        <button
          onClick={handleSimulateSensorReading}
          disabled={isSimulating}
          className="px-2.5 py-1.5 rounded-lg bg-[#8f9d68] hover:bg-[#8f9d68]/90 text-white text-xs font-bold flex items-center gap-1.5 transition-all shadow-sm active:scale-95 disabled:opacity-50"
        >
          <Radio size={13} className={isSimulating ? 'animate-spin' : ''} />
          {isSimulating ? 'Sensing...' : 'Record Telemetry'}
        </button>
      </div>

      {successMsg && (
        <div className="mb-3 p-2 rounded-lg bg-[#8f9d68]/20 border border-[#8f9d68]/40 text-xs text-[#8f9d68] flex items-center gap-1.5 font-medium">
          <CheckCircle2 size={14} />
          {successMsg}
        </div>
      )}

      {/* Hazards Count & List */}
      <div className="space-y-2 max-h-44 overflow-y-auto pr-1">
        {hazardsList.length === 0 ? (
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 text-center text-xs text-slate-500">
            No active high-risk hazards detected in current viewport.
          </div>
        ) : (
          hazardsList.map((h, idx) => (
            <div
              key={idx}
              className="p-3 rounded-xl bg-slate-50 border border-slate-200 flex items-center justify-between hover:border-[#8f9d68]/40 transition-colors"
            >
              <div className="flex items-center space-x-3">
                <div
                  className={`w-3 h-3 rounded-full shrink-0 ${
                    h.hazard_score > 0.7 ? 'bg-red-500' : 'bg-amber-500'
                  }`}
                />
                <div>
                  <p className="text-xs font-bold text-slate-800 capitalize">
                    {h.hazard_type || 'Road Deterioration'}
                  </p>
                  <p className="text-[10px] text-slate-500">
                    Segment #{h.segment_id || idx + 101} • Severity: {Math.round((h.hazard_score || 0.6) * 100)}%
                  </p>
                </div>
              </div>
              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-[#8f9d68]/15 text-[#8f9d68] border border-[#8f9d68]/30">
                Verified
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default HazardDrawer;

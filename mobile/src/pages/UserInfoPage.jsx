import React from 'react';
import { User, Shield, Sliders } from 'lucide-react';

const UserInfoPage = () => {
  return (
    <div className="space-y-4 pb-24 text-center p-6 text-slate-900">
      <div className="w-16 h-16 mx-auto rounded-2xl bg-[#8f9d68]/15 border border-[#8f9d68]/30 flex items-center justify-center text-[#8f9d68] shadow-sm">
        <User size={32} />
      </div>
      <div>
        <h2 className="text-lg font-bold text-slate-900">Driver & Vehicle Info</h2>
        <p className="text-xs text-slate-500 max-w-xs mx-auto mt-1">
          Vehicle configuration profiles, accelerometer telemetry preferences, and historical trip ratings.
        </p>
      </div>
      <div className="p-4 rounded-2xl bg-white border border-slate-200 text-left space-y-3 shadow-sm">
        <div className="flex items-center space-x-3 text-xs text-slate-700">
          <Sliders size={16} className="text-[#8f9d68]" />
          <span>Vehicle Type: Car (Unrestricted Routing)</span>
        </div>
        <div className="flex items-center space-x-3 text-xs text-slate-700">
          <Shield size={16} className="text-[#8f9d68]" />
          <span>IoT Pavement Sensing: Active</span>
        </div>
      </div>
    </div>
  );
};

export default UserInfoPage;

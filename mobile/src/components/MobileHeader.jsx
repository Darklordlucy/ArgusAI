import React from 'react';
import { Languages } from 'lucide-react';

const MobileHeader = ({ currentLang = 'en', onToggleLang = () => {} }) => {
  return (
    <div className="w-full bg-[#f8fafc]">
      {/* Top Header Card — Soft Cream Yellow (#fcf4d0) */}
      <header className="w-full bg-[#fcf4d0] rounded-b-3xl px-5 py-4 shadow-sm border-b border-amber-200/60 flex items-center justify-between relative">
        
        {/* Top Left Circle — Language Switcher (EN / HI) */}
        <button
          onClick={onToggleLang}
          className="w-11 h-11 rounded-full bg-[#8f9d68] text-white flex items-center justify-center font-extrabold text-xs shadow-md cursor-pointer hover:scale-105 active:scale-95 transition-transform border-2 border-white flex-col leading-none"
          title={`Switch Language (Current: ${currentLang === 'en' ? 'English' : 'Hindi'})`}
        >
          <div className="flex items-center gap-0.5">
            <Languages size={14} className="text-white" />
            <span className="text-[10px] uppercase font-black">{currentLang}</span>
          </div>
        </button>

        {/* Centered Title */}
        <div className="text-center flex-1">
          <h1 className="text-2xl font-black tracking-tight text-slate-900 font-sans">
            Argus
          </h1>
          <p className="text-[10px] uppercase font-bold tracking-widest text-[#8f9d68]">
            Powered by Asphr
          </p>
        </div>

        {/* Empty right balance space */}
        <div className="w-11" />
      </header>
    </div>
  );
};

export default MobileHeader;

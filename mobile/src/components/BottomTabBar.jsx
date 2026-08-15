import React from 'react';
import { Home, Navigation, User } from 'lucide-react';

const BottomTabBar = ({ activeTab = 'home', onTabChange = () => {} }) => {
  return (
    <nav className="fixed bottom-0 left-0 right-0 z-50 max-w-md mx-auto pointer-events-none">
      {/* Container holding the soft cream yellow (#fcf4d0) curved navigation bar */}
      <div className="relative pointer-events-auto bg-[#fcf4d0] border-t border-amber-200/70 px-6 py-2 shadow-[0_-6px_25px_rgba(0,0,0,0.06)] flex items-center justify-around min-h-[68px] rounded-t-3xl">
        
        {/* SVG Dip Curve background overlay around active tab */}
        <div className="absolute inset-0 overflow-hidden rounded-t-3xl pointer-events-none">
          <svg className="w-full h-full" viewBox="0 0 390 68" fill="none" preserveAspectRatio="none">
            <path
              d={
                activeTab === 'home'
                  ? 'M0 0 L40 0 C55 0 60 40 85 40 C110 40 115 0 130 0 L390 0 L390 68 L0 68 Z'
                  : activeTab === 'navigation'
                  ? 'M0 0 L145 0 C160 0 165 40 195 40 C225 40 230 0 245 0 L390 0 L390 68 L0 68 Z'
                  : 'M0 0 L260 0 C275 0 280 40 305 40 C330 40 335 0 350 0 L390 0 L390 68 L0 68 Z'
              }
              fill="#ffffff"
              opacity="0.95"
            />
          </svg>
        </div>

        {/* 1. Home Tab (Highlighted in #8f9d68 green when active) */}
        <button
          onClick={() => onTabChange('home')}
          className="relative z-10 flex flex-col items-center justify-center py-1 transition-all duration-300"
        >
          <div
            className={`w-12 h-12 rounded-full flex items-center justify-center transition-all duration-300 shadow-md ${
              activeTab === 'home'
                ? 'bg-[#8f9d68] text-white ring-4 ring-[#8f9d68]/25 scale-110'
                : 'bg-white text-slate-700 hover:bg-slate-50 border border-amber-200/60'
            }`}
          >
            <Home size={22} className={activeTab === 'home' ? 'text-white' : 'text-slate-700'} />
          </div>
          <span
            className={`text-[11px] mt-1 font-bold tracking-tight ${
              activeTab === 'home' ? 'text-[#8f9d68]' : 'text-slate-700'
            }`}
          >
            Home
          </span>
        </button>

        {/* 2. Navigation Tab (Center Circle with arrow pointing straight UPWARDS) */}
        <button
          onClick={() => onTabChange('navigation')}
          className="relative z-10 flex flex-col items-center justify-center py-1 transition-all duration-300"
        >
          <div
            className={`w-13 h-13 rounded-full flex items-center justify-center transition-all duration-300 shadow-md ${
              activeTab === 'navigation'
                ? 'bg-[#8f9d68] text-white ring-4 ring-[#8f9d68]/25 scale-110'
                : 'bg-white text-slate-700 hover:bg-slate-50 border border-amber-200/60'
            }`}
            style={{ width: '52px', height: '52px' }}
          >
            <Navigation
              size={22}
              className={`transform -rotate-45 ${activeTab === 'navigation' ? 'text-white' : 'text-slate-700'}`}
            />
          </div>
          <span
            className={`text-[11px] mt-1 font-bold tracking-tight ${
              activeTab === 'navigation' ? 'text-[#8f9d68]' : 'text-slate-700'
            }`}
          >
            Navigation
          </span>
        </button>

        {/* 3. Profile Tab */}
        <button
          onClick={() => onTabChange('user')}
          className="relative z-10 flex flex-col items-center justify-center py-1 transition-all duration-300"
        >
          <div
            className={`w-12 h-12 rounded-full flex items-center justify-center transition-all duration-300 shadow-md ${
              activeTab === 'user'
                ? 'bg-[#8f9d68] text-white ring-4 ring-[#8f9d68]/25 scale-110'
                : 'bg-white text-slate-700 hover:bg-slate-50 border border-amber-200/60'
            }`}
          >
            <User size={22} className={activeTab === 'user' ? 'text-white' : 'text-slate-700'} />
          </div>
          <span
            className={`text-[11px] mt-1 font-bold tracking-tight ${
              activeTab === 'user' ? 'text-[#8f9d68]' : 'text-slate-700'
            }`}
          >
            Profile
          </span>
        </button>

      </div>
    </nav>
  );
};

export default BottomTabBar;

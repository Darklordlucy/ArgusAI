import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const Logo = () => (
  <Link to="/" className="flex items-center">
    <img 
      src="/image.png" 
      alt="ASPHR Logo" 
      className="h-16 object-contain transition-all" 
    />
  </Link>
);

const ButtonLogo = () => (
  <svg width="14" height="14" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg" className="mr-2 text-brand-yellow">
    <path d="M50 15 L15 85 L85 85 Z" fill="currentColor"/>
    <path d="M50 35 L30 75 L70 75 Z" fill="#0F2027"/>
  </svg>
);

const Navbar = ({ theme = 'dark' }) => {
  const location = useLocation();

  const navLinks = [
    { path: '/', label: 'Home' },
    { path: '/maps', label: 'Maps' },
    { path: '/routes', label: 'Routes' },
    { path: '/services', label: 'Services' },
  ];

  return (
    <nav className="flex items-center justify-between px-10 py-3 fixed top-0 left-0 w-full z-50 bg-transparent">
      <Logo />
      
      {/* Navbar Container with #fef6d2 background pill for high contrast & visibility */}
      <div className="hidden md:flex items-center space-x-2 bg-[#fef6d2] px-4 py-2 rounded-full border border-amber-200/80 shadow-lg backdrop-blur-md">
        {navLinks.map((link) => {
          const isActive = location.pathname === link.path;
          return (
            <Link
              key={link.path}
              to={link.path}
              className={`px-4 py-1.5 rounded-full text-base font-extrabold transition-all duration-200 ${
                isActive
                  ? 'bg-[#8f9d68] text-white shadow-sm'
                  : 'text-slate-900 hover:text-[#8f9d68] hover:bg-white/60'
              }`}
            >
              {link.label}
            </Link>
          );
        })}
      </div>

      <Link
        to="/routes"
        className="bg-[#0F2027] hover:bg-black text-white px-6 py-2.5 rounded-full font-sans font-semibold text-sm flex items-center transition-all hover:scale-105 active:scale-95 shadow-xl border border-white/10"
      >
        <ButtonLogo />
        Start Your Journey
      </Link>
    </nav>
  );
};

export default Navbar;

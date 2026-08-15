import React, { useState, useEffect } from 'react';
import MobileHeader from './components/MobileHeader';
import BottomTabBar from './components/BottomTabBar';
import HomePage from './pages/HomePage';
import NavigationPage from './pages/NavigationPage';
import UserInfoPage from './pages/UserInfoPage';
import { wsService } from './services/websocket';

function App() {
  const [activeTab, setActiveTab] = useState('home');
  const [isWsConnected, setIsWsConnected] = useState(false);
  const [lang, setLang] = useState('en'); // 'en' | 'hi'

  useEffect(() => {
    wsService.connect();

    const unsubscribe = wsService.subscribe((msg) => {
      if (msg.type === 'status_change') {
        setIsWsConnected(msg.isConnected);
      }
    });

    return () => unsubscribe();
  }, []);

  const handleToggleLang = () => {
    setLang((prev) => (prev === 'en' ? 'hi' : 'en'));
  };

  return (
    <div className="min-h-screen bg-[#f8fafc] text-slate-900 max-w-md mx-auto relative shadow-2xl overflow-hidden font-sans border-x border-slate-200">
      {/* Mobile Top Header with Language Change (EN/HI) & Title */}
      <MobileHeader currentLang={lang} onToggleLang={handleToggleLang} />

      {/* Main Tab Content Container */}
      <main className="px-4 pt-1">
        {activeTab === 'home' && (
          <HomePage lang={lang} onNavigateToNav={() => setActiveTab('navigation')} />
        )}
        {activeTab === 'navigation' && <NavigationPage />}
        {activeTab === 'user' && <UserInfoPage />}
      </main>

      {/* Mobile Bottom Navigation Bar */}
      <BottomTabBar activeTab={activeTab} onTabChange={(tab) => setActiveTab(tab)} />
    </div>
  );
}

export default App;

import React, { useState, useEffect } from 'react';
import { 
  Activity, 
  AlertTriangle, 
  RefreshCw, 
  Compass, 
  ArrowRight,
  Search,
  Car,
  Route,
  Navigation,
  ShieldCheck,
  Zap,
  Sparkles
} from 'lucide-react';
import HomeMapView from '../components/HomeMapView';
import TrafficDrawer from '../components/TrafficDrawer';
import HazardDrawer from '../components/HazardDrawer';
import AnimatedTimelineStep from '../components/AnimatedTimelineStep';
import { fetchHazards, fetchHeavyTraffic } from '../services/api';
import { wsService } from '../services/websocket';

const HomePage = ({ lang = 'en', onNavigateToNav = () => {} }) => {
  // Active insight option: 'hazards' | 'traffic'
  const [selectedOption, setSelectedOption] = useState('hazards');

  // Real-time data states
  const [trafficData, setTrafficData] = useState([]);
  const [hazardsData, setHazardsData] = useState([]);
  const [selectedItem, setSelectedItem] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  // Translations dictionary for EN / HI
  const t = {
    en: {
      welcome: 'Welcome, Ravi!',
      badge: 'Smart Spatial Router',
      heroTitleLine1: 'Find Your Best Path',
      heroTitleLine2: 'WITH ASPHR',
      heroSub: 'Hazard-aware dynamic navigation powered by IoT telemetry & ML traffic forecasting.',
      quickInsights: 'Quick insights',
      refresh: 'Refresh',
      hazardsLabel: 'Hazards',
      trafficLabel: 'Live Traffic',
      showingLayer: 'Showing:',
      mapLayerText: 'Map Layer',
      howToUseTitle: 'How to use argus',
      howToUseSub: 'Scroll down to reveal step-by-step guidance',
      startRoute: 'Start Route',
      chooseRouteTitle: 'Choose your route',
      chooseRouteSub: 'Multi-objective routing strategies tuned to your journey goal',
    },
    hi: {
      welcome: 'स्वागत है, रवि!',
      badge: 'स्मार्ट स्थानिक राउटर',
      heroTitleLine1: 'अपना सर्वश्रेष्ठ मार्ग खोजें',
      heroTitleLine2: 'ASPHR के साथ',
      heroSub: 'IoT टेलीमेट्री और ML ट्रैफ़िक पूर्वानुमान द्वारा संचालित गतिशील सुरक्षित नेविगेशन।',
      quickInsights: 'त्वरित अंतर्दृष्टि',
      refresh: 'रिफ्रेश',
      hazardsLabel: 'सड़क खतरे',
      trafficLabel: 'लाइव ट्रैफिक',
      showingLayer: 'दिखा रहा है:',
      mapLayerText: 'मैप लेयर',
      howToUseTitle: 'Argus का उपयोग कैसे करें',
      howToUseSub: 'चरण-दर-चरण मार्गदर्शन देखने के लिए नीचे स्क्रॉल करें',
      startRoute: 'नेविगेशन शुरू करें',
      chooseRouteTitle: 'अपना मार्ग चुनें',
      chooseRouteSub: 'आपकी यात्रा के लक्ष्य के अनुसार बहु-उद्देश्यीय राउटिंग रणनीतियाँ',
    }
  }[lang] || t.en;

  // Default Mumbai datasets to display Mumbai data as default
  const DEFAULT_MUMBAI_TRAFFIC = [
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [72.8561, 19.0657] },
      properties: { name: "Bandra Kurla Complex (BKC) Connector", congestion_level: 3, speed_kmh: 12.5, color: "#EF4444", delay_sec: 420, magnitude: 3 }
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [72.8353, 18.9400] },
      properties: { name: "CSMT Junction & Fort Area", congestion_level: 2, speed_kmh: 22.0, color: "#F97316", delay_sec: 180, magnitude: 2 }
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [72.8697, 19.1197] },
      properties: { name: "Western Express Highway (Andheri Flyover)", congestion_level: 3, speed_kmh: 14.2, color: "#EF4444", delay_sec: 350, magnitude: 3 }
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [72.8411, 19.0178] },
      properties: { name: "Dadar TT Circle & Tilak Bridge", congestion_level: 2, speed_kmh: 18.5, color: "#F97316", delay_sec: 210, magnitude: 2 }
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [72.8223, 19.0410] },
      properties: { name: "Bandra-Worli Sea Link Toll Plaza", congestion_level: 1, speed_kmh: 45.0, color: "#8f9d68", delay_sec: 60, magnitude: 1 }
    }
  ];

  const DEFAULT_MUMBAI_HAZARDS = [
    {
      id: 101,
      latitude: 19.0657,
      longitude: 72.8687,
      hazard_score: 0.85,
      hazard_type: "pothole",
      segment_id: 10482,
      location_name: "BKC Signal, Mumbai"
    },
    {
      id: 102,
      latitude: 19.0178,
      longitude: 72.8411,
      hazard_score: 0.90,
      hazard_type: "crater",
      segment_id: 10483,
      location_name: "Dadar TT Bridge Flyover, Mumbai"
    },
    {
      id: 103,
      latitude: 19.1197,
      longitude: 72.8697,
      hazard_score: 0.75,
      hazard_type: "speed_bump",
      segment_id: 10484,
      location_name: "WEH Andheri East Exit, Mumbai"
    }
  ];

  // Load real-time data
  const loadData = async () => {
    setIsLoading(true);
    try {
      const trafficRes = await fetchHeavyTraffic().catch(() => ({ features: [] }));
      const trafficList = (trafficRes.features && trafficRes.features.length > 0)
        ? trafficRes.features
        : DEFAULT_MUMBAI_TRAFFIC;
      setTrafficData(trafficList);

      const hazardRes = await fetchHazards({
        minLat: 18.90,
        minLon: 72.75,
        maxLat: 19.50,
        maxLon: 73.20,
      }).catch(() => ({ hazards: [] }));
      const hazardList = (hazardRes.hazards && hazardRes.hazards.length > 0)
        ? hazardRes.hazards
        : DEFAULT_MUMBAI_HAZARDS;
      setHazardsData(hazardList);
    } catch (err) {
      console.warn('[HomePage] Data load error:', err);
      setTrafficData(DEFAULT_MUMBAI_TRAFFIC);
      setHazardsData(DEFAULT_MUMBAI_HAZARDS);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadData();

    const unsubscribe = wsService.subscribe((msg) => {
      if (msg.type === 'hazard_alert') {
        setHazardsData((prev) => [msg.data, ...prev]);
      }
    });

    return () => unsubscribe();
  }, []);

  // Quick insight options
  const insightOptions = [
    {
      id: 'hazards',
      label: t.hazardsLabel,
      subtext: `${hazardsData.length} ${lang === 'hi' ? 'खतरे दर्ज' : 'detected'}`,
      icon: AlertTriangle,
    },
    {
      id: 'traffic',
      label: t.trafficLabel,
      subtext: `${trafficData.length} ${lang === 'hi' ? 'घटनाएं' : 'incidents'}`,
      icon: Activity,
    },
  ];

  // User-specified concise 5 steps with Lucide icons (EN & HI)
  const howToUseSteps = lang === 'hi' ? [
    {
      icon: Search,
      title: 'गंतव्य खोजें',
      description: 'अपना स्थान खोजें और अपना मार्ग निर्धारित करें।',
    },
    {
      icon: Car,
      title: 'वाहन सेट करें',
      description: 'अपने वाहन का प्रकार और सीमाएं चुनें।',
    },
    {
      icon: Route,
      title: 'मार्ग चुनें',
      description: 'सबसे सुरक्षित, सबसे तेज़ या सीधा मार्ग चुनें।',
    },
    {
      icon: Activity,
      title: 'लाइव टेलीमेट्री देखें',
      description: 'वास्तविक समय की सड़क स्थिति और खतरे देखें।',
    },
    {
      icon: Navigation,
      title: 'नेविगेट करें और योगदान दें',
      description: 'अपने मार्ग का पालन करें और लाइव टेलीमेट्री साझा करें।',
    },
  ] : [
    {
      icon: Search,
      title: 'Search Destination',
      description: 'Find your location and set your route.',
    },
    {
      icon: Car,
      title: 'Set Vehicle',
      description: 'Choose your vehicle type and limits.',
    },
    {
      icon: Route,
      title: 'Choose Route',
      description: 'Select Safest, Fastest, or Straightest.',
    },
    {
      icon: Activity,
      title: 'View Live Telemetry',
      description: 'See real-time road conditions and hazards.',
    },
    {
      icon: Navigation,
      title: 'Navigate & Contribute',
      description: 'Follow your route and share live telemetry.',
    },
  ];

  // 4 Routing Strategies matching reference image (2x2 Grid)
  const routeStrategies = [
    {
      id: 'safest',
      label: lang === 'hi' ? 'सबसे सुरक्षित' : 'safest',
      icon: ShieldCheck,
      description: lang === 'hi' ? 'गड्ढों और खतरों से मुक्‍त' : 'Prioritizes low hazard & vibration segments',
      color: '#8f9d68',
    },
    {
      id: 'straightest',
      label: lang === 'hi' ? 'सीधा मार्ग' : 'straightest',
      icon: Compass,
      description: lang === 'hi' ? 'न्यूनतम मोड़' : 'Minimizes turns & bearing deviations',
      color: '#8f9d68',
    },
    {
      id: 'popular',
      label: lang === 'hi' ? 'लोकप्रिय दृश्य' : 'popular',
      icon: Sparkles,
      description: lang === 'hi' ? 'दर्शनीय स्थल मार्ग' : 'Scenic routing traversing POIs',
      color: '#ffd700',
    },
    {
      id: 'fastest',
      label: lang === 'hi' ? 'सबसे तेज़' : 'fastest',
      icon: Zap,
      description: lang === 'hi' ? 'LSTM ट्रैफ़िक पूर्वानुमान' : 'Minimizes travel time via TomTom & ML',
      color: '#8f9d68',
    },
  ];

  return (
    <div className="space-y-6 pb-32 font-sans text-slate-900">
      
      {/* Welcome Greeting */}
      <div className="px-1 pt-1 pb-0">
        <h2 className="text-lg font-extrabold text-slate-900">
          {t.welcome}
        </h2>
      </div>

      {/* 1. Hero Banner */}
      <div className="relative w-full rounded-3xl overflow-hidden shadow-sm border border-emerald-100 bg-gradient-to-br from-emerald-50 via-white to-amber-50 p-6 flex flex-col justify-end min-h-[200px]">
        <div className="absolute right-3 top-3 opacity-15 text-[#8f9d68]">
          <Compass size={110} />
        </div>

        <div className="relative z-10 space-y-1">
          <span className="px-2.5 py-1 rounded-full bg-[#8f9d68]/15 border border-[#8f9d68]/30 text-[10px] font-bold uppercase tracking-wider text-[#8f9d68] shadow-xs inline-block">
            {t.badge}
          </span>
          <h1 className="text-2xl font-black text-slate-900 tracking-tight leading-tight">
            {t.heroTitleLine1} <br />
            <span className="text-[#8f9d68]">{t.heroTitleLine2}</span>
          </h1>
          <p className="text-xs text-slate-600 max-w-[260px] font-medium pt-0.5">
            {t.heroSub}
          </p>
        </div>
      </div>

      {/* 2. Quick Insights Section */}
      <div className="space-y-2.5">
        <div className="flex items-center justify-between px-1">
          <h3 className="text-base font-extrabold text-slate-900 tracking-tight">
            {t.quickInsights}
          </h3>
          <button
            onClick={loadData}
            disabled={isLoading}
            className="text-[11px] font-bold text-[#8f9d68] hover:underline flex items-center gap-1"
          >
            <RefreshCw size={12} className={isLoading ? 'animate-spin' : ''} />
            {t.refresh}
          </button>
        </div>

        {/* 2 side-by-side insight cards */}
        <div className="grid grid-cols-2 gap-3">
          {insightOptions.map((opt) => {
            const isSelected = selectedOption === opt.id;
            const Icon = opt.icon;

            return (
              <button
                key={opt.id}
                onClick={() => {
                  setSelectedOption(opt.id);
                  setSelectedItem(null);
                }}
                className={`py-4 px-3 rounded-2xl text-left transition-all duration-200 flex flex-col justify-between space-y-2 shadow-sm border ${
                  isSelected
                    ? 'bg-[#f0fdf4] border-[#8f9d68] text-slate-900 font-bold ring-2 ring-[#8f9d68]/30 scale-[1.02]'
                    : 'bg-white border-slate-200 text-slate-700 hover:border-[#8f9d68]/50 hover:bg-slate-50'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className={`p-2 rounded-xl ${isSelected ? 'bg-[#8f9d68] text-white' : 'bg-slate-100 text-[#8f9d68]'}`}>
                    <Icon size={18} />
                  </div>
                  <span className={`text-[10px] font-extrabold px-2 py-0.5 rounded-full ${isSelected ? 'bg-[#8f9d68]/20 text-[#8f9d68]' : 'bg-slate-100 text-slate-500'}`}>
                    {opt.subtext}
                  </span>
                </div>
                <div>
                  <h4 className="text-sm font-black text-slate-900 tracking-tight">
                    {opt.label}
                  </h4>
                </div>
              </button>
            );
          })}
        </div>
      </div>



      {/* 4. Active Insight Drawer */}
      <div className="transition-all duration-300">
        {selectedOption === 'traffic' && (
          <TrafficDrawer trafficList={trafficData} />
        )}
        {selectedOption === 'hazards' && (
          <HazardDrawer hazardsList={hazardsData} onReportNewHazard={loadData} />
        )}
      </div>

      {/* 5. "How to use argus" Timeline with icons & scroll-reveal animation */}
      <div className="pt-4 pb-2 space-y-5">
        <div className="flex items-center justify-between px-1 border-b border-slate-200 pb-2.5">
          <div>
            <h3 className="text-base font-extrabold text-slate-900 tracking-tight">
              {t.howToUseTitle}
            </h3>
            <p className="text-[11px] text-slate-500 font-medium">
              {t.howToUseSub}
            </p>
          </div>
          <button
            onClick={() => onNavigateToNav('fastest')}
            className="text-xs font-bold text-[#8f9d68] hover:underline flex items-center gap-1 bg-[#8f9d68]/10 px-3 py-1.5 rounded-lg border border-[#8f9d68]/20"
          >
            {t.startRoute} <ArrowRight size={14} />
          </button>
        </div>

        {/* Vertical timeline with red dashed line and icon circles */}
        <div className="relative pl-3 space-y-7">
          <div className="absolute left-[26px] top-4 bottom-8 w-0.5 border-l-2 border-dashed border-red-500 z-0" />

          {howToUseSteps.map((step, idx) => (
            <AnimatedTimelineStep
              key={idx}
              step={step}
              index={idx}
              isLast={idx === howToUseSteps.length - 1}
            />
          ))}
        </div>
      </div>

      {/* 6. "Choose your route" Section matching reference image (2x2 Grid below "How to use") */}
      <div className="pt-4 space-y-3">
        <div className="px-1">
          <h3 className="text-base font-extrabold text-slate-900 tracking-tight">
            {t.chooseRouteTitle}
          </h3>
          <p className="text-[11px] text-slate-500 font-medium">
            {t.chooseRouteSub}
          </p>
        </div>

        {/* 2x2 Grid for safest, straightest, popular, fastest */}
        <div className="grid grid-cols-2 gap-3">
          {routeStrategies.map((strat) => {
            const Icon = strat.icon;

            return (
              <button
                key={strat.id}
                onClick={() => onNavigateToNav(strat.id)}
                className="p-5 rounded-2xl bg-white border border-slate-200 text-slate-900 hover:border-[#8f9d68] hover:bg-[#f0fdf4] shadow-sm hover:shadow-md transition-all duration-200 flex flex-col items-center justify-center text-center space-y-2 group cursor-pointer active:scale-95 min-h-[120px]"
              >
                <div className="p-3 rounded-2xl bg-[#8f9d68]/15 text-[#8f9d68] border border-[#8f9d68]/30 group-hover:bg-[#8f9d68] group-hover:text-white transition-colors">
                  <Icon size={24} />
                </div>
                <div>
                  <h4 className="text-sm font-black tracking-tight text-slate-900 capitalize group-hover:text-[#8f9d68] transition-colors">
                    {strat.label}
                  </h4>
                  <p className="text-[10px] text-slate-500 font-medium mt-0.5 line-clamp-1">
                    {strat.description}
                  </p>
                </div>
              </button>
            );
          })}
        </div>
      </div>

    </div>
  );
};

export default HomePage;

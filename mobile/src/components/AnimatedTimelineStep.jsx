import React, { useEffect, useRef, useState } from 'react';

const AnimatedTimelineStep = ({ step, index, isLast = false }) => {
  const [isVisible, setIsVisible] = useState(false);
  const domRef = useRef(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setIsVisible(true);
          }
        });
      },
      { threshold: 0.25 }
    );

    const currentRef = domRef.current;
    if (currentRef) {
      observer.observe(currentRef);
    }

    return () => {
      if (currentRef) observer.unobserve(currentRef);
    };
  }, []);

  const StepIcon = step.icon;

  return (
    <div
      ref={domRef}
      className={`relative z-10 flex items-start space-x-4 transition-all duration-700 ease-out transform ${
        isVisible ? 'opacity-100 translate-y-0 scale-100' : 'opacity-0 translate-y-8 scale-95'
      }`}
    >
      {/* Icon circle in vibrant #8f9d68 green instead of numbers */}
      <div
        className={`w-10 h-10 rounded-full border-2 border-white text-white font-bold flex items-center justify-center shrink-0 shadow-md transition-all duration-500 ${
          isVisible
            ? 'bg-[#8f9d68] ring-4 ring-[#8f9d68]/20 scale-110'
            : 'bg-slate-300 text-slate-600'
        }`}
      >
        {StepIcon ? <StepIcon size={20} className="text-white" /> : index + 1}
      </div>

      {/* Step text details */}
      <div className="pt-0.5 space-y-0.5">
        <h4 className="text-xs font-black text-slate-900 tracking-tight flex items-center gap-1.5">
          <span>{step.title}</span>
          {isVisible && (
            <span className="w-1.5 h-1.5 rounded-full bg-[#8f9d68] animate-ping inline-block" />
          )}
        </h4>
        <p className="text-[11px] text-slate-600 leading-relaxed max-w-xs font-medium">
          {step.description}
        </p>
      </div>
    </div>
  );
};

export default AnimatedTimelineStep;

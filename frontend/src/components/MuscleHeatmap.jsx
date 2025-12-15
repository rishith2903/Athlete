import React from 'react';

/**
 * SVG Muscle Heatmap Component
 * Displays a body diagram with muscles colored based on training intensity
 * @param {Object} muscleData - Object with muscle names as keys and intensity (0-100) as values
 * @param {boolean} isFront - Show front or back view
 */
const MuscleHeatmap = ({ muscleData = {}, isFront = true, onMuscleClick }) => {
    const getColor = (intensity) => {
        if (!intensity || intensity === 0) return '#E5E7EB'; // gray-200
        if (intensity < 25) return '#86EFAC'; // green-300
        if (intensity < 50) return '#FDE047'; // yellow-300
        if (intensity < 75) return '#FB923C'; // orange-400
        return '#EF4444'; // red-500
    };

    const getIntensity = (muscle) => muscleData[muscle] || 0;

    // Front view muscles
    const frontMuscles = {
        chest: { path: 'M60,95 Q80,90 100,95 Q100,115 80,120 Q60,115 60,95', label: 'Chest' },
        shoulders_front: { path: 'M45,85 Q55,80 60,90 L60,100 Q50,105 45,95 Z M100,90 Q105,80 115,85 L115,95 Q110,105 100,100 Z', label: 'Shoulders' },
        biceps: { path: 'M40,100 Q35,120 40,140 Q48,140 50,120 Q48,100 40,100 M120,100 Q125,120 120,140 Q112,140 110,120 Q112,100 120,100', label: 'Biceps' },
        abs: { path: 'M65,125 L95,125 L95,175 L65,175 Z', label: 'Abs' },
        obliques: { path: 'M55,130 L65,125 L65,175 L55,165 Z M95,125 L105,130 L105,165 L95,175 Z', label: 'Obliques' },
        quads: { path: 'M55,180 Q50,210 55,250 L75,250 Q80,210 75,180 Z M85,180 Q80,210 85,250 L105,250 Q110,210 105,180 Z', label: 'Quads' },
        forearms_front: { path: 'M35,145 Q30,165 35,185 L45,185 Q50,165 45,145 Z M115,145 Q120,165 115,185 L125,185 Q130,165 125,145 Z', label: 'Forearms' },
    };

    // Back view muscles
    const backMuscles = {
        traps: { path: 'M60,75 Q80,70 100,75 L100,90 Q80,85 60,90 Z', label: 'Traps' },
        lats: { path: 'M55,95 L60,95 L65,140 L55,140 Z M95,95 L105,95 L105,140 L95,140 Z', label: 'Lats' },
        shoulders_back: { path: 'M45,85 Q55,80 60,90 L60,100 Q50,105 45,95 Z M100,90 Q105,80 115,85 L115,95 Q110,105 100,100 Z', label: 'Rear Delts' },
        triceps: { path: 'M40,100 Q35,120 40,140 Q48,140 50,120 Q48,100 40,100 M120,100 Q125,120 120,140 Q112,140 110,120 Q112,100 120,100', label: 'Triceps' },
        lower_back: { path: 'M65,145 L95,145 L95,175 L65,175 Z', label: 'Lower Back' },
        glutes: { path: 'M55,175 L105,175 L105,200 Q80,210 55,200 Z', label: 'Glutes' },
        hamstrings: { path: 'M55,205 Q50,235 55,265 L75,265 Q78,235 75,205 Z M85,205 Q82,235 85,265 L105,265 Q110,235 105,205 Z', label: 'Hamstrings' },
        calves: { path: 'M58,270 Q55,295 60,320 L72,320 Q77,295 72,270 Z M88,270 Q83,295 88,320 L100,320 Q105,295 100,270 Z', label: 'Calves' },
    };

    const muscles = isFront ? frontMuscles : backMuscles;

    return (
        <div className="relative">
            <svg viewBox="0 0 160 340" className="w-full max-w-[200px] mx-auto">
                {/* Body outline */}
                <ellipse cx="80" cy="40" rx="25" ry="30" fill="#FCD5B8" stroke="#D4A574" strokeWidth="1" />

                {/* Neck */}
                <rect x="70" y="65" width="20" height="15" fill="#FCD5B8" stroke="#D4A574" strokeWidth="1" />

                {/* Torso */}
                <path
                    d="M45,80 Q40,85 45,180 Q55,185 80,185 Q105,185 115,180 Q120,85 115,80 Q100,75 80,75 Q60,75 45,80"
                    fill="#FCD5B8"
                    stroke="#D4A574"
                    strokeWidth="1"
                />

                {/* Arms */}
                <path d="M45,85 Q30,100 25,145 Q20,185 30,190 L45,190 Q55,145 50,100" fill="#FCD5B8" stroke="#D4A574" strokeWidth="1" />
                <path d="M115,85 Q130,100 135,145 Q140,185 130,190 L115,190 Q105,145 110,100" fill="#FCD5B8" stroke="#D4A574" strokeWidth="1" />

                {/* Legs */}
                <path d="M55,180 Q45,250 50,330 L75,330 Q85,250 75,180" fill="#FCD5B8" stroke="#D4A574" strokeWidth="1" />
                <path d="M85,180 Q75,250 80,330 L110,330 Q115,250 105,180" fill="#FCD5B8" stroke="#D4A574" strokeWidth="1" />

                {/* Muscle overlays */}
                {Object.entries(muscles).map(([key, { path, label }]) => (
                    <g key={key}>
                        <path
                            d={path}
                            fill={getColor(getIntensity(key))}
                            fillOpacity="0.7"
                            stroke="#666"
                            strokeWidth="0.5"
                            className="cursor-pointer hover:opacity-90 transition-opacity"
                            onClick={() => onMuscleClick?.(key, label)}
                        />
                    </g>
                ))}
            </svg>

            {/* Legend */}
            <div className="flex justify-center gap-4 mt-4 text-xs">
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded" style={{ backgroundColor: '#E5E7EB' }}></div>
                    <span className="text-gray-500 dark:text-gray-400">Rest</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded" style={{ backgroundColor: '#86EFAC' }}></div>
                    <span className="text-gray-500 dark:text-gray-400">Light</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded" style={{ backgroundColor: '#FDE047' }}></div>
                    <span className="text-gray-500 dark:text-gray-400">Moderate</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded" style={{ backgroundColor: '#FB923C' }}></div>
                    <span className="text-gray-500 dark:text-gray-400">Heavy</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded" style={{ backgroundColor: '#EF4444' }}></div>
                    <span className="text-gray-500 dark:text-gray-400">Intense</span>
                </div>
            </div>
        </div>
    );
};

export default MuscleHeatmap;

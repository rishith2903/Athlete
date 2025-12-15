import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Trophy, ChevronDown, Info } from 'lucide-react';

/**
 * Barbell Standards Component
 * Compare lifts against strength standards
 */
const BarbellStandards = () => {
    const [selectedLift, setSelectedLift] = useState('squat');
    const [bodyweight, setBodyweight] = useState(80);
    const [gender, setGender] = useState('male');
    const [userMax, setUserMax] = useState('');

    // Strength standards as multipliers of bodyweight
    const standards = {
        male: {
            squat: { beginner: 0.75, novice: 1.25, intermediate: 1.75, advanced: 2.25, elite: 2.75 },
            bench: { beginner: 0.5, novice: 0.85, intermediate: 1.15, advanced: 1.5, elite: 1.85 },
            deadlift: { beginner: 1.0, novice: 1.5, intermediate: 2.0, advanced: 2.5, elite: 3.0 },
            ohp: { beginner: 0.35, novice: 0.55, intermediate: 0.8, advanced: 1.0, elite: 1.2 },
            row: { beginner: 0.5, novice: 0.75, intermediate: 1.0, advanced: 1.25, elite: 1.5 },
        },
        female: {
            squat: { beginner: 0.5, novice: 0.85, intermediate: 1.25, advanced: 1.6, elite: 2.0 },
            bench: { beginner: 0.25, novice: 0.5, intermediate: 0.75, advanced: 1.0, elite: 1.25 },
            deadlift: { beginner: 0.75, novice: 1.15, intermediate: 1.5, advanced: 2.0, elite: 2.5 },
            ohp: { beginner: 0.2, novice: 0.35, intermediate: 0.5, advanced: 0.65, elite: 0.8 },
            row: { beginner: 0.35, novice: 0.55, intermediate: 0.75, advanced: 1.0, elite: 1.2 },
        }
    };

    const lifts = [
        { id: 'squat', name: 'Squat', icon: '🏋️' },
        { id: 'bench', name: 'Bench Press', icon: '💪' },
        { id: 'deadlift', name: 'Deadlift', icon: '🔥' },
        { id: 'ohp', name: 'Overhead Press', icon: '⬆️' },
        { id: 'row', name: 'Barbell Row', icon: '🚣' },
    ];

    const levels = [
        { key: 'beginner', label: 'Beginner', color: 'bg-gray-400', desc: 'Just started training' },
        { key: 'novice', label: 'Novice', color: 'bg-green-500', desc: '3-6 months of training' },
        { key: 'intermediate', label: 'Intermediate', color: 'bg-blue-500', desc: '1-2 years of training' },
        { key: 'advanced', label: 'Advanced', color: 'bg-purple-500', desc: '3-5 years of training' },
        { key: 'elite', label: 'Elite', color: 'bg-yellow-500', desc: 'Competitive level' },
    ];

    const currentStandards = standards[gender][selectedLift];
    const userWeight = parseFloat(userMax) || 0;
    const userRatio = bodyweight > 0 ? userWeight / bodyweight : 0;

    const getUserLevel = () => {
        if (userRatio >= currentStandards.elite) return 'elite';
        if (userRatio >= currentStandards.advanced) return 'advanced';
        if (userRatio >= currentStandards.intermediate) return 'intermediate';
        if (userRatio >= currentStandards.novice) return 'novice';
        if (userRatio >= currentStandards.beginner) return 'beginner';
        return 'untrained';
    };

    const userLevel = getUserLevel();
    const nextLevel = levels.find((_, i) => levels[i]?.key === userLevel && levels[i + 1])?.key || 'elite';
    const nextTarget = currentStandards[nextLevel] ? currentStandards[nextLevel] * bodyweight : null;

    return (
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm space-y-6">
            <div className="flex items-center gap-3">
                <Trophy className="h-6 w-6 text-yellow-500" />
                <h2 className="text-xl font-bold text-gray-900 dark:text-white">Strength Standards</h2>
            </div>

            {/* Inputs */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Lift</label>
                    <select
                        value={selectedLift}
                        onChange={(e) => setSelectedLift(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
                    >
                        {lifts.map(lift => (
                            <option key={lift.id} value={lift.id}>{lift.icon} {lift.name}</option>
                        ))}
                    </select>
                </div>
                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Gender</label>
                    <select
                        value={gender}
                        onChange={(e) => setGender(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
                    >
                        <option value="male">Male</option>
                        <option value="female">Female</option>
                    </select>
                </div>
                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Bodyweight (kg)</label>
                    <input
                        type="number"
                        value={bodyweight}
                        onChange={(e) => setBodyweight(parseFloat(e.target.value) || 0)}
                        className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
                    />
                </div>
                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Your 1RM (kg)</label>
                    <input
                        type="number"
                        value={userMax}
                        onChange={(e) => setUserMax(e.target.value)}
                        placeholder="Enter your max"
                        className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
                    />
                </div>
            </div>

            {/* Standards Table */}
            <div className="overflow-x-auto">
                <table className="w-full">
                    <thead>
                        <tr className="border-b border-gray-200 dark:border-gray-700">
                            <th className="text-left py-2 text-sm font-medium text-gray-500 dark:text-gray-400">Level</th>
                            <th className="text-right py-2 text-sm font-medium text-gray-500 dark:text-gray-400">Multiplier</th>
                            <th className="text-right py-2 text-sm font-medium text-gray-500 dark:text-gray-400">Weight (kg)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {levels.map(level => {
                            const multiplier = currentStandards[level.key];
                            const weight = Math.round(bodyweight * multiplier);
                            const isUserLevel = userLevel === level.key;

                            return (
                                <tr
                                    key={level.key}
                                    className={`border-b border-gray-100 dark:border-gray-700 ${isUserLevel ? 'bg-blue-50 dark:bg-blue-900/20' : ''}`}
                                >
                                    <td className="py-3">
                                        <div className="flex items-center gap-2">
                                            <div className={`w-3 h-3 rounded-full ${level.color}`}></div>
                                            <span className={`font-medium ${isUserLevel ? 'text-blue-600 dark:text-blue-400' : 'text-gray-900 dark:text-white'}`}>
                                                {level.label}
                                                {isUserLevel && <span className="ml-2 text-xs bg-blue-100 dark:bg-blue-800 px-2 py-0.5 rounded">You</span>}
                                            </span>
                                        </div>
                                    </td>
                                    <td className="text-right py-3 text-gray-600 dark:text-gray-400">{multiplier}x BW</td>
                                    <td className="text-right py-3 font-semibold text-gray-900 dark:text-white">{weight} kg</td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>

            {/* Progress to next level */}
            {userMax && nextTarget && userLevel !== 'elite' && (
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4"
                >
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-blue-800 dark:text-blue-200">
                            Progress to {levels.find(l => l.key === nextLevel)?.label}
                        </span>
                        <span className="text-sm text-blue-600 dark:text-blue-400">
                            {Math.round(nextTarget - userWeight)} kg to go
                        </span>
                    </div>
                    <div className="h-2 bg-blue-200 dark:bg-blue-800 rounded-full overflow-hidden">
                        <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${Math.min((userWeight / nextTarget) * 100, 100)}%` }}
                            className="h-full bg-blue-500 rounded-full"
                        />
                    </div>
                </motion.div>
            )}

            {/* Info */}
            <div className="flex items-start gap-2 text-xs text-gray-500 dark:text-gray-400">
                <Info className="h-4 w-4 mt-0.5 flex-shrink-0" />
                <span>Standards based on strength level classifications. Individual results may vary based on training history, age, and other factors.</span>
            </div>
        </div>
    );
};

export default BarbellStandards;

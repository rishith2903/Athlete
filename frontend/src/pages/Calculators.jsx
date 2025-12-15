import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Calculator,
    Weight,
    Target,
    RefreshCw,
    Info,
    Copy,
    Check
} from 'lucide-react';

const Calculators = () => {
    const [activeTab, setActiveTab] = useState('1rm');

    return (
        <div className="space-y-6">
            {/* Header */}
            <div>
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Calculators</h1>
                <p className="text-gray-600 dark:text-gray-400">Fitness calculation tools</p>
            </div>

            {/* Tabs */}
            <div className="flex gap-2 bg-gray-100 dark:bg-gray-800 p-1 rounded-xl">
                {[
                    { id: '1rm', label: '1RM Calculator', icon: Target },
                    { id: 'plate', label: 'Plate Calculator', icon: Weight },
                ].map(tab => (
                    <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={`flex-1 flex items-center justify-center gap-2 py-3 rounded-lg font-medium transition ${activeTab === tab.id
                                ? 'bg-white dark:bg-gray-700 text-blue-600 dark:text-blue-400 shadow-sm'
                                : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
                            }`}
                    >
                        <tab.icon className="h-5 w-5" />
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* Calculator Content */}
            <AnimatePresence mode="wait">
                {activeTab === '1rm' && (
                    <motion.div
                        key="1rm"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                    >
                        <OneRepMaxCalculator />
                    </motion.div>
                )}
                {activeTab === 'plate' && (
                    <motion.div
                        key="plate"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                    >
                        <PlateCalculator />
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

// 1RM Calculator Component
const OneRepMaxCalculator = () => {
    const [weight, setWeight] = useState('');
    const [reps, setReps] = useState('');
    const [results, setResults] = useState(null);
    const [copied, setCopied] = useState(false);

    const formulas = {
        epley: (w, r) => w * (1 + r / 30),
        brzycki: (w, r) => w * (36 / (37 - r)),
        lander: (w, r) => (100 * w) / (101.3 - 2.67123 * r),
        lombardi: (w, r) => w * Math.pow(r, 0.1),
        oconner: (w, r) => w * (1 + 0.025 * r),
    };

    const calculate = () => {
        const w = parseFloat(weight);
        const r = parseInt(reps);

        if (!w || !r || r < 1 || r > 30) {
            alert('Please enter valid weight and reps (1-30)');
            return;
        }

        const estimates = {
            epley: formulas.epley(w, r),
            brzycki: formulas.brzycki(w, r),
            lander: formulas.lander(w, r),
            lombardi: formulas.lombardi(w, r),
            oconner: formulas.oconner(w, r),
        };

        const average = Object.values(estimates).reduce((a, b) => a + b, 0) / 5;

        // Calculate rep maxes
        const repMaxes = [];
        for (let i = 1; i <= 12; i++) {
            const percentage = 100 - (i - 1) * 2.5;
            repMaxes.push({
                reps: i,
                weight: Math.round(average * (percentage / 100)),
                percentage: Math.round(percentage)
            });
        }

        setResults({ estimates, average, repMaxes });
    };

    const copyResults = () => {
        if (!results) return;
        const text = `Estimated 1RM: ${Math.round(results.average)} kg\n\n` +
            results.repMaxes.map(r => `${r.reps} rep: ${r.weight} kg (${r.percentage}%)`).join('\n');
        navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    return (
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm space-y-6">
            <div className="flex items-start justify-between">
                <div>
                    <h2 className="text-xl font-bold text-gray-900 dark:text-white">One Rep Max Calculator</h2>
                    <p className="text-gray-600 dark:text-gray-400">Estimate your max lift from submaximal sets</p>
                </div>
                <div className="p-3 bg-blue-100 dark:bg-blue-900/30 rounded-xl">
                    <Target className="h-6 w-6 text-blue-600 dark:text-blue-400" />
                </div>
            </div>

            {/* Input Fields */}
            <div className="grid grid-cols-2 gap-4">
                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        Weight Lifted
                    </label>
                    <div className="relative">
                        <input
                            type="number"
                            value={weight}
                            onChange={(e) => setWeight(e.target.value)}
                            placeholder="100"
                            className="w-full px-4 py-3 border border-gray-200 dark:border-gray-700 rounded-lg bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white text-lg"
                        />
                        <span className="absolute right-4 top-1/2 -translate-y-1/2 text-gray-500">kg</span>
                    </div>
                </div>
                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        Reps Completed
                    </label>
                    <input
                        type="number"
                        value={reps}
                        onChange={(e) => setReps(e.target.value)}
                        placeholder="5"
                        min="1"
                        max="30"
                        className="w-full px-4 py-3 border border-gray-200 dark:border-gray-700 rounded-lg bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white text-lg"
                    />
                </div>
            </div>

            <button
                onClick={calculate}
                className="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-semibold flex items-center justify-center gap-2 transition"
            >
                <Calculator className="h-5 w-5" />
                Calculate 1RM
            </button>

            {/* Results */}
            {results && (
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="space-y-6"
                >
                    {/* Main Result */}
                    <div className="bg-gradient-to-r from-blue-600 to-indigo-600 rounded-xl p-6 text-white">
                        <div className="text-center">
                            <div className="text-sm opacity-80 mb-1">Estimated One Rep Max</div>
                            <div className="text-5xl font-bold">{Math.round(results.average)} kg</div>
                        </div>
                    </div>

                    {/* Rep Maxes Table */}
                    <div>
                        <div className="flex items-center justify-between mb-3">
                            <h3 className="font-semibold text-gray-900 dark:text-white">Training Percentages</h3>
                            <button
                                onClick={copyResults}
                                className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg text-gray-500"
                            >
                                {copied ? <Check className="h-5 w-5 text-green-500" /> : <Copy className="h-5 w-5" />}
                            </button>
                        </div>
                        <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 gap-2">
                            {results.repMaxes.map(rm => (
                                <div
                                    key={rm.reps}
                                    className="bg-gray-50 dark:bg-gray-900 p-3 rounded-lg text-center"
                                >
                                    <div className="text-xs text-gray-500 dark:text-gray-400">{rm.reps} rep</div>
                                    <div className="text-lg font-bold text-gray-900 dark:text-white">{rm.weight}</div>
                                    <div className="text-xs text-gray-500 dark:text-gray-400">{rm.percentage}%</div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Formulas */}
                    <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                        <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-2">
                            <Info className="h-4 w-4" />
                            <span>Formula Estimates</span>
                        </div>
                        <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 text-sm">
                            {Object.entries(results.estimates).map(([name, value]) => (
                                <div key={name} className="text-center">
                                    <div className="text-gray-500 dark:text-gray-400 capitalize">{name}</div>
                                    <div className="font-semibold text-gray-900 dark:text-white">{Math.round(value)} kg</div>
                                </div>
                            ))}
                        </div>
                    </div>
                </motion.div>
            )}
        </div>
    );
};

// Plate Calculator Component
const PlateCalculator = () => {
    const [targetWeight, setTargetWeight] = useState('');
    const [barWeight, setBarWeight] = useState(20);
    const [unit, setUnit] = useState('kg');
    const [plates, setPlates] = useState(null);

    const plateWeights = {
        kg: [25, 20, 15, 10, 5, 2.5, 1.25],
        lbs: [45, 35, 25, 10, 5, 2.5]
    };

    const calculate = () => {
        const target = parseFloat(targetWeight);
        const bar = parseFloat(barWeight);

        if (!target || target < bar) {
            alert(`Target weight must be at least ${bar} ${unit}`);
            return;
        }

        const perSide = (target - bar) / 2;
        const available = plateWeights[unit];
        const result = [];
        let remaining = perSide;

        for (const plate of available) {
            const count = Math.floor(remaining / plate);
            if (count > 0) {
                result.push({ weight: plate, count });
                remaining -= plate * count;
            }
        }

        if (remaining > 0.1) {
            result.push({ weight: remaining, count: 1, note: 'adjust' });
        }

        setPlates({
            perSide: result,
            totalPerSide: perSide,
            barWeight: bar
        });
    };

    const getPlateColor = (weight) => {
        const colors = {
            45: 'bg-blue-500', 35: 'bg-yellow-500', 25: 'bg-red-500',
            20: 'bg-blue-500', 15: 'bg-yellow-500', 10: 'bg-green-500',
            5: 'bg-gray-400', 2.5: 'bg-gray-300', 1.25: 'bg-gray-200'
        };
        return colors[weight] || 'bg-gray-400';
    };

    return (
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm space-y-6">
            <div className="flex items-start justify-between">
                <div>
                    <h2 className="text-xl font-bold text-gray-900 dark:text-white">Plate Calculator</h2>
                    <p className="text-gray-600 dark:text-gray-400">Calculate plates needed for your target weight</p>
                </div>
                <div className="p-3 bg-green-100 dark:bg-green-900/30 rounded-xl">
                    <Weight className="h-6 w-6 text-green-600 dark:text-green-400" />
                </div>
            </div>

            {/* Unit Toggle */}
            <div className="flex gap-2">
                {['kg', 'lbs'].map(u => (
                    <button
                        key={u}
                        onClick={() => { setUnit(u); setBarWeight(u === 'kg' ? 20 : 45); }}
                        className={`px-4 py-2 rounded-lg font-medium transition ${unit === u
                                ? 'bg-blue-600 text-white'
                                : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                            }`}
                    >
                        {u.toUpperCase()}
                    </button>
                ))}
            </div>

            {/* Input Fields */}
            <div className="grid grid-cols-2 gap-4">
                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        Target Weight
                    </label>
                    <div className="relative">
                        <input
                            type="number"
                            value={targetWeight}
                            onChange={(e) => setTargetWeight(e.target.value)}
                            placeholder={unit === 'kg' ? '100' : '225'}
                            className="w-full px-4 py-3 border border-gray-200 dark:border-gray-700 rounded-lg bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white text-lg"
                        />
                        <span className="absolute right-4 top-1/2 -translate-y-1/2 text-gray-500">{unit}</span>
                    </div>
                </div>
                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        Bar Weight
                    </label>
                    <select
                        value={barWeight}
                        onChange={(e) => setBarWeight(parseFloat(e.target.value))}
                        className="w-full px-4 py-3 border border-gray-200 dark:border-gray-700 rounded-lg bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white text-lg"
                    >
                        {unit === 'kg' ? (
                            <>
                                <option value="20">20 kg (Standard)</option>
                                <option value="15">15 kg (Women's)</option>
                                <option value="10">10 kg (EZ Bar)</option>
                            </>
                        ) : (
                            <>
                                <option value="45">45 lbs (Standard)</option>
                                <option value="35">35 lbs (Women's)</option>
                                <option value="25">25 lbs (EZ Bar)</option>
                            </>
                        )}
                    </select>
                </div>
            </div>

            <button
                onClick={calculate}
                className="w-full py-3 bg-green-600 hover:bg-green-700 text-white rounded-xl font-semibold flex items-center justify-center gap-2 transition"
            >
                <Calculator className="h-5 w-5" />
                Calculate Plates
            </button>

            {/* Results */}
            {plates && (
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="space-y-6"
                >
                    {/* Visual Barbell */}
                    <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6">
                        <div className="text-center mb-4">
                            <div className="text-sm text-gray-500 dark:text-gray-400">Per Side</div>
                            <div className="text-2xl font-bold text-gray-900 dark:text-white">
                                {plates.totalPerSide} {unit}
                            </div>
                        </div>

                        {/* Barbell Visualization */}
                        <div className="flex items-center justify-center gap-1">
                            {/* Left plates (reversed) */}
                            <div className="flex items-center gap-1">
                                {[...plates.perSide].reverse().map((p, i) => (
                                    Array(p.count).fill(0).map((_, j) => (
                                        <div
                                            key={`left-${i}-${j}`}
                                            className={`h-16 w-3 ${getPlateColor(p.weight)} rounded-sm`}
                                            title={`${p.weight} ${unit}`}
                                        />
                                    ))
                                ))}
                            </div>

                            {/* Bar */}
                            <div className="h-4 w-24 bg-gray-400 dark:bg-gray-600 rounded-full" />

                            {/* Right plates */}
                            <div className="flex items-center gap-1">
                                {plates.perSide.map((p, i) => (
                                    Array(p.count).fill(0).map((_, j) => (
                                        <div
                                            key={`right-${i}-${j}`}
                                            className={`h-16 w-3 ${getPlateColor(p.weight)} rounded-sm`}
                                            title={`${p.weight} ${unit}`}
                                        />
                                    ))
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Plate List */}
                    <div>
                        <h3 className="font-semibold text-gray-900 dark:text-white mb-3">Plates Per Side</h3>
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                            {plates.perSide.map((p, i) => (
                                <div
                                    key={i}
                                    className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-900 rounded-lg"
                                >
                                    <div className="flex items-center gap-2">
                                        <div className={`h-8 w-2 ${getPlateColor(p.weight)} rounded-sm`} />
                                        <span className="font-medium text-gray-900 dark:text-white">
                                            {p.weight} {unit}
                                        </span>
                                    </div>
                                    <span className="text-lg font-bold text-blue-600 dark:text-blue-400">
                                        ×{p.count}
                                    </span>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Summary */}
                    <div className="bg-blue-50 dark:bg-blue-900/30 rounded-lg p-4 text-center">
                        <div className="text-sm text-blue-800 dark:text-blue-200">
                            Bar ({plates.barWeight} {unit}) + Plates ({plates.totalPerSide * 2} {unit}) = <strong>{parseFloat(targetWeight)} {unit}</strong>
                        </div>
                    </div>
                </motion.div>
            )}
        </div>
    );
};

export default Calculators;

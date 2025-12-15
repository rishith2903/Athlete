import React, { useState, useEffect, Suspense } from 'react';
import { motion } from 'framer-motion';
import { BarChart3, TrendingUp, Target, Flame, Calendar, ChevronDown, Box, Grid } from 'lucide-react';
import MuscleHeatmap from '../components/MuscleHeatmap';
import StrengthCurves from '../components/StrengthCurves';
import BarbellStandards from '../components/BarbellStandards';
import api from '../services/api';

// Lazy load 3D component for performance
const Body3D = React.lazy(() => import('../components/Body3D'));

/**
 * Analytics Page
 * Muscle heatmaps, strength curves, and barbell standards
 */
const Analytics = () => {
    const [muscleData, setMuscleData] = useState({});
    const [view3D, setView3D] = useState(true);
    const [viewFront, setViewFront] = useState(true);
    const [selectedExercise, setSelectedExercise] = useState(null);
    const [exercises, setExercises] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchData();
    }, []);

    const fetchData = async () => {
        try {
            setLoading(true);

            // Fetch muscle volume data
            const volumeRes = await api.get('/stats/volume-by-muscle?days=7');
            const volumeData = volumeRes.data.data || {};

            // Convert to intensity (0-100)
            const maxVolume = Math.max(...Object.values(volumeData), 1);
            const intensityData = {};
            Object.entries(volumeData).forEach(([muscle, volume]) => {
                intensityData[muscle.toLowerCase()] = (volume / maxVolume) * 100;
            });
            setMuscleData(intensityData);

            // Fetch exercises
            const exRes = await api.get('/exercises?limit=20');
            setExercises(exRes.data.data || SAMPLE_EXERCISES);
            if (exRes.data.data?.length > 0) {
                setSelectedExercise(exRes.data.data[0]);
            }
        } catch (error) {
            console.error('Failed to fetch data:', error);
            setMuscleData(SAMPLE_MUSCLE_DATA);
            setExercises(SAMPLE_EXERCISES);
            setSelectedExercise(SAMPLE_EXERCISES[0]);
        } finally {
            setLoading(false);
        }
    };

    const handleMuscleClick = (muscle, label) => {
        console.log('Clicked muscle:', muscle, label);
        // Could filter exercises by this muscle
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div>
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Analytics</h1>
                <p className="text-gray-600 dark:text-gray-400">Detailed training insights</p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Muscle Heatmap */}
                <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="font-semibold text-gray-900 dark:text-white flex items-center gap-2">
                            <Flame className="h-5 w-5 text-orange-500" />
                            Muscle Training Heatmap
                        </h3>
                        <div className="flex gap-2">
                            {/* 2D/3D Toggle */}
                            <div className="flex gap-1 bg-gray-100 dark:bg-gray-700 p-1 rounded-lg">
                                <button
                                    onClick={() => setView3D(true)}
                                    className={`p-1.5 rounded transition ${view3D ? 'bg-white dark:bg-gray-600 shadow-sm' : ''}`}
                                    title="3D View"
                                >
                                    <Box className="h-4 w-4 text-gray-600 dark:text-gray-300" />
                                </button>
                                <button
                                    onClick={() => setView3D(false)}
                                    className={`p-1.5 rounded transition ${!view3D ? 'bg-white dark:bg-gray-600 shadow-sm' : ''}`}
                                    title="2D View"
                                >
                                    <Grid className="h-4 w-4 text-gray-600 dark:text-gray-300" />
                                </button>
                            </div>
                            {/* Front/Back Toggle (only for 2D) */}
                            {!view3D && (
                                <div className="flex gap-1">
                                    <button
                                        onClick={() => setViewFront(true)}
                                        className={`px-3 py-1 text-sm rounded transition ${viewFront ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400' : 'text-gray-500'}`}
                                    >
                                        Front
                                    </button>
                                    <button
                                        onClick={() => setViewFront(false)}
                                        className={`px-3 py-1 text-sm rounded transition ${!viewFront ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400' : 'text-gray-500'}`}
                                    >
                                        Back
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
                        Last 7 days training intensity {view3D && '• Drag to rotate'}
                    </p>

                    {loading ? (
                        <div className="h-64 bg-gray-100 dark:bg-gray-700 rounded animate-pulse"></div>
                    ) : view3D ? (
                        <Suspense fallback={<div className="h-[500px] bg-gray-100 dark:bg-gray-700 rounded animate-pulse flex items-center justify-center text-gray-500">Loading 3D...</div>}>
                            <Body3D
                                muscleData={muscleData}
                                onMuscleClick={handleMuscleClick}
                            />
                        </Suspense>
                    ) : (
                        <MuscleHeatmap
                            muscleData={muscleData}
                            isFront={viewFront}
                            onMuscleClick={handleMuscleClick}
                        />
                    )}
                </div>

                {/* Strength Curves */}
                <div>
                    <div className="mb-4">
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                            Select Exercise
                        </label>
                        <div className="relative">
                            <select
                                value={selectedExercise?.id || ''}
                                onChange={(e) => {
                                    const ex = exercises.find(ex => ex.id === e.target.value);
                                    setSelectedExercise(ex);
                                }}
                                className="w-full px-4 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white appearance-none pr-10"
                            >
                                {exercises.map(ex => (
                                    <option key={ex.id} value={ex.id}>{ex.name}</option>
                                ))}
                            </select>
                            <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400 pointer-events-none" />
                        </div>
                    </div>

                    {selectedExercise && (
                        <StrengthCurves
                            exerciseId={selectedExercise.id}
                            exerciseName={selectedExercise.name}
                        />
                    )}
                </div>
            </div>

            {/* Barbell Standards */}
            <BarbellStandards />
        </div>
    );
};

// Sample data
const SAMPLE_MUSCLE_DATA = {
    chest: 80,
    shoulders_front: 60,
    biceps: 45,
    abs: 30,
    quads: 90,
    traps: 50,
    lats: 70,
    triceps: 55,
    glutes: 85,
    hamstrings: 75,
    calves: 40,
};

const SAMPLE_EXERCISES = [
    { id: '1', name: 'Bench Press' },
    { id: '2', name: 'Squat' },
    { id: '3', name: 'Deadlift' },
    { id: '4', name: 'Overhead Press' },
    { id: '5', name: 'Barbell Row' },
];

export default Analytics;

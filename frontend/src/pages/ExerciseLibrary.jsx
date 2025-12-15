import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Search,
    Filter,
    Dumbbell,
    ChevronRight,
    X,
    Target,
    Zap,
    Plus
} from 'lucide-react';
import api from '../services/api';

const CATEGORIES = [
    'All', 'CHEST', 'BACK', 'SHOULDERS', 'LEGS', 'ARMS', 'CORE', 'CARDIO', 'FULL_BODY'
];

const EQUIPMENT = [
    'All', 'barbell', 'dumbbells', 'machine', 'cable', 'bodyweight', 'kettlebell', 'bands'
];

const DIFFICULTIES = ['All', 'BEGINNER', 'INTERMEDIATE', 'ADVANCED'];

const ExerciseLibrary = () => {
    const [exercises, setExercises] = useState([]);
    const [filteredExercises, setFilteredExercises] = useState([]);
    const [loading, setLoading] = useState(true);
    const [searchQuery, setSearchQuery] = useState('');
    const [selectedCategory, setSelectedCategory] = useState('All');
    const [selectedEquipment, setSelectedEquipment] = useState('All');
    const [selectedDifficulty, setSelectedDifficulty] = useState('All');
    const [showFilters, setShowFilters] = useState(false);
    const [selectedExercise, setSelectedExercise] = useState(null);

    useEffect(() => {
        fetchExercises();
    }, []);

    useEffect(() => {
        filterExercises();
    }, [exercises, searchQuery, selectedCategory, selectedEquipment, selectedDifficulty]);

    const fetchExercises = async () => {
        try {
            setLoading(true);
            const response = await api.get('/exercises');
            setExercises(response.data.data || []);
        } catch (error) {
            console.error('Failed to fetch exercises:', error);
            // Use sample data if API fails
            setExercises(SAMPLE_EXERCISES);
        } finally {
            setLoading(false);
        }
    };

    const filterExercises = () => {
        let filtered = [...exercises];

        if (searchQuery) {
            filtered = filtered.filter(ex =>
                ex.name.toLowerCase().includes(searchQuery.toLowerCase())
            );
        }

        if (selectedCategory !== 'All') {
            filtered = filtered.filter(ex => ex.category === selectedCategory);
        }

        if (selectedEquipment !== 'All') {
            filtered = filtered.filter(ex =>
                ex.equipment?.some(eq => eq.toLowerCase().includes(selectedEquipment.toLowerCase()))
            );
        }

        if (selectedDifficulty !== 'All') {
            filtered = filtered.filter(ex => ex.difficulty === selectedDifficulty);
        }

        setFilteredExercises(filtered);
    };

    const getMuscleColor = (muscle) => {
        const colors = {
            chest: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400',
            back: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400',
            shoulders: 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400',
            legs: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
            arms: 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400',
            core: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400',
            default: 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300'
        };
        return colors[muscle?.toLowerCase()] || colors.default;
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Exercise Library</h1>
                    <p className="text-gray-600 dark:text-gray-400">Browse and search exercises</p>
                </div>
                <div className="text-sm text-gray-500 dark:text-gray-400">
                    {filteredExercises.length} exercises found
                </div>
            </div>

            {/* Search and Filters */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm space-y-4">
                <div className="flex gap-3">
                    <div className="flex-1 relative">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" />
                        <input
                            type="text"
                            placeholder="Search exercises..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="w-full pl-10 pr-4 py-2.5 border border-gray-200 dark:border-gray-700 rounded-lg bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        />
                    </div>
                    <button
                        onClick={() => setShowFilters(!showFilters)}
                        className={`px-4 py-2.5 rounded-lg flex items-center gap-2 transition ${showFilters
                                ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400'
                                : 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300'
                            }`}
                    >
                        <Filter className="h-5 w-5" />
                        <span className="hidden sm:inline">Filters</span>
                    </button>
                </div>

                {/* Filter Options */}
                <AnimatePresence>
                    {showFilters && (
                        <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            className="overflow-hidden"
                        >
                            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Category</label>
                                    <select
                                        value={selectedCategory}
                                        onChange={(e) => setSelectedCategory(e.target.value)}
                                        className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white"
                                    >
                                        {CATEGORIES.map(cat => (
                                            <option key={cat} value={cat}>{cat}</option>
                                        ))}
                                    </select>
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Equipment</label>
                                    <select
                                        value={selectedEquipment}
                                        onChange={(e) => setSelectedEquipment(e.target.value)}
                                        className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white"
                                    >
                                        {EQUIPMENT.map(eq => (
                                            <option key={eq} value={eq}>{eq}</option>
                                        ))}
                                    </select>
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Difficulty</label>
                                    <select
                                        value={selectedDifficulty}
                                        onChange={(e) => setSelectedDifficulty(e.target.value)}
                                        className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white"
                                    >
                                        {DIFFICULTIES.map(diff => (
                                            <option key={diff} value={diff}>{diff}</option>
                                        ))}
                                    </select>
                                </div>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>

            {/* Category Chips */}
            <div className="flex gap-2 overflow-x-auto pb-2 scrollbar-hide">
                {CATEGORIES.map(cat => (
                    <button
                        key={cat}
                        onClick={() => setSelectedCategory(cat)}
                        className={`px-4 py-2 rounded-full whitespace-nowrap text-sm font-medium transition ${selectedCategory === cat
                                ? 'bg-blue-600 text-white'
                                : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
                            }`}
                    >
                        {cat}
                    </button>
                ))}
            </div>

            {/* Exercise Grid */}
            {loading ? (
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                    {[1, 2, 3, 4, 5, 6].map(i => (
                        <div key={i} className="bg-white dark:bg-gray-800 rounded-xl p-4 animate-pulse">
                            <div className="h-32 bg-gray-200 dark:bg-gray-700 rounded-lg mb-4"></div>
                            <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4 mb-2"></div>
                            <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-1/2"></div>
                        </div>
                    ))}
                </div>
            ) : (
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                    {filteredExercises.map((exercise, index) => (
                        <motion.div
                            key={exercise.id || index}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: index * 0.05 }}
                            onClick={() => setSelectedExercise(exercise)}
                            className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm hover:shadow-md transition cursor-pointer group"
                        >
                            {/* Exercise Image */}
                            <div className="h-32 bg-gradient-to-br from-gray-100 to-gray-200 dark:from-gray-700 dark:to-gray-800 rounded-lg mb-4 flex items-center justify-center overflow-hidden">
                                {exercise.imageUrl ? (
                                    <img src={exercise.imageUrl} alt={exercise.name} className="w-full h-full object-cover" />
                                ) : (
                                    <Dumbbell className="h-12 w-12 text-gray-400" />
                                )}
                            </div>

                            {/* Exercise Info */}
                            <div className="space-y-2">
                                <div className="flex items-start justify-between">
                                    <h3 className="font-semibold text-gray-900 dark:text-white group-hover:text-blue-600 dark:group-hover:text-blue-400 transition">
                                        {exercise.name}
                                    </h3>
                                    <ChevronRight className="h-5 w-5 text-gray-400 group-hover:text-blue-600 transition" />
                                </div>

                                {/* Muscles */}
                                <div className="flex flex-wrap gap-1">
                                    {exercise.primaryMuscles?.slice(0, 2).map(muscle => (
                                        <span key={muscle} className={`px-2 py-0.5 rounded text-xs ${getMuscleColor(muscle)}`}>
                                            {muscle}
                                        </span>
                                    ))}
                                </div>

                                {/* Meta */}
                                <div className="flex items-center gap-3 text-sm text-gray-500 dark:text-gray-400">
                                    <span className="flex items-center gap-1">
                                        <Target className="h-4 w-4" />
                                        {exercise.category}
                                    </span>
                                    <span className="flex items-center gap-1">
                                        <Zap className="h-4 w-4" />
                                        {exercise.difficulty}
                                    </span>
                                </div>
                            </div>
                        </motion.div>
                    ))}
                </div>
            )}

            {/* Exercise Detail Modal */}
            <AnimatePresence>
                {selectedExercise && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4"
                        onClick={() => setSelectedExercise(null)}
                    >
                        <motion.div
                            initial={{ scale: 0.9, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            exit={{ scale: 0.9, opacity: 0 }}
                            onClick={(e) => e.stopPropagation()}
                            className="bg-white dark:bg-gray-800 rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto"
                        >
                            <div className="p-6 space-y-6">
                                {/* Header */}
                                <div className="flex justify-between items-start">
                                    <h2 className="text-2xl font-bold text-gray-900 dark:text-white">{selectedExercise.name}</h2>
                                    <button
                                        onClick={() => setSelectedExercise(null)}
                                        className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg"
                                    >
                                        <X className="h-5 w-5 text-gray-500" />
                                    </button>
                                </div>

                                {/* Image */}
                                <div className="h-48 bg-gradient-to-br from-gray-100 to-gray-200 dark:from-gray-700 dark:to-gray-800 rounded-xl flex items-center justify-center">
                                    {selectedExercise.imageUrl ? (
                                        <img src={selectedExercise.imageUrl} alt={selectedExercise.name} className="w-full h-full object-cover rounded-xl" />
                                    ) : (
                                        <Dumbbell className="h-16 w-16 text-gray-400" />
                                    )}
                                </div>

                                {/* Details */}
                                <div className="grid grid-cols-2 gap-4">
                                    <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
                                        <div className="text-sm text-gray-500 dark:text-gray-400">Category</div>
                                        <div className="font-semibold text-gray-900 dark:text-white">{selectedExercise.category}</div>
                                    </div>
                                    <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
                                        <div className="text-sm text-gray-500 dark:text-gray-400">Difficulty</div>
                                        <div className="font-semibold text-gray-900 dark:text-white">{selectedExercise.difficulty}</div>
                                    </div>
                                </div>

                                {/* Muscles */}
                                <div>
                                    <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Target Muscles</h3>
                                    <div className="flex flex-wrap gap-2">
                                        {selectedExercise.primaryMuscles?.map(muscle => (
                                            <span key={muscle} className={`px-3 py-1 rounded-full text-sm ${getMuscleColor(muscle)}`}>
                                                {muscle} (primary)
                                            </span>
                                        ))}
                                        {selectedExercise.secondaryMuscles?.map(muscle => (
                                            <span key={muscle} className="px-3 py-1 rounded-full text-sm bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300">
                                                {muscle}
                                            </span>
                                        ))}
                                    </div>
                                </div>

                                {/* Equipment */}
                                {selectedExercise.equipment?.length > 0 && (
                                    <div>
                                        <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Equipment</h3>
                                        <div className="flex flex-wrap gap-2">
                                            {selectedExercise.equipment?.map(eq => (
                                                <span key={eq} className="px-3 py-1 rounded-full text-sm bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400">
                                                    {eq}
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {/* Instructions */}
                                {selectedExercise.instructions && (
                                    <div>
                                        <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Instructions</h3>
                                        <p className="text-gray-600 dark:text-gray-400">{selectedExercise.instructions}</p>
                                    </div>
                                )}

                                {/* Add to Workout Button */}
                                <button className="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-semibold flex items-center justify-center gap-2 transition">
                                    <Plus className="h-5 w-5" />
                                    Add to Current Workout
                                </button>
                            </div>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

// Sample data for when API is unavailable
const SAMPLE_EXERCISES = [
    { id: '1', name: 'Bench Press', category: 'CHEST', primaryMuscles: ['chest'], secondaryMuscles: ['triceps', 'shoulders'], equipment: ['barbell', 'bench'], difficulty: 'INTERMEDIATE', instructions: 'Lie on a flat bench and press the barbell up from chest level.' },
    { id: '2', name: 'Squat', category: 'LEGS', primaryMuscles: ['quads', 'glutes'], secondaryMuscles: ['hamstrings', 'core'], equipment: ['barbell', 'squat rack'], difficulty: 'INTERMEDIATE', instructions: 'Stand with feet shoulder-width apart, squat down until thighs are parallel to ground.' },
    { id: '3', name: 'Deadlift', category: 'BACK', primaryMuscles: ['back', 'hamstrings'], secondaryMuscles: ['glutes', 'core'], equipment: ['barbell'], difficulty: 'ADVANCED', instructions: 'Bend at hips and knees to grab barbell, then stand up straight.' },
    { id: '4', name: 'Pull-up', category: 'BACK', primaryMuscles: ['lats', 'back'], secondaryMuscles: ['biceps'], equipment: ['pull-up bar'], difficulty: 'INTERMEDIATE', instructions: 'Hang from bar and pull yourself up until chin is above bar.' },
    { id: '5', name: 'Overhead Press', category: 'SHOULDERS', primaryMuscles: ['shoulders'], secondaryMuscles: ['triceps'], equipment: ['barbell'], difficulty: 'INTERMEDIATE', instructions: 'Press barbell overhead from shoulder level.' },
    { id: '6', name: 'Bicep Curl', category: 'ARMS', primaryMuscles: ['biceps'], secondaryMuscles: [], equipment: ['dumbbells'], difficulty: 'BEGINNER', instructions: 'Curl dumbbells from sides up to shoulders.' },
];

export default ExerciseLibrary;

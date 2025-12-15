import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import {
    Play,
    Pause,
    Plus,
    Check,
    X,
    Timer,
    Dumbbell,
    ChevronDown,
    ChevronUp,
    Trash2,
    Save,
    Trophy,
    RotateCcw
} from 'lucide-react';
import api from '../services/api';

const WorkoutLogger = () => {
    const navigate = useNavigate();
    const [workoutName, setWorkoutName] = useState('');
    const [exercises, setExercises] = useState([]);
    const [workoutStartTime, setWorkoutStartTime] = useState(null);
    const [isWorkoutActive, setIsWorkoutActive] = useState(false);
    const [showExercisePicker, setShowExercisePicker] = useState(false);
    const [restTimer, setRestTimer] = useState(0);
    const [isTimerRunning, setIsTimerRunning] = useState(false);
    const [restDuration, setRestDuration] = useState(90); // Default 90 seconds
    const [saving, setSaving] = useState(false);
    const [newPRs, setNewPRs] = useState([]);

    // Rest Timer Effect
    useEffect(() => {
        let interval;
        if (isTimerRunning && restTimer > 0) {
            interval = setInterval(() => {
                setRestTimer(prev => {
                    if (prev <= 1) {
                        setIsTimerRunning(false);
                        // Play notification sound or vibrate
                        if ('vibrate' in navigator) {
                            navigator.vibrate(200);
                        }
                        return 0;
                    }
                    return prev - 1;
                });
            }, 1000);
        }
        return () => clearInterval(interval);
    }, [isTimerRunning, restTimer]);

    const startWorkout = () => {
        setWorkoutStartTime(new Date());
        setIsWorkoutActive(true);
        if (!workoutName) {
            const date = new Date();
            setWorkoutName(`Workout ${date.toLocaleDateString()}`);
        }
    };

    const startRestTimer = () => {
        setRestTimer(restDuration);
        setIsTimerRunning(true);
    };

    const stopRestTimer = () => {
        setIsTimerRunning(false);
        setRestTimer(0);
    };

    const addExercise = (exercise) => {
        setExercises(prev => [...prev, {
            ...exercise,
            exerciseId: exercise.id,
            exerciseName: exercise.name,
            sets: [createEmptySet(1)],
            expanded: true
        }]);
        setShowExercisePicker(false);
    };

    const createEmptySet = (setNumber) => ({
        setNumber,
        weight: '',
        reps: '',
        isWarmup: false,
        completed: false,
        isPR: false
    });

    const addSet = (exerciseIndex) => {
        setExercises(prev => {
            const updated = [...prev];
            const newSetNumber = updated[exerciseIndex].sets.length + 1;
            updated[exerciseIndex].sets.push(createEmptySet(newSetNumber));
            return updated;
        });
    };

    const updateSet = (exerciseIndex, setIndex, field, value) => {
        setExercises(prev => {
            const updated = [...prev];
            updated[exerciseIndex].sets[setIndex][field] = value;
            return updated;
        });
    };

    const removeSet = (exerciseIndex, setIndex) => {
        setExercises(prev => {
            const updated = [...prev];
            updated[exerciseIndex].sets = updated[exerciseIndex].sets.filter((_, i) => i !== setIndex);
            // Renumber sets
            updated[exerciseIndex].sets.forEach((set, i) => {
                set.setNumber = i + 1;
            });
            return updated;
        });
    };

    const toggleSetComplete = (exerciseIndex, setIndex) => {
        setExercises(prev => {
            const updated = [...prev];
            updated[exerciseIndex].sets[setIndex].completed = !updated[exerciseIndex].sets[setIndex].completed;
            return updated;
        });
        // Start rest timer after completing a set
        if (!exercises[exerciseIndex].sets[setIndex].completed) {
            startRestTimer();
        }
    };

    const removeExercise = (index) => {
        setExercises(prev => prev.filter((_, i) => i !== index));
    };

    const toggleExerciseExpanded = (index) => {
        setExercises(prev => {
            const updated = [...prev];
            updated[index].expanded = !updated[index].expanded;
            return updated;
        });
    };

    const formatTime = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    const getWorkoutDuration = () => {
        if (!workoutStartTime) return '0:00';
        const now = new Date();
        const diff = Math.floor((now - workoutStartTime) / 1000);
        const hours = Math.floor(diff / 3600);
        const mins = Math.floor((diff % 3600) / 60);
        const secs = diff % 60;
        if (hours > 0) {
            return `${hours}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        }
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    const saveWorkout = async () => {
        if (exercises.length === 0) {
            alert('Add at least one exercise to save the workout');
            return;
        }

        setSaving(true);
        try {
            const workoutLog = {
                name: workoutName,
                startTime: workoutStartTime?.toISOString(),
                endTime: new Date().toISOString(),
                exercises: exercises.map((ex, i) => ({
                    exerciseId: ex.exerciseId,
                    exerciseName: ex.exerciseName,
                    order: i + 1,
                    sets: ex.sets.map(set => ({
                        ...set,
                        weight: parseFloat(set.weight) || 0,
                        reps: parseInt(set.reps) || 0
                    }))
                }))
            };

            const response = await api.post('/workout-logs', workoutLog);

            // Check for new PRs in response
            if (response.data.data?.newPRs?.length > 0) {
                setNewPRs(response.data.data.newPRs);
            }

            navigate('/workouts/history');
        } catch (error) {
            console.error('Failed to save workout:', error);
            alert('Failed to save workout. Please try again.');
        } finally {
            setSaving(false);
        }
    };

    const discardWorkout = () => {
        if (confirm('Are you sure you want to discard this workout?')) {
            setExercises([]);
            setWorkoutStartTime(null);
            setIsWorkoutActive(false);
            setWorkoutName('');
        }
    };

    return (
        <div className="space-y-6 pb-24">
            {/* Header */}
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                <div>
                    <input
                        type="text"
                        value={workoutName}
                        onChange={(e) => setWorkoutName(e.target.value)}
                        placeholder="Workout Name"
                        className="text-2xl font-bold bg-transparent border-none focus:outline-none text-gray-900 dark:text-white placeholder-gray-400"
                    />
                    {isWorkoutActive && (
                        <p className="text-blue-600 dark:text-blue-400 font-mono text-lg">
                            ⏱ {getWorkoutDuration()}
                        </p>
                    )}
                </div>

                {!isWorkoutActive ? (
                    <button
                        onClick={startWorkout}
                        className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-semibold flex items-center gap-2 transition"
                    >
                        <Play className="h-5 w-5" />
                        Start Workout
                    </button>
                ) : (
                    <div className="flex gap-2">
                        <button
                            onClick={discardWorkout}
                            className="px-4 py-2 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-xl flex items-center gap-2 hover:bg-gray-200 dark:hover:bg-gray-700 transition"
                        >
                            <X className="h-5 w-5" />
                            Discard
                        </button>
                        <button
                            onClick={saveWorkout}
                            disabled={saving}
                            className="px-6 py-3 bg-green-600 hover:bg-green-700 text-white rounded-xl font-semibold flex items-center gap-2 transition disabled:opacity-50"
                        >
                            <Save className="h-5 w-5" />
                            {saving ? 'Saving...' : 'Finish'}
                        </button>
                    </div>
                )}
            </div>

            {/* Rest Timer */}
            <AnimatePresence>
                {(isTimerRunning || restTimer > 0) && (
                    <motion.div
                        initial={{ opacity: 0, y: -20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        className="bg-blue-600 dark:bg-blue-700 rounded-xl p-4 text-white"
                    >
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-3">
                                <Timer className="h-6 w-6" />
                                <span className="font-medium">Rest Timer</span>
                            </div>
                            <div className="flex items-center gap-4">
                                <span className="text-3xl font-mono font-bold">{formatTime(restTimer)}</span>
                                <div className="flex gap-2">
                                    <button
                                        onClick={() => setIsTimerRunning(!isTimerRunning)}
                                        className="p-2 bg-white/20 rounded-lg hover:bg-white/30 transition"
                                    >
                                        {isTimerRunning ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5" />}
                                    </button>
                                    <button
                                        onClick={stopRestTimer}
                                        className="p-2 bg-white/20 rounded-lg hover:bg-white/30 transition"
                                    >
                                        <RotateCcw className="h-5 w-5" />
                                    </button>
                                </div>
                            </div>
                        </div>
                        <div className="mt-3 flex gap-2">
                            {[30, 60, 90, 120, 180].map(sec => (
                                <button
                                    key={sec}
                                    onClick={() => { setRestDuration(sec); setRestTimer(sec); setIsTimerRunning(true); }}
                                    className={`px-3 py-1 rounded-lg text-sm ${restDuration === sec ? 'bg-white text-blue-600' : 'bg-white/20 hover:bg-white/30'
                                        }`}
                                >
                                    {sec < 60 ? `${sec}s` : `${sec / 60}m`}
                                </button>
                            ))}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Exercises */}
            <div className="space-y-4">
                {exercises.map((exercise, exerciseIndex) => (
                    <motion.div
                        key={exerciseIndex}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="bg-white dark:bg-gray-800 rounded-xl shadow-sm overflow-hidden"
                    >
                        {/* Exercise Header */}
                        <div
                            onClick={() => toggleExerciseExpanded(exerciseIndex)}
                            className="p-4 flex items-center justify-between cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700/50"
                        >
                            <div className="flex items-center gap-3">
                                <div className="h-10 w-10 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center">
                                    <Dumbbell className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                                </div>
                                <div>
                                    <h3 className="font-semibold text-gray-900 dark:text-white">{exercise.exerciseName}</h3>
                                    <p className="text-sm text-gray-500 dark:text-gray-400">
                                        {exercise.sets.filter(s => s.completed).length}/{exercise.sets.length} sets completed
                                    </p>
                                </div>
                            </div>
                            <div className="flex items-center gap-2">
                                <button
                                    onClick={(e) => { e.stopPropagation(); removeExercise(exerciseIndex); }}
                                    className="p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg"
                                >
                                    <Trash2 className="h-4 w-4" />
                                </button>
                                {exercise.expanded ? <ChevronUp className="h-5 w-5 text-gray-400" /> : <ChevronDown className="h-5 w-5 text-gray-400" />}
                            </div>
                        </div>

                        {/* Sets */}
                        <AnimatePresence>
                            {exercise.expanded && (
                                <motion.div
                                    initial={{ height: 0 }}
                                    animate={{ height: 'auto' }}
                                    exit={{ height: 0 }}
                                    className="overflow-hidden"
                                >
                                    <div className="px-4 pb-4 space-y-2">
                                        {/* Header Row */}
                                        <div className="grid grid-cols-12 gap-2 text-sm text-gray-500 dark:text-gray-400 font-medium px-2">
                                            <div className="col-span-2">Set</div>
                                            <div className="col-span-3">Weight</div>
                                            <div className="col-span-3">Reps</div>
                                            <div className="col-span-2">Warmup</div>
                                            <div className="col-span-2"></div>
                                        </div>

                                        {/* Set Rows */}
                                        {exercise.sets.map((set, setIndex) => (
                                            <div
                                                key={setIndex}
                                                className={`grid grid-cols-12 gap-2 items-center p-2 rounded-lg ${set.completed ? 'bg-green-50 dark:bg-green-900/20' : 'bg-gray-50 dark:bg-gray-900'
                                                    }`}
                                            >
                                                <div className="col-span-2 font-medium text-gray-900 dark:text-white">
                                                    {set.isPR && <Trophy className="h-4 w-4 text-yellow-500 inline mr-1" />}
                                                    {set.setNumber}
                                                </div>
                                                <div className="col-span-3">
                                                    <input
                                                        type="number"
                                                        value={set.weight}
                                                        onChange={(e) => updateSet(exerciseIndex, setIndex, 'weight', e.target.value)}
                                                        placeholder="0"
                                                        className="w-full px-2 py-1.5 text-sm border border-gray-200 dark:border-gray-700 rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
                                                    />
                                                </div>
                                                <div className="col-span-3">
                                                    <input
                                                        type="number"
                                                        value={set.reps}
                                                        onChange={(e) => updateSet(exerciseIndex, setIndex, 'reps', e.target.value)}
                                                        placeholder="0"
                                                        className="w-full px-2 py-1.5 text-sm border border-gray-200 dark:border-gray-700 rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
                                                    />
                                                </div>
                                                <div className="col-span-2 flex justify-center">
                                                    <input
                                                        type="checkbox"
                                                        checked={set.isWarmup}
                                                        onChange={(e) => updateSet(exerciseIndex, setIndex, 'isWarmup', e.target.checked)}
                                                        className="h-4 w-4 rounded border-gray-300 dark:border-gray-600"
                                                    />
                                                </div>
                                                <div className="col-span-2 flex gap-1 justify-end">
                                                    <button
                                                        onClick={() => toggleSetComplete(exerciseIndex, setIndex)}
                                                        className={`p-1.5 rounded ${set.completed
                                                                ? 'bg-green-500 text-white'
                                                                : 'bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-400 hover:bg-green-100'
                                                            }`}
                                                    >
                                                        <Check className="h-4 w-4" />
                                                    </button>
                                                    <button
                                                        onClick={() => removeSet(exerciseIndex, setIndex)}
                                                        className="p-1.5 rounded bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-400 hover:bg-red-100 hover:text-red-500"
                                                    >
                                                        <X className="h-4 w-4" />
                                                    </button>
                                                </div>
                                            </div>
                                        ))}

                                        {/* Add Set Button */}
                                        <button
                                            onClick={() => addSet(exerciseIndex)}
                                            className="w-full py-2 border-2 border-dashed border-gray-200 dark:border-gray-700 rounded-lg text-gray-500 dark:text-gray-400 hover:border-blue-500 hover:text-blue-500 transition flex items-center justify-center gap-2"
                                        >
                                            <Plus className="h-4 w-4" />
                                            Add Set
                                        </button>
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </motion.div>
                ))}

                {/* Add Exercise Button */}
                <button
                    onClick={() => setShowExercisePicker(true)}
                    className="w-full py-4 border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-xl text-gray-500 dark:text-gray-400 hover:border-blue-500 hover:text-blue-500 transition flex items-center justify-center gap-2"
                >
                    <Plus className="h-5 w-5" />
                    Add Exercise
                </button>
            </div>

            {/* Exercise Picker Modal */}
            <AnimatePresence>
                {showExercisePicker && (
                    <ExercisePickerModal
                        onSelect={addExercise}
                        onClose={() => setShowExercisePicker(false)}
                    />
                )}
            </AnimatePresence>
        </div>
    );
};

// Exercise Picker Modal Component
const ExercisePickerModal = ({ onSelect, onClose }) => {
    const [search, setSearch] = useState('');
    const [exercises, setExercises] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchExercises();
    }, []);

    const fetchExercises = async () => {
        try {
            const response = await api.get('/exercises');
            setExercises(response.data.data || SAMPLE_EXERCISES);
        } catch (error) {
            setExercises(SAMPLE_EXERCISES);
        } finally {
            setLoading(false);
        }
    };

    const filteredExercises = exercises.filter(ex =>
        ex.name.toLowerCase().includes(search.toLowerCase())
    );

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 z-50 flex items-end sm:items-center justify-center"
            onClick={onClose}
        >
            <motion.div
                initial={{ y: '100%' }}
                animate={{ y: 0 }}
                exit={{ y: '100%' }}
                onClick={(e) => e.stopPropagation()}
                className="bg-white dark:bg-gray-800 w-full sm:max-w-lg sm:rounded-2xl rounded-t-2xl max-h-[80vh] overflow-hidden"
            >
                <div className="p-4 border-b border-gray-200 dark:border-gray-700">
                    <div className="flex items-center justify-between mb-4">
                        <h2 className="text-xl font-bold text-gray-900 dark:text-white">Add Exercise</h2>
                        <button onClick={onClose} className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg">
                            <X className="h-5 w-5 text-gray-500" />
                        </button>
                    </div>
                    <input
                        type="text"
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                        placeholder="Search exercises..."
                        className="w-full px-4 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white"
                    />
                </div>
                <div className="overflow-y-auto max-h-96">
                    {loading ? (
                        <div className="p-4 text-center text-gray-500">Loading...</div>
                    ) : (
                        filteredExercises.map((exercise, i) => (
                            <div
                                key={exercise.id || i}
                                onClick={() => onSelect(exercise)}
                                className="p-4 flex items-center gap-3 hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer border-b border-gray-100 dark:border-gray-700"
                            >
                                <div className="h-10 w-10 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center">
                                    <Dumbbell className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                                </div>
                                <div>
                                    <div className="font-medium text-gray-900 dark:text-white">{exercise.name}</div>
                                    <div className="text-sm text-gray-500 dark:text-gray-400">{exercise.category}</div>
                                </div>
                            </div>
                        ))
                    )}
                </div>
            </motion.div>
        </motion.div>
    );
};

const SAMPLE_EXERCISES = [
    { id: '1', name: 'Bench Press', category: 'CHEST' },
    { id: '2', name: 'Squat', category: 'LEGS' },
    { id: '3', name: 'Deadlift', category: 'BACK' },
    { id: '4', name: 'Pull-up', category: 'BACK' },
    { id: '5', name: 'Overhead Press', category: 'SHOULDERS' },
    { id: '6', name: 'Barbell Row', category: 'BACK' },
];

export default WorkoutLogger;

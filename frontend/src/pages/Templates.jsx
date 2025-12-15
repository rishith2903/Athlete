import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import {
    Plus,
    Play,
    Edit,
    Trash2,
    Copy,
    Clock,
    Dumbbell,
    ChevronDown,
    ChevronUp,
    X,
    Save,
    Folder
} from 'lucide-react';
import api from '../services/api';

const CATEGORIES = ['PUSH', 'PULL', 'LEGS', 'UPPER', 'LOWER', 'FULL_BODY', 'CARDIO', 'CUSTOM'];

const Templates = () => {
    const navigate = useNavigate();
    const [templates, setTemplates] = useState([]);
    const [loading, setLoading] = useState(true);
    const [showCreateModal, setShowCreateModal] = useState(false);
    const [selectedTemplate, setSelectedTemplate] = useState(null);
    const [expandedTemplates, setExpandedTemplates] = useState({});

    useEffect(() => {
        fetchTemplates();
    }, []);

    const fetchTemplates = async () => {
        try {
            setLoading(true);
            const response = await api.get('/templates');
            setTemplates(response.data.data || []);
        } catch (error) {
            console.error('Failed to fetch templates:', error);
            setTemplates(SAMPLE_TEMPLATES);
        } finally {
            setLoading(false);
        }
    };

    const startWorkout = async (templateId) => {
        try {
            const response = await api.post(`/templates/${templateId}/start`);
            // Navigate to workout logger with pre-filled data
            navigate('/workouts/log', { state: { workoutLog: response.data.data } });
        } catch (error) {
            console.error('Failed to start workout:', error);
            navigate('/workouts/log');
        }
    };

    const deleteTemplate = async (templateId) => {
        if (!confirm('Are you sure you want to delete this template?')) return;
        try {
            await api.delete(`/templates/${templateId}`);
            setTemplates(prev => prev.filter(t => t.id !== templateId));
        } catch (error) {
            console.error('Failed to delete template:', error);
        }
    };

    const toggleExpanded = (id) => {
        setExpandedTemplates(prev => ({ ...prev, [id]: !prev[id] }));
    };

    const getCategoryColor = (category) => {
        const colors = {
            PUSH: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400',
            PULL: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400',
            LEGS: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
            UPPER: 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400',
            LOWER: 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400',
            FULL_BODY: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400',
            CARDIO: 'bg-pink-100 text-pink-700 dark:bg-pink-900/30 dark:text-pink-400',
            CUSTOM: 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300',
        };
        return colors[category] || colors.CUSTOM;
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Workout Templates</h1>
                    <p className="text-gray-600 dark:text-gray-400">Save and reuse your favorite workouts</p>
                </div>
                <button
                    onClick={() => setShowCreateModal(true)}
                    className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-semibold flex items-center gap-2 transition"
                >
                    <Plus className="h-5 w-5" />
                    Create Template
                </button>
            </div>

            {/* Category Pills */}
            <div className="flex gap-2 overflow-x-auto pb-2">
                {CATEGORIES.map(cat => (
                    <span key={cat} className={`px-3 py-1.5 rounded-full text-sm font-medium ${getCategoryColor(cat)}`}>
                        {cat.replace('_', ' ')}
                    </span>
                ))}
            </div>

            {/* Templates Grid */}
            {loading ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {[1, 2, 3].map(i => (
                        <div key={i} className="bg-white dark:bg-gray-800 rounded-xl p-4 animate-pulse">
                            <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded w-1/2 mb-4"></div>
                            <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4 mb-2"></div>
                            <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/2"></div>
                        </div>
                    ))}
                </div>
            ) : templates.length === 0 ? (
                <div className="text-center py-12 bg-white dark:bg-gray-800 rounded-xl">
                    <Folder className="h-16 w-16 text-gray-300 dark:text-gray-600 mx-auto mb-4" />
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">No Templates Yet</h3>
                    <p className="text-gray-600 dark:text-gray-400 mb-4">Create your first workout template to get started</p>
                    <button
                        onClick={() => setShowCreateModal(true)}
                        className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium"
                    >
                        Create Template
                    </button>
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {templates.map((template, index) => (
                        <motion.div
                            key={template.id || index}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: index * 0.05 }}
                            className="bg-white dark:bg-gray-800 rounded-xl shadow-sm overflow-hidden"
                        >
                            {/* Template Header */}
                            <div className="p-4">
                                <div className="flex items-start justify-between mb-2">
                                    <h3 className="font-semibold text-gray-900 dark:text-white">{template.name}</h3>
                                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${getCategoryColor(template.category)}`}>
                                        {template.category}
                                    </span>
                                </div>

                                {template.description && (
                                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">{template.description}</p>
                                )}

                                <div className="flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
                                    <span className="flex items-center gap-1">
                                        <Dumbbell className="h-4 w-4" />
                                        {template.exercises?.length || 0} exercises
                                    </span>
                                    {template.estimatedDuration && (
                                        <span className="flex items-center gap-1">
                                            <Clock className="h-4 w-4" />
                                            {template.estimatedDuration} min
                                        </span>
                                    )}
                                </div>
                            </div>

                            {/* Exercises List (Collapsible) */}
                            <div
                                onClick={() => toggleExpanded(template.id)}
                                className="px-4 py-2 bg-gray-50 dark:bg-gray-700/50 border-t border-gray-100 dark:border-gray-700 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center justify-between"
                            >
                                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                                    View Exercises
                                </span>
                                {expandedTemplates[template.id] ? (
                                    <ChevronUp className="h-4 w-4 text-gray-500" />
                                ) : (
                                    <ChevronDown className="h-4 w-4 text-gray-500" />
                                )}
                            </div>

                            <AnimatePresence>
                                {expandedTemplates[template.id] && (
                                    <motion.div
                                        initial={{ height: 0 }}
                                        animate={{ height: 'auto' }}
                                        exit={{ height: 0 }}
                                        className="overflow-hidden"
                                    >
                                        <div className="px-4 py-2 bg-gray-50 dark:bg-gray-700/50 space-y-1">
                                            {template.exercises?.map((ex, i) => (
                                                <div key={i} className="flex items-center gap-2 text-sm">
                                                    <span className="w-5 h-5 rounded bg-gray-200 dark:bg-gray-600 flex items-center justify-center text-xs text-gray-600 dark:text-gray-300">
                                                        {i + 1}
                                                    </span>
                                                    <span className="text-gray-700 dark:text-gray-300">{ex.exerciseName}</span>
                                                    <span className="text-gray-500 dark:text-gray-400">
                                                        {ex.targetSets} × {ex.targetReps || '?'}
                                                    </span>
                                                </div>
                                            ))}
                                        </div>
                                    </motion.div>
                                )}
                            </AnimatePresence>

                            {/* Actions */}
                            <div className="p-3 flex gap-2 border-t border-gray-100 dark:border-gray-700">
                                <button
                                    onClick={() => startWorkout(template.id)}
                                    className="flex-1 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium flex items-center justify-center gap-2 transition"
                                >
                                    <Play className="h-4 w-4" />
                                    Start
                                </button>
                                <button
                                    onClick={() => setSelectedTemplate(template)}
                                    className="p-2 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600"
                                >
                                    <Edit className="h-4 w-4" />
                                </button>
                                <button
                                    onClick={() => deleteTemplate(template.id)}
                                    className="p-2 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 rounded-lg hover:bg-red-100 hover:text-red-600 dark:hover:bg-red-900/30 dark:hover:text-red-400"
                                >
                                    <Trash2 className="h-4 w-4" />
                                </button>
                            </div>
                        </motion.div>
                    ))}
                </div>
            )}

            {/* Create Template Modal */}
            <AnimatePresence>
                {showCreateModal && (
                    <CreateTemplateModal
                        onClose={() => setShowCreateModal(false)}
                        onSave={(template) => {
                            setTemplates(prev => [template, ...prev]);
                            setShowCreateModal(false);
                        }}
                    />
                )}
            </AnimatePresence>
        </div>
    );
};

// Create Template Modal
const CreateTemplateModal = ({ onClose, onSave }) => {
    const [name, setName] = useState('');
    const [category, setCategory] = useState('CUSTOM');
    const [description, setDescription] = useState('');
    const [exercises, setExercises] = useState([]);
    const [saving, setSaving] = useState(false);

    const addExercise = () => {
        setExercises(prev => [...prev, {
            exerciseName: '',
            targetSets: 3,
            targetReps: '8-12',
            restSeconds: 90
        }]);
    };

    const updateExercise = (index, field, value) => {
        setExercises(prev => {
            const updated = [...prev];
            updated[index][field] = value;
            return updated;
        });
    };

    const removeExercise = (index) => {
        setExercises(prev => prev.filter((_, i) => i !== index));
    };

    const handleSave = async () => {
        if (!name.trim()) {
            alert('Please enter a template name');
            return;
        }

        setSaving(true);
        try {
            const template = {
                name,
                category,
                description,
                exercises: exercises.map((ex, i) => ({
                    ...ex,
                    order: i + 1
                }))
            };

            const response = await api.post('/templates', template);
            onSave(response.data.data);
        } catch (error) {
            console.error('Failed to save template:', error);
            // Still add locally for demo
            onSave({ id: Date.now().toString(), name, category, description, exercises });
        } finally {
            setSaving(false);
        }
    };

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4"
            onClick={onClose}
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
                    <div className="flex justify-between items-center">
                        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Create Template</h2>
                        <button onClick={onClose} className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg">
                            <X className="h-5 w-5 text-gray-500" />
                        </button>
                    </div>

                    {/* Form */}
                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Template Name</label>
                            <input
                                type="text"
                                value={name}
                                onChange={(e) => setName(e.target.value)}
                                placeholder="e.g., Push Day, Upper Body Strength"
                                className="w-full px-4 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
                            />
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Category</label>
                                <select
                                    value={category}
                                    onChange={(e) => setCategory(e.target.value)}
                                    className="w-full px-4 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
                                >
                                    {CATEGORIES.map(cat => (
                                        <option key={cat} value={cat}>{cat.replace('_', ' ')}</option>
                                    ))}
                                </select>
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Description</label>
                                <input
                                    type="text"
                                    value={description}
                                    onChange={(e) => setDescription(e.target.value)}
                                    placeholder="Optional description"
                                    className="w-full px-4 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
                                />
                            </div>
                        </div>
                    </div>

                    {/* Exercises */}
                    <div>
                        <h3 className="font-semibold text-gray-900 dark:text-white mb-3">Exercises</h3>
                        <div className="space-y-3">
                            {exercises.map((ex, i) => (
                                <div key={i} className="flex gap-2 items-center p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
                                    <span className="w-6 text-sm text-gray-500">{i + 1}.</span>
                                    <input
                                        type="text"
                                        value={ex.exerciseName}
                                        onChange={(e) => updateExercise(i, 'exerciseName', e.target.value)}
                                        placeholder="Exercise name"
                                        className="flex-1 px-3 py-1.5 border border-gray-200 dark:border-gray-700 rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-white text-sm"
                                    />
                                    <input
                                        type="number"
                                        value={ex.targetSets}
                                        onChange={(e) => updateExercise(i, 'targetSets', parseInt(e.target.value))}
                                        className="w-16 px-2 py-1.5 border border-gray-200 dark:border-gray-700 rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-white text-sm text-center"
                                        placeholder="Sets"
                                    />
                                    <span className="text-gray-400">×</span>
                                    <input
                                        type="text"
                                        value={ex.targetReps}
                                        onChange={(e) => updateExercise(i, 'targetReps', e.target.value)}
                                        className="w-20 px-2 py-1.5 border border-gray-200 dark:border-gray-700 rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-white text-sm text-center"
                                        placeholder="Reps"
                                    />
                                    <button
                                        onClick={() => removeExercise(i)}
                                        className="p-1.5 text-gray-400 hover:text-red-500"
                                    >
                                        <X className="h-4 w-4" />
                                    </button>
                                </div>
                            ))}
                            <button
                                onClick={addExercise}
                                className="w-full py-2 border-2 border-dashed border-gray-200 dark:border-gray-700 rounded-lg text-gray-500 dark:text-gray-400 hover:border-blue-500 hover:text-blue-500 transition flex items-center justify-center gap-2"
                            >
                                <Plus className="h-4 w-4" />
                                Add Exercise
                            </button>
                        </div>
                    </div>

                    {/* Actions */}
                    <div className="flex gap-3">
                        <button
                            onClick={onClose}
                            className="flex-1 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg font-medium hover:bg-gray-200 dark:hover:bg-gray-600"
                        >
                            Cancel
                        </button>
                        <button
                            onClick={handleSave}
                            disabled={saving}
                            className="flex-1 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold flex items-center justify-center gap-2 disabled:opacity-50"
                        >
                            <Save className="h-5 w-5" />
                            {saving ? 'Saving...' : 'Save Template'}
                        </button>
                    </div>
                </div>
            </motion.div>
        </motion.div>
    );
};

// Sample data
const SAMPLE_TEMPLATES = [
    {
        id: '1',
        name: 'Push Day',
        category: 'PUSH',
        description: 'Chest, shoulders, and triceps',
        estimatedDuration: 60,
        exercises: [
            { exerciseName: 'Bench Press', targetSets: 4, targetReps: '6-8', order: 1 },
            { exerciseName: 'Overhead Press', targetSets: 4, targetReps: '8-10', order: 2 },
            { exerciseName: 'Incline Dumbbell Press', targetSets: 3, targetReps: '10-12', order: 3 },
            { exerciseName: 'Tricep Pushdowns', targetSets: 3, targetReps: '12-15', order: 4 },
        ]
    },
    {
        id: '2',
        name: 'Pull Day',
        category: 'PULL',
        description: 'Back and biceps',
        estimatedDuration: 55,
        exercises: [
            { exerciseName: 'Deadlift', targetSets: 4, targetReps: '5', order: 1 },
            { exerciseName: 'Pull-ups', targetSets: 4, targetReps: '8-10', order: 2 },
            { exerciseName: 'Barbell Row', targetSets: 4, targetReps: '8-10', order: 3 },
            { exerciseName: 'Bicep Curls', targetSets: 3, targetReps: '12-15', order: 4 },
        ]
    },
];

export default Templates;

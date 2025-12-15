import React, { useState, useRef } from 'react';
import { motion, Reorder, AnimatePresence } from 'framer-motion';
import {
    GripVertical,
    Link2,
    Unlink,
    Plus,
    Trash2,
    ChevronDown,
    ChevronUp,
    Timer,
    Zap
} from 'lucide-react';

/**
 * Superset/Circuit Builder Component
 * Groups exercises together with minimal rest indicators
 */
const SupersetBuilder = ({ exercises, onUpdate, onAddExercise }) => {
    const [groups, setGroups] = useState(() => {
        // Initialize groups from exercises
        return exercises.reduce((acc, ex, idx) => {
            const groupId = ex.supersetGroup || `single_${idx}`;
            if (!acc[groupId]) {
                acc[groupId] = { id: groupId, type: ex.supersetGroup ? 'superset' : 'single', exercises: [] };
            }
            acc[groupId].exercises.push(ex);
            return acc;
        }, {});
    });

    const handleCreateSuperset = (exerciseIds) => {
        const newGroupId = `superset_${Date.now()}`;
        const updatedExercises = exercises.map(ex => {
            if (exerciseIds.includes(ex.id)) {
                return { ...ex, supersetGroup: newGroupId };
            }
            return ex;
        });
        onUpdate(updatedExercises);
    };

    const handleBreakSuperset = (groupId) => {
        const updatedExercises = exercises.map(ex => {
            if (ex.supersetGroup === groupId) {
                return { ...ex, supersetGroup: null };
            }
            return ex;
        });
        onUpdate(updatedExercises);
    };

    return (
        <div className="space-y-4">
            {Object.values(groups).map((group, groupIndex) => (
                <ExerciseGroup
                    key={group.id}
                    group={group}
                    groupIndex={groupIndex}
                    onBreakSuperset={() => handleBreakSuperset(group.id)}
                    onUpdate={onUpdate}
                    allExercises={exercises}
                />
            ))}
        </div>
    );
};

/**
 * Exercise Group Component (Single or Superset)
 */
const ExerciseGroup = ({ group, groupIndex, onBreakSuperset, onUpdate, allExercises }) => {
    const [collapsed, setCollapsed] = useState(false);
    const isSuperset = group.type === 'superset' || group.exercises.length > 1;

    return (
        <motion.div
            layout
            className={`rounded-xl overflow-hidden ${isSuperset
                    ? 'border-2 border-purple-500 dark:border-purple-400 bg-purple-50 dark:bg-purple-900/20'
                    : 'bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700'
                }`}
        >
            {/* Superset Header */}
            {isSuperset && (
                <div className="flex items-center justify-between px-4 py-2 bg-purple-100 dark:bg-purple-900/40">
                    <div className="flex items-center gap-2 text-purple-700 dark:text-purple-300">
                        <Zap className="h-4 w-4" />
                        <span className="font-medium text-sm">
                            Superset ({group.exercises.length} exercises)
                        </span>
                        <span className="text-xs bg-purple-200 dark:bg-purple-800 px-2 py-0.5 rounded">
                            No rest between
                        </span>
                    </div>
                    <div className="flex items-center gap-1">
                        <button
                            onClick={() => setCollapsed(!collapsed)}
                            className="p-1 hover:bg-purple-200 dark:hover:bg-purple-800 rounded"
                        >
                            {collapsed ? <ChevronDown className="h-4 w-4" /> : <ChevronUp className="h-4 w-4" />}
                        </button>
                        <button
                            onClick={onBreakSuperset}
                            className="p-1 hover:bg-purple-200 dark:hover:bg-purple-800 rounded"
                            title="Break superset"
                        >
                            <Unlink className="h-4 w-4" />
                        </button>
                    </div>
                </div>
            )}

            {/* Exercises */}
            <AnimatePresence>
                {!collapsed && (
                    <motion.div
                        initial={false}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="divide-y divide-gray-100 dark:divide-gray-700"
                    >
                        {group.exercises.map((exercise, exIndex) => (
                            <ExerciseRow
                                key={exercise.id}
                                exercise={exercise}
                                exIndex={exIndex}
                                isInSuperset={isSuperset}
                                isLastInSuperset={exIndex === group.exercises.length - 1}
                            />
                        ))}
                    </motion.div>
                )}
            </AnimatePresence>
        </motion.div>
    );
};

/**
 * Individual Exercise Row
 */
const ExerciseRow = ({ exercise, exIndex, isInSuperset, isLastInSuperset }) => {
    return (
        <div className={`p-4 ${isInSuperset ? 'bg-white/50 dark:bg-gray-800/50' : ''}`}>
            <div className="flex items-center gap-3">
                {/* Drag Handle */}
                <div className="cursor-grab active:cursor-grabbing p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded">
                    <GripVertical className="h-5 w-5 text-gray-400" />
                </div>

                {/* Superset Connector */}
                {isInSuperset && !isLastInSuperset && (
                    <div className="absolute left-8 -bottom-3 w-0.5 h-6 bg-purple-400 z-10" />
                )}

                {/* Exercise Info */}
                <div className="flex-1">
                    <div className="font-medium text-gray-900 dark:text-white">{exercise.exerciseName || exercise.name}</div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">
                        {exercise.sets?.length || 0} sets • {exercise.targetMuscles?.join(', ') || 'Full body'}
                    </div>
                </div>

                {/* Warmup Badge */}
                {exercise.isWarmup && (
                    <span className="px-2 py-0.5 bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400 text-xs rounded-full">
                        Warm-up
                    </span>
                )}

                {/* Set Summary */}
                <div className="text-right">
                    {exercise.sets?.slice(0, 3).map((set, i) => (
                        <span key={i} className="text-sm text-gray-600 dark:text-gray-400">
                            {i > 0 && ' • '}{set.reps}×{set.weight}kg
                        </span>
                    ))}
                </div>
            </div>
        </div>
    );
};

/**
 * Superset Toggle Button (for use in workout logger)
 */
export const SupersetToggle = ({ selectedExercises, onCreateSuperset, disabled }) => {
    if (selectedExercises?.length < 2) return null;

    return (
        <motion.button
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            onClick={() => onCreateSuperset(selectedExercises)}
            disabled={disabled}
            className="fixed bottom-20 left-1/2 -translate-x-1/2 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-full shadow-lg flex items-center gap-2 z-40"
        >
            <Link2 className="h-4 w-4" />
            Create Superset ({selectedExercises.length} selected)
        </motion.button>
    );
};

export default SupersetBuilder;

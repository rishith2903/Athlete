import React from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { Activity, Clock, Flame, Award, TrendingUp } from 'lucide-react';

const ActivitySummary = ({ stats, loading = false }) => {
    const isEmpty = !stats || (
        stats.totalWorkouts === 0 &&
        stats.totalMinutes === 0 &&
        stats.totalCalories === 0
    );

    if (loading) {
        return (
            <div className="bg-white rounded-xl p-6 shadow-sm animate-pulse">
                <div className="h-6 bg-gray-200 rounded w-48 mb-4"></div>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    {[...Array(4)].map((_, i) => (
                        <div key={i} className="bg-gray-50 rounded-lg p-4">
                            <div className="h-4 bg-gray-200 rounded w-20 mb-2"></div>
                            <div className="h-8 bg-gray-200 rounded w-12"></div>
                        </div>
                    ))}
                </div>
            </div>
        );
    }

    if (isEmpty) {
        return (
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.15 }}
                className="bg-white rounded-xl p-6 shadow-sm"
            >
                <div className="flex items-center justify-between mb-4">
                    <h2 className="text-lg font-semibold text-gray-900">Recent Activity (Last 30 Days)</h2>
                </div>
                <div className="text-center py-8">
                    <div className="h-16 w-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                        <Activity className="h-8 w-8 text-gray-400" />
                    </div>
                    <h3 className="text-lg font-medium text-gray-900 mb-2">No activity yet</h3>
                    <p className="text-gray-500 mb-4">Start your first workout to see your progress here!</p>
                    <Link
                        to="/workouts/generate"
                        className="inline-flex items-center text-blue-600 hover:text-blue-700 font-medium"
                    >
                        <TrendingUp className="h-4 w-4 mr-2" />
                        Generate a Workout
                    </Link>
                </div>
            </motion.div>
        );
    }

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 }}
            className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm"
        >
            <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Recent Activity (Last 30 Days)</h2>
                <Link to="/workouts" className="text-sm text-blue-600 hover:text-blue-700 font-medium">
                    View Details
                </Link>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-gray-50 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-gray-600">Total Workouts</span>
                        <Activity className="h-4 w-4 text-gray-400" />
                    </div>
                    <p className="text-2xl font-bold text-gray-900">{stats.totalWorkouts}</p>
                    <p className="text-xs text-gray-500 mt-1">
                        {stats.averageWorkoutsPerWeek?.toFixed(1) || '0.0'} per week
                    </p>
                </div>
                <div className="bg-gray-50 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-gray-600">Total Minutes</span>
                        <Clock className="h-4 w-4 text-gray-400" />
                    </div>
                    <p className="text-2xl font-bold text-gray-900">{stats.totalMinutes}</p>
                    <p className="text-xs text-gray-500 mt-1">
                        {(stats.totalMinutes / 60).toFixed(1)} hours
                    </p>
                </div>
                <div className="bg-gray-50 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-gray-600">Calories Burned</span>
                        <Flame className="h-4 w-4 text-gray-400" />
                    </div>
                    <p className="text-2xl font-bold text-gray-900">{stats.totalCalories?.toLocaleString() || 0}</p>
                    <p className="text-xs text-gray-500 mt-1">
                        {(stats.totalCalories / 30).toFixed(0)} per day avg
                    </p>
                </div>
                <div className="bg-gray-50 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-gray-600">Consistency</span>
                        <Award className="h-4 w-4 text-gray-400" />
                    </div>
                    <p className="text-2xl font-bold text-gray-900">
                        {((stats.totalWorkouts / 30) * 100).toFixed(0)}%
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                        {stats.totalWorkouts} of 30 days
                    </p>
                </div>
            </div>
        </motion.div>
    );
};

export default ActivitySummary;

import React from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { Plus } from 'lucide-react';

const WelcomeBanner = ({ userName }) => {
    const firstName = userName?.split(' ')[0] || 'there';

    return (
        <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white dark:bg-gray-800 rounded-2xl p-8 relative overflow-hidden border border-gray-100 dark:border-gray-700 shadow-sm"
        >
            {/* Decorative circles */}
            <div className="absolute top-0 right-0 w-32 h-32 bg-blue-50 dark:bg-blue-900/20 rounded-full -translate-y-1/2 translate-x-1/2"></div>
            <div className="absolute bottom-0 left-20 w-24 h-24 bg-gray-50 dark:bg-gray-700/20 rounded-full translate-y-1/2"></div>

            <div className="relative z-10">
                <h1 className="text-3xl font-bold mb-2 text-gray-900 dark:text-white">
                    Welcome back, <span className="text-blue-600 dark:text-blue-400">{firstName}</span>! 👋
                </h1>
                <p className="text-gray-600 dark:text-gray-400 mb-6">
                    You're doing great! Keep up the momentum.
                </p>
                <div className="flex flex-wrap gap-4">
                    <Link
                        to="/workouts/generate"
                        className="inline-flex items-center bg-blue-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-blue-700 transition-colors shadow-sm"
                    >
                        <Plus className="h-5 w-5 mr-2" />
                        Start Workout
                    </Link>
                    <Link
                        to="/nutrition"
                        className="inline-flex items-center bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-white px-6 py-3 rounded-lg font-medium hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors border border-gray-200 dark:border-gray-600"
                    >
                        Track Meal
                    </Link>
                </div>
            </div>
        </motion.div>
    );
};

export default WelcomeBanner;


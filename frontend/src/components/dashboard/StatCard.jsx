import React from 'react';
import { motion } from 'framer-motion';
import { TrendingUp } from 'lucide-react';

const StatCard = ({ icon: Icon, title, value, change, color, loading = false }) => {
    if (loading) {
        return (
            <div className="bg-white rounded-xl p-6 shadow-sm animate-pulse">
                <div className="flex items-center justify-between">
                    <div className="flex-1">
                        <div className="h-4 bg-gray-200 rounded w-24 mb-2"></div>
                        <div className="h-8 bg-gray-200 rounded w-16"></div>
                    </div>
                    <div className="h-12 w-12 bg-gray-200 rounded-lg"></div>
                </div>
            </div>
        );
    }

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow"
        >
            <div className="flex items-center justify-between">
                <div>
                    <p className="text-sm text-gray-600 mb-1">{title}</p>
                    <p className="text-2xl font-bold text-gray-900">{value}</p>
                    {change !== undefined && change !== null && (
                        <p className={`text-sm mt-2 flex items-center ${change > 0 ? 'text-green-600' : 'text-red-600'}`}>
                            <TrendingUp className="h-4 w-4 mr-1" />
                            {change > 0 ? '+' : ''}{change}%
                        </p>
                    )}
                </div>
                <div className={`h-12 w-12 rounded-lg ${color} flex items-center justify-center`}>
                    <Icon className="h-6 w-6 text-white" />
                </div>
            </div>
        </motion.div>
    );
};

export default StatCard;

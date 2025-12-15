import React from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { Camera, Activity, TrendingUp, Calendar } from 'lucide-react';

const quickActionItems = [
    {
        to: '/pose-analysis',
        icon: Camera,
        label: 'Check Form',
        bgColor: 'bg-blue-100',
        iconColor: 'text-blue-600'
    },
    {
        to: '/chatbot',
        icon: Activity,
        label: 'AI Coach',
        bgColor: 'bg-purple-100',
        iconColor: 'text-purple-600'
    },
    {
        to: '/progress',
        icon: TrendingUp,
        label: 'View Progress',
        bgColor: 'bg-green-100',
        iconColor: 'text-green-600'
    },
    {
        to: '/schedule',
        icon: Calendar,
        label: 'Schedule',
        bgColor: 'bg-orange-100',
        iconColor: 'text-orange-600'
    }
];

const QuickActions = () => {
    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="bg-gradient-to-r from-gray-50 to-gray-100 rounded-xl p-6"
        >
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {quickActionItems.map((item) => (
                    <Link
                        key={item.to}
                        to={item.to}
                        className="flex flex-col items-center p-4 bg-white rounded-lg hover:shadow-md transition-all hover:-translate-y-0.5"
                    >
                        <div className={`h-12 w-12 ${item.bgColor} rounded-lg flex items-center justify-center mb-2`}>
                            <item.icon className={`h-6 w-6 ${item.iconColor}`} />
                        </div>
                        <span className="text-sm font-medium text-gray-900">{item.label}</span>
                    </Link>
                ))}
            </div>
        </motion.div>
    );
};

export default QuickActions;

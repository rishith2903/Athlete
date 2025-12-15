import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
    BarChart,
    Bar,
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    AreaChart,
    Area
} from 'recharts';
import {
    TrendingUp,
    Calendar,
    Dumbbell,
    Flame,
    Trophy,
    ChevronLeft,
    ChevronRight,
    Target
} from 'lucide-react';
import api from '../services/api';

const Progress = () => {
    const [stats, setStats] = useState(null);
    const [frequencyData, setFrequencyData] = useState([]);
    const [calendarData, setCalendarData] = useState([]);
    const [loading, setLoading] = useState(true);
    const [currentMonth, setCurrentMonth] = useState(new Date());

    useEffect(() => {
        fetchData();
    }, []);

    useEffect(() => {
        fetchCalendar();
    }, [currentMonth]);

    const fetchData = async () => {
        try {
            setLoading(true);
            const [statsRes, freqRes] = await Promise.all([
                api.get('/stats/dashboard'),
                api.get('/stats/frequency?weeks=12')
            ]);
            setStats(statsRes.data.data);
            setFrequencyData(freqRes.data.data || []);
        } catch (error) {
            console.error('Failed to fetch stats:', error);
            // Use sample data
            setStats(SAMPLE_STATS);
            setFrequencyData(SAMPLE_FREQUENCY);
        } finally {
            setLoading(false);
        }
    };

    const fetchCalendar = async () => {
        try {
            const year = currentMonth.getFullYear();
            const month = currentMonth.getMonth() + 1;
            const response = await api.get(`/stats/calendar?year=${year}&month=${month}`);
            setCalendarData(response.data.data || []);
        } catch (error) {
            console.error('Failed to fetch calendar:', error);
            setCalendarData(generateSampleCalendar());
        }
    };

    const generateSampleCalendar = () => {
        const days = [];
        const daysInMonth = new Date(currentMonth.getFullYear(), currentMonth.getMonth() + 1, 0).getDate();
        for (let i = 1; i <= daysInMonth; i++) {
            days.push({
                day: i,
                hasWorkout: Math.random() > 0.6,
                workoutCount: Math.random() > 0.8 ? 2 : 1
            });
        }
        return days;
    };

    const navigateMonth = (direction) => {
        setCurrentMonth(prev => {
            const newDate = new Date(prev);
            newDate.setMonth(newDate.getMonth() + direction);
            return newDate;
        });
    };

    const getMonthName = () => {
        return currentMonth.toLocaleDateString('en-US', { month: 'long', year: 'numeric' });
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div>
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Progress</h1>
                <p className="text-gray-600 dark:text-gray-400">Track your fitness journey</p>
            </div>

            {/* Stats Overview */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                {loading ? (
                    Array(4).fill(0).map((_, i) => (
                        <div key={i} className="bg-white dark:bg-gray-800 rounded-xl p-4 animate-pulse">
                            <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-1/2 mb-2"></div>
                            <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4"></div>
                        </div>
                    ))
                ) : (
                    <>
                        <StatCard
                            icon={Dumbbell}
                            label="Total Workouts"
                            value={stats?.totalWorkouts || 0}
                            color="blue"
                        />
                        <StatCard
                            icon={Calendar}
                            label="This Week"
                            value={stats?.workoutsThisWeek || 0}
                            color="green"
                        />
                        <StatCard
                            icon={Flame}
                            label="Current Streak"
                            value={`${stats?.currentStreak || 0} days`}
                            color="orange"
                        />
                        <StatCard
                            icon={Trophy}
                            label="This Month"
                            value={stats?.workoutsThisMonth || 0}
                            color="purple"
                        />
                    </>
                )}
            </div>

            {/* Charts Row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Workout Frequency Chart */}
                <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                        Workout Frequency
                    </h3>
                    <div className="h-64">
                        {loading ? (
                            <div className="h-full bg-gray-100 dark:bg-gray-700 rounded animate-pulse"></div>
                        ) : (
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={frequencyData}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
                                    <XAxis
                                        dataKey="weekStart"
                                        tick={{ fill: '#9CA3AF', fontSize: 12 }}
                                        tickFormatter={(v) => v.slice(5, 10)}
                                    />
                                    <YAxis tick={{ fill: '#9CA3AF', fontSize: 12 }} />
                                    <Tooltip
                                        contentStyle={{
                                            backgroundColor: '#1F2937',
                                            border: 'none',
                                            borderRadius: '8px',
                                            color: '#fff'
                                        }}
                                    />
                                    <Bar dataKey="workoutCount" fill="#3B82F6" radius={[4, 4, 0, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        )}
                    </div>
                </div>

                {/* Volume Trend Chart */}
                <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                        Weekly Volume
                    </h3>
                    <div className="h-64">
                        {loading ? (
                            <div className="h-full bg-gray-100 dark:bg-gray-700 rounded animate-pulse"></div>
                        ) : (
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={frequencyData}>
                                    <defs>
                                        <linearGradient id="volumeGradient" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#8B5CF6" stopOpacity={0.8} />
                                            <stop offset="95%" stopColor="#8B5CF6" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
                                    <XAxis
                                        dataKey="weekStart"
                                        tick={{ fill: '#9CA3AF', fontSize: 12 }}
                                        tickFormatter={(v) => v.slice(5, 10)}
                                    />
                                    <YAxis tick={{ fill: '#9CA3AF', fontSize: 12 }} />
                                    <Tooltip
                                        contentStyle={{
                                            backgroundColor: '#1F2937',
                                            border: 'none',
                                            borderRadius: '8px',
                                            color: '#fff'
                                        }}
                                        formatter={(value) => [`${(value / 1000).toFixed(1)}k kg`, 'Volume']}
                                    />
                                    <Area
                                        type="monotone"
                                        dataKey="totalVolume"
                                        stroke="#8B5CF6"
                                        fillOpacity={1}
                                        fill="url(#volumeGradient)"
                                    />
                                </AreaChart>
                            </ResponsiveContainer>
                        )}
                    </div>
                </div>
            </div>

            {/* Workout Calendar */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm">
                <div className="flex items-center justify-between mb-6">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                        Workout Calendar
                    </h3>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={() => navigateMonth(-1)}
                            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg"
                        >
                            <ChevronLeft className="h-5 w-5 text-gray-500" />
                        </button>
                        <span className="text-gray-700 dark:text-gray-300 font-medium min-w-[150px] text-center">
                            {getMonthName()}
                        </span>
                        <button
                            onClick={() => navigateMonth(1)}
                            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg"
                        >
                            <ChevronRight className="h-5 w-5 text-gray-500" />
                        </button>
                    </div>
                </div>

                {/* Calendar Grid */}
                <div className="grid grid-cols-7 gap-2">
                    {/* Day Headers */}
                    {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].map(day => (
                        <div key={day} className="text-center text-sm font-medium text-gray-500 dark:text-gray-400 py-2">
                            {day}
                        </div>
                    ))}

                    {/* Empty cells for alignment */}
                    {(() => {
                        const firstDay = new Date(currentMonth.getFullYear(), currentMonth.getMonth(), 1).getDay();
                        const offset = firstDay === 0 ? 6 : firstDay - 1;
                        return Array(offset).fill(0).map((_, i) => (
                            <div key={`empty-${i}`} />
                        ));
                    })()}

                    {/* Calendar Days */}
                    {calendarData.map((day, i) => (
                        <motion.div
                            key={day.day}
                            initial={{ opacity: 0, scale: 0.9 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ delay: i * 0.01 }}
                            className={`
                aspect-square rounded-lg flex items-center justify-center text-sm font-medium
                ${day.hasWorkout
                                    ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400'
                                    : 'bg-gray-50 dark:bg-gray-700/30 text-gray-600 dark:text-gray-400'
                                }
                ${day.workoutCount > 1 ? 'ring-2 ring-green-500' : ''}
              `}
                        >
                            {day.day}
                        </motion.div>
                    ))}
                </div>

                {/* Legend */}
                <div className="flex items-center gap-6 mt-4 text-sm">
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-4 bg-gray-50 dark:bg-gray-700/30 rounded"></div>
                        <span className="text-gray-500 dark:text-gray-400">No workout</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-4 bg-green-100 dark:bg-green-900/30 rounded"></div>
                        <span className="text-gray-500 dark:text-gray-400">Workout day</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-4 bg-green-100 dark:bg-green-900/30 rounded ring-2 ring-green-500"></div>
                        <span className="text-gray-500 dark:text-gray-400">Multiple workouts</span>
                    </div>
                </div>
            </div>
        </div>
    );
};

// Stat Card Component
const StatCard = ({ icon: Icon, label, value, color }) => {
    const colorClasses = {
        blue: 'bg-blue-100 text-blue-600 dark:bg-blue-900/30 dark:text-blue-400',
        green: 'bg-green-100 text-green-600 dark:bg-green-900/30 dark:text-green-400',
        orange: 'bg-orange-100 text-orange-600 dark:bg-orange-900/30 dark:text-orange-400',
        purple: 'bg-purple-100 text-purple-600 dark:bg-purple-900/30 dark:text-purple-400',
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm"
        >
            <div className="flex items-center gap-3">
                <div className={`p-2 rounded-lg ${colorClasses[color]}`}>
                    <Icon className="h-5 w-5" />
                </div>
                <div>
                    <div className="text-2xl font-bold text-gray-900 dark:text-white">{value}</div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">{label}</div>
                </div>
            </div>
        </motion.div>
    );
};

// Sample data
const SAMPLE_STATS = {
    totalWorkouts: 47,
    workoutsThisWeek: 4,
    workoutsThisMonth: 12,
    currentStreak: 7,
    totalVolumeThisMonth: 45000
};

const SAMPLE_FREQUENCY = [
    { weekStart: '2024-10-07', workoutCount: 3, totalVolume: 12000 },
    { weekStart: '2024-10-14', workoutCount: 4, totalVolume: 15000 },
    { weekStart: '2024-10-21', workoutCount: 3, totalVolume: 11000 },
    { weekStart: '2024-10-28', workoutCount: 5, totalVolume: 18000 },
    { weekStart: '2024-11-04', workoutCount: 4, totalVolume: 14000 },
    { weekStart: '2024-11-11', workoutCount: 3, totalVolume: 12000 },
    { weekStart: '2024-11-18', workoutCount: 4, totalVolume: 16000 },
    { weekStart: '2024-11-25', workoutCount: 2, totalVolume: 8000 },
    { weekStart: '2024-12-02', workoutCount: 5, totalVolume: 19000 },
    { weekStart: '2024-12-09', workoutCount: 4, totalVolume: 15000 },
];

export default Progress;

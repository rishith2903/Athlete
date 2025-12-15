import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Trophy,
    Flame,
    Star,
    Zap,
    Crown,
    Medal,
    Target,
    TrendingUp,
    Users,
    ChevronRight
} from 'lucide-react';
import api from '../services/api';

const Achievements = () => {
    const [stats, setStats] = useState(null);
    const [achievements, setAchievements] = useState([]);
    const [loading, setLoading] = useState(true);
    const [activeTab, setActiveTab] = useState('achievements');

    useEffect(() => {
        fetchData();
    }, []);

    const fetchData = async () => {
        try {
            setLoading(true);
            const [statsRes, achievementsRes] = await Promise.all([
                api.get('/gamification/stats'),
                api.get('/gamification/achievements')
            ]);
            setStats(statsRes.data.data);
            setAchievements(achievementsRes.data.data || []);
        } catch (error) {
            console.error('Failed to fetch data:', error);
            setStats(SAMPLE_STATS);
            setAchievements(SAMPLE_ACHIEVEMENTS);
        } finally {
            setLoading(false);
        }
    };

    const getTierColor = (tier) => {
        const colors = {
            1: 'from-amber-600 to-amber-800', // Bronze
            2: 'from-gray-300 to-gray-500', // Silver
            3: 'from-yellow-400 to-yellow-600', // Gold
            4: 'from-cyan-400 to-purple-600', // Platinum
        };
        return colors[tier] || colors[1];
    };

    const getTierName = (tier) => {
        const names = { 1: 'Bronze', 2: 'Silver', 3: 'Gold', 4: 'Platinum' };
        return names[tier] || 'Bronze';
    };

    const getCategoryIcon = (category) => {
        const icons = {
            MILESTONE: Trophy,
            STREAK: Flame,
            STRENGTH: Zap,
            VOLUME: TrendingUp,
            CONSISTENCY: Star,
        };
        return icons[category] || Trophy;
    };

    return (
        <div className="space-y-6">
            {/* Header with Level & XP */}
            <div className="bg-gradient-to-r from-purple-600 to-indigo-600 rounded-2xl p-6 text-white">
                <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                    <div className="flex items-center gap-4">
                        <div className="h-16 w-16 bg-white/20 rounded-full flex items-center justify-center">
                            <Crown className="h-8 w-8" />
                        </div>
                        <div>
                            <div className="text-sm opacity-80">Level</div>
                            <div className="text-4xl font-bold">{stats?.level || 1}</div>
                        </div>
                    </div>

                    <div className="flex-1 max-w-md">
                        <div className="flex justify-between text-sm mb-1">
                            <span>XP Progress</span>
                            <span>{stats?.totalXp || 0} XP</span>
                        </div>
                        <div className="h-3 bg-white/20 rounded-full overflow-hidden">
                            <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${(stats?.totalXp % 100) || 0}%` }}
                                className="h-full bg-white rounded-full"
                            />
                        </div>
                    </div>

                    <div className="flex gap-6">
                        <div className="text-center">
                            <div className="text-2xl font-bold">{stats?.currentStreak || 0}</div>
                            <div className="text-sm opacity-80 flex items-center gap-1">
                                <Flame className="h-4 w-4" />
                                Day Streak
                            </div>
                        </div>
                        <div className="text-center">
                            <div className="text-2xl font-bold">{achievements.length}</div>
                            <div className="text-sm opacity-80 flex items-center gap-1">
                                <Medal className="h-4 w-4" />
                                Badges
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatCard icon={Trophy} label="Total Workouts" value={stats?.totalWorkouts || 0} color="blue" />
                <StatCard icon={Flame} label="Longest Streak" value={`${stats?.longestStreak || 0} days`} color="orange" />
                <StatCard icon={TrendingUp} label="Total Volume" value={`${((stats?.totalVolume || 0) / 1000).toFixed(1)}k kg`} color="green" />
                <StatCard icon={Target} label="PRs Broken" value={stats?.totalPRs || 0} color="purple" />
            </div>

            {/* Tabs */}
            <div className="flex gap-2 bg-gray-100 dark:bg-gray-800 p-1 rounded-xl">
                {[
                    { id: 'achievements', label: 'Achievements', icon: Trophy },
                    { id: 'leaderboard', label: 'Leaderboard', icon: Users },
                ].map(tab => (
                    <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={`flex-1 flex items-center justify-center gap-2 py-3 rounded-lg font-medium transition ${activeTab === tab.id
                                ? 'bg-white dark:bg-gray-700 text-blue-600 dark:text-blue-400 shadow-sm'
                                : 'text-gray-600 dark:text-gray-400'
                            }`}
                    >
                        <tab.icon className="h-5 w-5" />
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* Content */}
            <AnimatePresence mode="wait">
                {activeTab === 'achievements' && (
                    <motion.div
                        key="achievements"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        className="space-y-6"
                    >
                        {/* Achievement Categories */}
                        {['MILESTONE', 'STREAK', 'STRENGTH', 'VOLUME'].map(category => {
                            const categoryAchievements = achievements.filter(a => a.category === category);
                            if (categoryAchievements.length === 0) return null;

                            const CategoryIcon = getCategoryIcon(category);

                            return (
                                <div key={category}>
                                    <div className="flex items-center gap-2 mb-3">
                                        <CategoryIcon className="h-5 w-5 text-gray-500 dark:text-gray-400" />
                                        <h3 className="font-semibold text-gray-900 dark:text-white">{category}</h3>
                                        <span className="text-sm text-gray-500 dark:text-gray-400">
                                            ({categoryAchievements.length})
                                        </span>
                                    </div>
                                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                                        {categoryAchievements.map((achievement, i) => (
                                            <AchievementCard key={achievement.id || i} achievement={achievement} />
                                        ))}
                                    </div>
                                </div>
                            );
                        })}

                        {achievements.length === 0 && !loading && (
                            <div className="text-center py-12 bg-white dark:bg-gray-800 rounded-xl">
                                <Trophy className="h-16 w-16 text-gray-300 dark:text-gray-600 mx-auto mb-4" />
                                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">No Achievements Yet</h3>
                                <p className="text-gray-600 dark:text-gray-400">Complete workouts to earn badges!</p>
                            </div>
                        )}
                    </motion.div>
                )}

                {activeTab === 'leaderboard' && (
                    <motion.div
                        key="leaderboard"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                    >
                        <LeaderboardView />
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

// Achievement Card Component
const AchievementCard = ({ achievement }) => {
    const getTierGradient = (tier) => {
        const gradients = {
            1: 'from-amber-500 to-amber-700',
            2: 'from-gray-400 to-gray-600',
            3: 'from-yellow-400 to-yellow-600',
            4: 'from-cyan-400 to-purple-500',
        };
        return gradients[tier] || gradients[1];
    };

    return (
        <motion.div
            whileHover={{ scale: 1.02 }}
            className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-100 dark:border-gray-700"
        >
            <div className={`w-12 h-12 rounded-full bg-gradient-to-br ${getTierGradient(achievement.tier)} flex items-center justify-center text-2xl mb-3`}>
                {achievement.icon}
            </div>
            <h4 className="font-semibold text-gray-900 dark:text-white">{achievement.name}</h4>
            <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">{achievement.description}</p>
            <div className="flex items-center justify-between">
                <span className="text-xs text-purple-600 dark:text-purple-400 font-medium">
                    +{achievement.xpReward} XP
                </span>
                <span className="text-xs text-gray-400">
                    {new Date(achievement.earnedAt).toLocaleDateString()}
                </span>
            </div>
        </motion.div>
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
        <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm">
            <div className="flex items-center gap-3">
                <div className={`p-2 rounded-lg ${colorClasses[color]}`}>
                    <Icon className="h-5 w-5" />
                </div>
                <div>
                    <div className="text-xl font-bold text-gray-900 dark:text-white">{value}</div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">{label}</div>
                </div>
            </div>
        </div>
    );
};

// Leaderboard View Component
const LeaderboardView = () => {
    const [leaderboard, setLeaderboard] = useState([]);
    const [leaderboardType, setLeaderboardType] = useState('xp');
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchLeaderboard();
    }, [leaderboardType]);

    const fetchLeaderboard = async () => {
        try {
            setLoading(true);
            const response = await api.get(`/gamification/leaderboard?type=${leaderboardType}`);
            setLeaderboard(response.data.data || []);
        } catch (error) {
            console.error('Failed to fetch leaderboard:', error);
            setLeaderboard(SAMPLE_LEADERBOARD);
        } finally {
            setLoading(false);
        }
    };

    const getRankDisplay = (rank) => {
        if (rank === 1) return <span className="text-2xl">🥇</span>;
        if (rank === 2) return <span className="text-2xl">🥈</span>;
        if (rank === 3) return <span className="text-2xl">🥉</span>;
        return <span className="text-lg font-bold text-gray-500">#{rank}</span>;
    };

    const getValue = (user) => {
        switch (leaderboardType) {
            case 'xp': return `${user.totalXp?.toLocaleString() || 0} XP`;
            case 'streak': return `${user.currentStreak || 0} days`;
            case 'volume': return `${((user.totalVolume || 0) / 1000).toFixed(1)}k kg`;
            case 'workouts': return `${user.totalWorkouts || 0} workouts`;
            default: return `${user.totalXp || 0} XP`;
        }
    };

    return (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm overflow-hidden">
            {/* Leaderboard Type Selector */}
            <div className="p-4 border-b border-gray-200 dark:border-gray-700">
                <div className="flex gap-2 overflow-x-auto">
                    {[
                        { id: 'xp', label: 'XP', icon: Star },
                        { id: 'streak', label: 'Streak', icon: Flame },
                        { id: 'volume', label: 'Volume', icon: TrendingUp },
                        { id: 'workouts', label: 'Workouts', icon: Trophy },
                    ].map(type => (
                        <button
                            key={type.id}
                            onClick={() => setLeaderboardType(type.id)}
                            className={`px-4 py-2 rounded-lg flex items-center gap-2 text-sm font-medium transition whitespace-nowrap ${leaderboardType === type.id
                                    ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400'
                                    : 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-400'
                                }`}
                        >
                            <type.icon className="h-4 w-4" />
                            {type.label}
                        </button>
                    ))}
                </div>
            </div>

            {/* Leaderboard List */}
            <div className="divide-y divide-gray-100 dark:divide-gray-700">
                {loading ? (
                    Array(5).fill(0).map((_, i) => (
                        <div key={i} className="p-4 animate-pulse flex items-center gap-4">
                            <div className="w-8 h-8 bg-gray-200 dark:bg-gray-700 rounded-full"></div>
                            <div className="flex-1">
                                <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/3 mb-2"></div>
                                <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-1/4"></div>
                            </div>
                        </div>
                    ))
                ) : (
                    leaderboard.map((user, i) => (
                        <div key={user.userId || i} className="p-4 flex items-center gap-4 hover:bg-gray-50 dark:hover:bg-gray-700/50">
                            <div className="w-10 flex justify-center">
                                {getRankDisplay(i + 1)}
                            </div>
                            <div className="h-10 w-10 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-full flex items-center justify-center text-white font-bold">
                                {user.userId?.charAt(0)?.toUpperCase() || 'U'}
                            </div>
                            <div className="flex-1">
                                <div className="font-medium text-gray-900 dark:text-white">
                                    User {user.userId?.slice(-4) || 'Anonymous'}
                                </div>
                                <div className="text-sm text-gray-500 dark:text-gray-400">
                                    Level {user.level || 1}
                                </div>
                            </div>
                            <div className="text-right">
                                <div className="font-bold text-gray-900 dark:text-white">
                                    {getValue(user)}
                                </div>
                            </div>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
};

// Sample data
const SAMPLE_STATS = {
    level: 5,
    totalXp: 1250,
    currentStreak: 7,
    longestStreak: 14,
    totalWorkouts: 47,
    totalVolume: 45000,
    totalPRs: 12
};

const SAMPLE_ACHIEVEMENTS = [
    { id: '1', name: 'First Workout', description: 'Complete your first workout', icon: '💪', category: 'MILESTONE', tier: 1, xpReward: 50, earnedAt: '2024-11-01' },
    { id: '2', name: 'Getting Started', description: 'Complete 10 workouts', icon: '🏋️', category: 'MILESTONE', tier: 1, xpReward: 100, earnedAt: '2024-11-15' },
    { id: '3', name: 'Week Warrior', description: '7 day workout streak', icon: '🔥', category: 'STREAK', tier: 1, xpReward: 100, earnedAt: '2024-12-01' },
    { id: '4', name: 'Personal Best', description: 'Break your first PR', icon: '⭐', category: 'STRENGTH', tier: 1, xpReward: 75, earnedAt: '2024-11-20' },
    { id: '5', name: '10K Club', description: 'Lift 10,000 kg total', icon: '🏗️', category: 'VOLUME', tier: 1, xpReward: 100, earnedAt: '2024-11-25' },
];

const SAMPLE_LEADERBOARD = [
    { userId: 'user123abc', level: 12, totalXp: 5600, currentStreak: 21, totalVolume: 125000, totalWorkouts: 89 },
    { userId: 'user456def', level: 10, totalXp: 4200, currentStreak: 14, totalVolume: 98000, totalWorkouts: 72 },
    { userId: 'user789ghi', level: 8, totalXp: 3100, currentStreak: 7, totalVolume: 75000, totalWorkouts: 56 },
    { userId: 'user012jkl', level: 6, totalXp: 1800, currentStreak: 5, totalVolume: 52000, totalWorkouts: 41 },
    { userId: 'user345mno', level: 5, totalXp: 1250, currentStreak: 3, totalVolume: 45000, totalWorkouts: 35 },
];

export default Achievements;

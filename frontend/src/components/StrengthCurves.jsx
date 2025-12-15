import React, { useState, useEffect } from 'react';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    ReferenceLine
} from 'recharts';
import { TrendingUp, Calendar } from 'lucide-react';
import api from '../services/api';

/**
 * Strength Curves Component
 * Shows progression of a specific exercise over time
 */
const StrengthCurves = ({ exerciseId, exerciseName }) => {
    const [data, setData] = useState([]);
    const [loading, setLoading] = useState(true);
    const [timeRange, setTimeRange] = useState('3m'); // 1m, 3m, 6m, 1y, all

    useEffect(() => {
        fetchStrengthData();
    }, [exerciseId, timeRange]);

    const fetchStrengthData = async () => {
        try {
            setLoading(true);
            const response = await api.get(`/stats/strength/${exerciseId}?range=${timeRange}`);
            setData(response.data.data || []);
        } catch (error) {
            console.error('Failed to fetch strength data:', error);
            // Sample data
            setData(SAMPLE_STRENGTH_DATA);
        } finally {
            setLoading(false);
        }
    };

    const maxWeight = Math.max(...data.map(d => d.estimated1RM || 0), 0);
    const minWeight = Math.min(...data.filter(d => d.estimated1RM > 0).map(d => d.estimated1RM), maxWeight);
    const improvement = data.length > 1 ?
        ((data[data.length - 1]?.estimated1RM - data[0]?.estimated1RM) / data[0]?.estimated1RM * 100).toFixed(1) : 0;

    return (
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm">
            <div className="flex items-center justify-between mb-4">
                <div>
                    <h3 className="font-semibold text-gray-900 dark:text-white flex items-center gap-2">
                        <TrendingUp className="h-5 w-5 text-blue-500" />
                        {exerciseName || 'Exercise'} Progression
                    </h3>
                    <p className="text-sm text-gray-500 dark:text-gray-400">Estimated 1RM over time</p>
                </div>
                <div className="flex gap-1">
                    {['1m', '3m', '6m', '1y', 'all'].map(range => (
                        <button
                            key={range}
                            onClick={() => setTimeRange(range)}
                            className={`px-2 py-1 text-xs rounded transition ${timeRange === range
                                    ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400'
                                    : 'text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700'
                                }`}
                        >
                            {range.toUpperCase()}
                        </button>
                    ))}
                </div>
            </div>

            {/* Stats Row */}
            <div className="grid grid-cols-3 gap-4 mb-4">
                <div className="text-center p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                    <div className="text-lg font-bold text-gray-900 dark:text-white">{Math.round(maxWeight)} kg</div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">Best 1RM</div>
                </div>
                <div className="text-center p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                    <div className="text-lg font-bold text-gray-900 dark:text-white">{data.length}</div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">Sessions</div>
                </div>
                <div className="text-center p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                    <div className={`text-lg font-bold ${improvement > 0 ? 'text-green-500' : 'text-red-500'}`}>
                        {improvement > 0 ? '+' : ''}{improvement}%
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">Progress</div>
                </div>
            </div>

            {/* Chart */}
            <div className="h-64">
                {loading ? (
                    <div className="h-full bg-gray-100 dark:bg-gray-700 rounded animate-pulse"></div>
                ) : data.length === 0 ? (
                    <div className="h-full flex items-center justify-center text-gray-500 dark:text-gray-400">
                        No data available
                    </div>
                ) : (
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={data}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
                            <XAxis
                                dataKey="date"
                                tick={{ fill: '#9CA3AF', fontSize: 11 }}
                                tickFormatter={(v) => new Date(v).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                            />
                            <YAxis
                                domain={[Math.floor(minWeight * 0.9), Math.ceil(maxWeight * 1.1)]}
                                tick={{ fill: '#9CA3AF', fontSize: 11 }}
                            />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: '#1F2937',
                                    border: 'none',
                                    borderRadius: '8px',
                                    color: '#fff'
                                }}
                                formatter={(value) => [`${Math.round(value)} kg`, 'Est. 1RM']}
                                labelFormatter={(label) => new Date(label).toLocaleDateString()}
                            />
                            <ReferenceLine y={maxWeight} stroke="#22C55E" strokeDasharray="5 5" />
                            <Line
                                type="monotone"
                                dataKey="estimated1RM"
                                stroke="#3B82F6"
                                strokeWidth={2}
                                dot={{ fill: '#3B82F6', strokeWidth: 2, r: 4 }}
                                activeDot={{ r: 6 }}
                            />
                        </LineChart>
                    </ResponsiveContainer>
                )}
            </div>
        </div>
    );
};

// Sample data
const SAMPLE_STRENGTH_DATA = [
    { date: '2024-10-01', estimated1RM: 100, weight: 80, reps: 8 },
    { date: '2024-10-08', estimated1RM: 102, weight: 82.5, reps: 7 },
    { date: '2024-10-15', estimated1RM: 105, weight: 85, reps: 7 },
    { date: '2024-10-22', estimated1RM: 107, weight: 87.5, reps: 6 },
    { date: '2024-10-29', estimated1RM: 110, weight: 90, reps: 6 },
    { date: '2024-11-05', estimated1RM: 112, weight: 90, reps: 7 },
    { date: '2024-11-12', estimated1RM: 115, weight: 95, reps: 5 },
    { date: '2024-11-19', estimated1RM: 117, weight: 95, reps: 6 },
    { date: '2024-11-26', estimated1RM: 120, weight: 100, reps: 5 },
    { date: '2024-12-03', estimated1RM: 122, weight: 100, reps: 6 },
];

export default StrengthCurves;

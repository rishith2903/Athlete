import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer
} from 'recharts';
import {
    Plus,
    TrendingUp,
    TrendingDown,
    Scale,
    Ruler,
    Calendar,
    Trash2,
    X,
    Save
} from 'lucide-react';
import api from '../services/api';

const BodyMeasurements = () => {
    const [measurements, setMeasurements] = useState([]);
    const [loading, setLoading] = useState(true);
    const [showAddModal, setShowAddModal] = useState(false);
    const [activeChart, setActiveChart] = useState('weight');

    useEffect(() => {
        fetchMeasurements();
    }, []);

    const fetchMeasurements = async () => {
        try {
            setLoading(true);
            const response = await api.get('/measurements');
            setMeasurements(response.data.data || []);
        } catch (error) {
            console.error('Failed to fetch measurements:', error);
            setMeasurements(SAMPLE_MEASUREMENTS);
        } finally {
            setLoading(false);
        }
    };

    const deleteMeasurement = async (id) => {
        if (!confirm('Delete this measurement?')) return;
        try {
            await api.delete(`/measurements/${id}`);
            setMeasurements(prev => prev.filter(m => m.id !== id));
        } catch (error) {
            console.error('Failed to delete:', error);
        }
    };

    const getLatest = () => measurements[0] || {};
    const getPrevious = () => measurements[1] || {};

    const getChange = (field) => {
        const latest = getLatest()[field];
        const previous = getPrevious()[field];
        if (latest && previous) {
            return (latest - previous).toFixed(1);
        }
        return null;
    };

    const chartData = measurements.slice().reverse().map(m => ({
        date: new Date(m.measurementDate).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        weight: m.weight,
        bodyFat: m.bodyFatPercentage,
        chest: m.chest,
        waist: m.waist,
        leftBicep: m.leftBicep,
        rightBicep: m.rightBicep
    }));

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Body Measurements</h1>
                    <p className="text-gray-600 dark:text-gray-400">Track your progress over time</p>
                </div>
                <button
                    onClick={() => setShowAddModal(true)}
                    className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-semibold flex items-center gap-2 transition"
                >
                    <Plus className="h-5 w-5" />
                    Add Measurement
                </button>
            </div>

            {/* Current Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatCard
                    icon={Scale}
                    label="Weight"
                    value={`${getLatest().weight || '-'} kg`}
                    change={getChange('weight')}
                    unit="kg"
                />
                <StatCard
                    icon={TrendingUp}
                    label="Body Fat"
                    value={`${getLatest().bodyFatPercentage || '-'}%`}
                    change={getChange('bodyFatPercentage')}
                    unit="%"
                    inverse
                />
                <StatCard
                    icon={Ruler}
                    label="Chest"
                    value={`${getLatest().chest || '-'} cm`}
                    change={getChange('chest')}
                    unit="cm"
                />
                <StatCard
                    icon={Ruler}
                    label="Waist"
                    value={`${getLatest().waist || '-'} cm`}
                    change={getChange('waist')}
                    unit="cm"
                    inverse
                />
            </div>

            {/* Chart */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm">
                <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Progress Chart</h3>
                    <div className="flex gap-2">
                        {[
                            { id: 'weight', label: 'Weight' },
                            { id: 'bodyFat', label: 'Body Fat' },
                            { id: 'chest', label: 'Chest' },
                            { id: 'waist', label: 'Waist' },
                        ].map(opt => (
                            <button
                                key={opt.id}
                                onClick={() => setActiveChart(opt.id)}
                                className={`px-3 py-1 rounded-lg text-sm font-medium transition ${activeChart === opt.id
                                        ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400'
                                        : 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-400'
                                    }`}
                            >
                                {opt.label}
                            </button>
                        ))}
                    </div>
                </div>
                <div className="h-64">
                    {loading ? (
                        <div className="h-full bg-gray-100 dark:bg-gray-700 rounded animate-pulse"></div>
                    ) : measurements.length === 0 ? (
                        <div className="h-full flex items-center justify-center text-gray-500 dark:text-gray-400">
                            No measurements yet. Add your first one!
                        </div>
                    ) : (
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={chartData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
                                <XAxis dataKey="date" tick={{ fill: '#9CA3AF', fontSize: 12 }} />
                                <YAxis tick={{ fill: '#9CA3AF', fontSize: 12 }} />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: '#1F2937',
                                        border: 'none',
                                        borderRadius: '8px',
                                        color: '#fff'
                                    }}
                                />
                                <Line
                                    type="monotone"
                                    dataKey={activeChart}
                                    stroke="#3B82F6"
                                    strokeWidth={2}
                                    dot={{ fill: '#3B82F6', strokeWidth: 2 }}
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    )}
                </div>
            </div>

            {/* Measurement History */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm overflow-hidden">
                <div className="p-4 border-b border-gray-200 dark:border-gray-700">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white">History</h3>
                </div>
                <div className="divide-y divide-gray-100 dark:divide-gray-700">
                    {loading ? (
                        Array(3).fill(0).map((_, i) => (
                            <div key={i} className="p-4 animate-pulse">
                                <div className="h-5 bg-gray-200 dark:bg-gray-700 rounded w-1/4 mb-2"></div>
                                <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4"></div>
                            </div>
                        ))
                    ) : measurements.length === 0 ? (
                        <div className="p-8 text-center text-gray-500 dark:text-gray-400">
                            No measurements recorded yet
                        </div>
                    ) : (
                        measurements.slice(0, 10).map((m, i) => (
                            <div key={m.id || i} className="p-4 flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700/50">
                                <div className="flex items-center gap-4">
                                    <div className="p-2 bg-gray-100 dark:bg-gray-700 rounded-lg">
                                        <Calendar className="h-5 w-5 text-gray-500 dark:text-gray-400" />
                                    </div>
                                    <div>
                                        <div className="font-medium text-gray-900 dark:text-white">
                                            {new Date(m.measurementDate).toLocaleDateString('en-US', {
                                                month: 'short', day: 'numeric', year: 'numeric'
                                            })}
                                        </div>
                                        <div className="text-sm text-gray-500 dark:text-gray-400">
                                            {m.weight && `${m.weight} kg`}
                                            {m.bodyFatPercentage && ` • ${m.bodyFatPercentage}% BF`}
                                            {m.chest && ` • Chest: ${m.chest} cm`}
                                        </div>
                                    </div>
                                </div>
                                <button
                                    onClick={() => deleteMeasurement(m.id)}
                                    className="p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg"
                                >
                                    <Trash2 className="h-4 w-4" />
                                </button>
                            </div>
                        ))
                    )}
                </div>
            </div>

            {/* Add Measurement Modal */}
            <AnimatePresence>
                {showAddModal && (
                    <AddMeasurementModal
                        onClose={() => setShowAddModal(false)}
                        onSave={(measurement) => {
                            setMeasurements(prev => [measurement, ...prev]);
                            setShowAddModal(false);
                        }}
                    />
                )}
            </AnimatePresence>
        </div>
    );
};

// Stat Card Component
const StatCard = ({ icon: Icon, label, value, change, unit, inverse }) => {
    const isPositive = change && parseFloat(change) > 0;
    const isNegative = change && parseFloat(change) < 0;
    const changeColor = inverse
        ? (isPositive ? 'text-red-500' : isNegative ? 'text-green-500' : 'text-gray-500')
        : (isPositive ? 'text-green-500' : isNegative ? 'text-red-500' : 'text-gray-500');

    return (
        <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm">
            <div className="flex items-center gap-3">
                <div className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
                    <Icon className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                </div>
                <div className="flex-1">
                    <div className="text-sm text-gray-500 dark:text-gray-400">{label}</div>
                    <div className="text-xl font-bold text-gray-900 dark:text-white">{value}</div>
                </div>
                {change && (
                    <div className={`flex items-center gap-1 text-sm font-medium ${changeColor}`}>
                        {isPositive ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
                        {isPositive ? '+' : ''}{change}
                    </div>
                )}
            </div>
        </div>
    );
};

// Add Measurement Modal
const AddMeasurementModal = ({ onClose, onSave }) => {
    const [form, setForm] = useState({
        weight: '',
        bodyFatPercentage: '',
        chest: '',
        waist: '',
        hips: '',
        leftBicep: '',
        rightBicep: '',
        leftThigh: '',
        rightThigh: '',
        notes: ''
    });
    const [saving, setSaving] = useState(false);

    const handleSubmit = async () => {
        setSaving(true);
        try {
            const measurement = {
                ...form,
                weight: form.weight ? parseFloat(form.weight) : null,
                bodyFatPercentage: form.bodyFatPercentage ? parseFloat(form.bodyFatPercentage) : null,
                chest: form.chest ? parseFloat(form.chest) : null,
                waist: form.waist ? parseFloat(form.waist) : null,
                hips: form.hips ? parseFloat(form.hips) : null,
                leftBicep: form.leftBicep ? parseFloat(form.leftBicep) : null,
                rightBicep: form.rightBicep ? parseFloat(form.rightBicep) : null,
                leftThigh: form.leftThigh ? parseFloat(form.leftThigh) : null,
                rightThigh: form.rightThigh ? parseFloat(form.rightThigh) : null,
                measurementDate: new Date().toISOString()
            };

            const response = await api.post('/measurements', measurement);
            onSave(response.data.data);
        } catch (error) {
            console.error('Failed to save:', error);
            // Still add locally for demo
            onSave({ ...form, id: Date.now().toString(), measurementDate: new Date().toISOString() });
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
                    <div className="flex justify-between items-center">
                        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Add Measurement</h2>
                        <button onClick={onClose} className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg">
                            <X className="h-5 w-5 text-gray-500" />
                        </button>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                        <InputField label="Weight (kg)" value={form.weight} onChange={v => setForm(f => ({ ...f, weight: v }))} />
                        <InputField label="Body Fat (%)" value={form.bodyFatPercentage} onChange={v => setForm(f => ({ ...f, bodyFatPercentage: v }))} />
                    </div>

                    <div className="grid grid-cols-3 gap-4">
                        <InputField label="Chest (cm)" value={form.chest} onChange={v => setForm(f => ({ ...f, chest: v }))} />
                        <InputField label="Waist (cm)" value={form.waist} onChange={v => setForm(f => ({ ...f, waist: v }))} />
                        <InputField label="Hips (cm)" value={form.hips} onChange={v => setForm(f => ({ ...f, hips: v }))} />
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                        <InputField label="Left Bicep (cm)" value={form.leftBicep} onChange={v => setForm(f => ({ ...f, leftBicep: v }))} />
                        <InputField label="Right Bicep (cm)" value={form.rightBicep} onChange={v => setForm(f => ({ ...f, rightBicep: v }))} />
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                        <InputField label="Left Thigh (cm)" value={form.leftThigh} onChange={v => setForm(f => ({ ...f, leftThigh: v }))} />
                        <InputField label="Right Thigh (cm)" value={form.rightThigh} onChange={v => setForm(f => ({ ...f, rightThigh: v }))} />
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Notes</label>
                        <textarea
                            value={form.notes}
                            onChange={(e) => setForm(f => ({ ...f, notes: e.target.value }))}
                            placeholder="Any notes..."
                            rows={2}
                            className="w-full px-4 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
                        />
                    </div>

                    <div className="flex gap-3">
                        <button
                            onClick={onClose}
                            className="flex-1 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg font-medium"
                        >
                            Cancel
                        </button>
                        <button
                            onClick={handleSubmit}
                            disabled={saving}
                            className="flex-1 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold flex items-center justify-center gap-2 disabled:opacity-50"
                        >
                            <Save className="h-5 w-5" />
                            {saving ? 'Saving...' : 'Save'}
                        </button>
                    </div>
                </div>
            </motion.div>
        </motion.div>
    );
};

const InputField = ({ label, value, onChange }) => (
    <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">{label}</label>
        <input
            type="number"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
        />
    </div>
);

// Sample data
const SAMPLE_MEASUREMENTS = [
    { id: '1', measurementDate: '2024-12-14T10:00:00', weight: 82, bodyFatPercentage: 18, chest: 102, waist: 84, leftBicep: 38, rightBicep: 38.5 },
    { id: '2', measurementDate: '2024-12-07T10:00:00', weight: 82.5, bodyFatPercentage: 18.5, chest: 101, waist: 85, leftBicep: 37.5, rightBicep: 38 },
    { id: '3', measurementDate: '2024-11-30T10:00:00', weight: 83, bodyFatPercentage: 19, chest: 100, waist: 86, leftBicep: 37, rightBicep: 37.5 },
];

export default BodyMeasurements;

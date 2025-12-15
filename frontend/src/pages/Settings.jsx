import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
    Settings as SettingsIcon,
    Scale,
    Ruler,
    Moon,
    Sun,
    Bell,
    Volume2,
    Download,
    Upload,
    Share2,
    Trash2,
    Save,
    Check,
    AlertTriangle
} from 'lucide-react';
import { useSettings } from '../contexts/SettingsContext';
import { useTheme } from '../contexts/ThemeContext';
import api from '../services/api';

const Settings = () => {
    const { settings, updateSetting } = useSettings();
    const { isDark, toggleTheme } = useTheme();
    const [exporting, setExporting] = useState(false);
    const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
    const [saved, setSaved] = useState(false);

    const handleExport = async (format) => {
        setExporting(true);
        try {
            // Fetch all user data
            const [workoutsRes, measurementsRes, templatesRes] = await Promise.all([
                api.get('/workout-logs'),
                api.get('/measurements'),
                api.get('/templates'),
            ]);

            const data = {
                exportDate: new Date().toISOString(),
                workouts: workoutsRes.data.data || [],
                measurements: measurementsRes.data.data || [],
                templates: templatesRes.data.data || [],
                settings: settings,
            };

            let content, filename, type;

            if (format === 'json') {
                content = JSON.stringify(data, null, 2);
                filename = `aithlete_export_${Date.now()}.json`;
                type = 'application/json';
            } else {
                // CSV - flatten workouts for export
                const csvRows = ['Date,Workout,Exercise,Sets,Reps,Weight,Volume'];
                data.workouts.forEach(w => {
                    w.exercises?.forEach(ex => {
                        ex.sets?.forEach(set => {
                            csvRows.push([
                                w.startTime?.split('T')[0] || '',
                                w.name || '',
                                ex.exerciseName || '',
                                1,
                                set.reps || 0,
                                set.weight || 0,
                                (set.reps || 0) * (set.weight || 0)
                            ].join(','));
                        });
                    });
                });
                content = csvRows.join('\n');
                filename = `aithlete_export_${Date.now()}.csv`;
                type = 'text/csv';
            }

            // Download file
            const blob = new Blob([content], { type });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        } catch (error) {
            console.error('Export failed:', error);
            alert('Export failed. Please try again.');
        } finally {
            setExporting(false);
        }
    };

    const handleSave = () => {
        setSaved(true);
        setTimeout(() => setSaved(false), 2000);
    };

    return (
        <div className="space-y-6 max-w-2xl mx-auto">
            {/* Header */}
            <div>
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Settings</h1>
                <p className="text-gray-600 dark:text-gray-400">Customize your experience</p>
            </div>

            {/* Units Section */}
            <SettingsSection title="Units" icon={Scale}>
                <SettingsRow
                    label="Weight Unit"
                    description="Used for exercises and body weight"
                >
                    <ToggleButtons
                        options={[
                            { value: 'kg', label: 'kg' },
                            { value: 'lbs', label: 'lbs' }
                        ]}
                        value={settings.weightUnit}
                        onChange={(v) => updateSetting('weightUnit', v)}
                    />
                </SettingsRow>
                <SettingsRow
                    label="Measurement Unit"
                    description="Used for body measurements"
                >
                    <ToggleButtons
                        options={[
                            { value: 'cm', label: 'cm' },
                            { value: 'inches', label: 'in' }
                        ]}
                        value={settings.measurementUnit}
                        onChange={(v) => updateSetting('measurementUnit', v)}
                    />
                </SettingsRow>
            </SettingsSection>

            {/* Appearance Section */}
            <SettingsSection title="Appearance" icon={isDark ? Moon : Sun}>
                <SettingsRow
                    label="Dark Mode"
                    description="Toggle dark/light theme"
                >
                    <Toggle
                        checked={isDark}
                        onChange={toggleTheme}
                    />
                </SettingsRow>
            </SettingsSection>

            {/* Notifications Section */}
            <SettingsSection title="Notifications" icon={Bell}>
                <SettingsRow
                    label="Push Notifications"
                    description="Get workout reminders"
                >
                    <Toggle
                        checked={settings.notifications}
                        onChange={(v) => updateSetting('notifications', v)}
                    />
                </SettingsRow>
                <SettingsRow
                    label="Sound Effects"
                    description="Timer and PR sounds"
                >
                    <Toggle
                        checked={settings.soundEffects}
                        onChange={(v) => updateSetting('soundEffects', v)}
                    />
                </SettingsRow>
            </SettingsSection>

            {/* Rest Timer Section */}
            <SettingsSection title="Workout Defaults" icon={SettingsIcon}>
                <SettingsRow
                    label="Default Rest Timer"
                    description="Default rest between sets"
                >
                    <select
                        value={settings.restTimerDefault}
                        onChange={(e) => updateSetting('restTimerDefault', parseInt(e.target.value))}
                        className="px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
                    >
                        <option value="30">30 seconds</option>
                        <option value="60">1 minute</option>
                        <option value="90">1.5 minutes</option>
                        <option value="120">2 minutes</option>
                        <option value="180">3 minutes</option>
                    </select>
                </SettingsRow>
            </SettingsSection>

            {/* Data Management Section */}
            <SettingsSection title="Data Management" icon={Download}>
                <SettingsRow
                    label="Export Data"
                    description="Download your workout data"
                >
                    <div className="flex gap-2">
                        <button
                            onClick={() => handleExport('json')}
                            disabled={exporting}
                            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium disabled:opacity-50"
                        >
                            {exporting ? 'Exporting...' : 'JSON'}
                        </button>
                        <button
                            onClick={() => handleExport('csv')}
                            disabled={exporting}
                            className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg text-sm font-medium disabled:opacity-50"
                        >
                            {exporting ? 'Exporting...' : 'CSV'}
                        </button>
                    </div>
                </SettingsRow>
                <SettingsRow
                    label="Import Data"
                    description="Restore from backup"
                >
                    <label className="px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg text-sm font-medium cursor-pointer hover:bg-gray-200 dark:hover:bg-gray-600">
                        <Upload className="h-4 w-4 inline mr-2" />
                        Import
                        <input type="file" accept=".json" className="hidden" />
                    </label>
                </SettingsRow>
            </SettingsSection>

            {/* Danger Zone */}
            <div className="bg-red-50 dark:bg-red-900/20 rounded-xl p-6 border border-red-200 dark:border-red-800">
                <div className="flex items-center gap-3 mb-4">
                    <AlertTriangle className="h-5 w-5 text-red-600 dark:text-red-400" />
                    <h3 className="font-semibold text-red-900 dark:text-red-100">Danger Zone</h3>
                </div>
                <div className="flex items-center justify-between">
                    <div>
                        <div className="font-medium text-red-900 dark:text-red-100">Delete All Data</div>
                        <div className="text-sm text-red-700 dark:text-red-300">Permanently remove all workouts and data</div>
                    </div>
                    <button
                        onClick={() => setShowDeleteConfirm(true)}
                        className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg text-sm font-medium flex items-center gap-2"
                    >
                        <Trash2 className="h-4 w-4" />
                        Delete
                    </button>
                </div>
            </div>

            {/* Save Button */}
            <button
                onClick={handleSave}
                className="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-semibold flex items-center justify-center gap-2 transition"
            >
                {saved ? <Check className="h-5 w-5" /> : <Save className="h-5 w-5" />}
                {saved ? 'Saved!' : 'Save Settings'}
            </button>

            {/* Delete Confirmation Modal */}
            {showDeleteConfirm && (
                <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
                    <div className="bg-white dark:bg-gray-800 rounded-xl p-6 max-w-md w-full">
                        <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">Delete All Data?</h3>
                        <p className="text-gray-600 dark:text-gray-400 mb-6">
                            This will permanently delete all your workouts, measurements, and templates. This action cannot be undone.
                        </p>
                        <div className="flex gap-3">
                            <button
                                onClick={() => setShowDeleteConfirm(false)}
                                className="flex-1 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg font-medium"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={() => {
                                    alert('Data deletion would happen here');
                                    setShowDeleteConfirm(false);
                                }}
                                className="flex-1 py-2 bg-red-600 text-white rounded-lg font-medium"
                            >
                                Delete Everything
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

// Helper Components
const SettingsSection = ({ title, icon: Icon, children }) => (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-100 dark:border-gray-700 flex items-center gap-3">
            <Icon className="h-5 w-5 text-gray-500 dark:text-gray-400" />
            <h2 className="font-semibold text-gray-900 dark:text-white">{title}</h2>
        </div>
        <div className="divide-y divide-gray-100 dark:divide-gray-700">
            {children}
        </div>
    </div>
);

const SettingsRow = ({ label, description, children }) => (
    <div className="px-6 py-4 flex items-center justify-between">
        <div>
            <div className="font-medium text-gray-900 dark:text-white">{label}</div>
            {description && (
                <div className="text-sm text-gray-500 dark:text-gray-400">{description}</div>
            )}
        </div>
        {children}
    </div>
);

const Toggle = ({ checked, onChange }) => (
    <button
        onClick={() => onChange(!checked)}
        className={`w-12 h-6 rounded-full transition-colors ${checked ? 'bg-blue-600' : 'bg-gray-300 dark:bg-gray-600'
            }`}
    >
        <motion.div
            animate={{ x: checked ? 24 : 2 }}
            className="w-5 h-5 bg-white rounded-full shadow"
        />
    </button>
);

const ToggleButtons = ({ options, value, onChange }) => (
    <div className="flex bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
        {options.map(opt => (
            <button
                key={opt.value}
                onClick={() => onChange(opt.value)}
                className={`px-4 py-1.5 rounded-md text-sm font-medium transition ${value === opt.value
                    ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm'
                    : 'text-gray-600 dark:text-gray-400'
                    }`}
            >
                {opt.label}
            </button>
        ))}
    </div>
);

export default Settings;

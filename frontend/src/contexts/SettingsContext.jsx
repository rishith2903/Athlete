import React, { useState, useEffect, createContext, useContext } from 'react';

const SettingsContext = createContext();

export const useSettings = () => {
    const context = useContext(SettingsContext);
    if (!context) {
        throw new Error('useSettings must be used within SettingsProvider');
    }
    return context;
};

export const SettingsProvider = ({ children }) => {
    const [settings, setSettings] = useState(() => {
        const saved = localStorage.getItem('aithlete_settings');
        return saved ? JSON.parse(saved) : {
            weightUnit: 'kg', // kg or lbs
            measurementUnit: 'cm', // cm or inches
            restTimerDefault: 90,
            darkMode: false,
            notifications: true,
            soundEffects: true,
        };
    });

    useEffect(() => {
        localStorage.setItem('aithlete_settings', JSON.stringify(settings));
    }, [settings]);

    const updateSetting = (key, value) => {
        setSettings(prev => ({ ...prev, [key]: value }));
    };

    // Unit conversion helpers
    const convertWeight = (value, toUnit = settings.weightUnit) => {
        if (!value) return value;
        const num = parseFloat(value);
        if (toUnit === 'lbs') {
            return Math.round(num * 2.20462 * 10) / 10;
        }
        return Math.round(num / 2.20462 * 10) / 10;
    };

    const convertMeasurement = (value, toUnit = settings.measurementUnit) => {
        if (!value) return value;
        const num = parseFloat(value);
        if (toUnit === 'inches') {
            return Math.round(num / 2.54 * 10) / 10;
        }
        return Math.round(num * 2.54 * 10) / 10;
    };

    const formatWeight = (value) => {
        if (!value) return '-';
        return `${value} ${settings.weightUnit}`;
    };

    const formatMeasurement = (value) => {
        if (!value) return '-';
        return `${value} ${settings.measurementUnit}`;
    };

    return (
        <SettingsContext.Provider value={{
            settings,
            updateSetting,
            convertWeight,
            convertMeasurement,
            formatWeight,
            formatMeasurement,
        }}>
            {children}
        </SettingsContext.Provider>
    );
};

export default SettingsContext;

import React, { createContext, useContext, useState, useEffect } from 'react';

const ThemeContext = createContext();

export const useTheme = () => {
    const context = useContext(ThemeContext);
    if (!context) {
        throw new Error('useTheme must be used within a ThemeProvider');
    }
    return context;
};

export const ThemeProvider = ({ children }) => {
    const [isDark, setIsDark] = useState(() => {
        // Check localStorage first, default to light mode
        const saved = localStorage.getItem('theme');
        if (saved) {
            return saved === 'dark';
        }
        // Default to light mode (not system preference)
        return false;
    });

    useEffect(() => {
        // Update document class and localStorage
        if (isDark) {
            document.documentElement.classList.add('dark');
            localStorage.setItem('theme', 'dark');
        } else {
            document.documentElement.classList.remove('dark');
            localStorage.setItem('theme', 'light');
        }

        // Also sync with old aithlete_settings if it exists
        try {
            const oldSettings = localStorage.getItem('aithlete_settings');
            if (oldSettings) {
                const parsed = JSON.parse(oldSettings);
                if (parsed.darkMode !== isDark) {
                    parsed.darkMode = isDark;
                    localStorage.setItem('aithlete_settings', JSON.stringify(parsed));
                }
            }
        } catch (e) {
            // Ignore parsing errors
        }
    }, [isDark]);

    const toggleTheme = () => setIsDark(prev => !prev);

    return (
        <ThemeContext.Provider value={{ isDark, toggleTheme }}>
            {children}
        </ThemeContext.Provider>
    );
};

export default ThemeContext;

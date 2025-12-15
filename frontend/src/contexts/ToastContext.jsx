import React, { createContext, useContext, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Trophy, Flame, CheckCircle, AlertCircle, Info } from 'lucide-react';

const ToastContext = createContext();

export const useToast = () => {
    const context = useContext(ToastContext);
    if (!context) {
        throw new Error('useToast must be used within ToastProvider');
    }
    return context;
};

export const ToastProvider = ({ children }) => {
    const [toasts, setToasts] = useState([]);

    const addToast = useCallback((toast) => {
        const id = Date.now();
        const newToast = { id, ...toast };
        setToasts(prev => [...prev, newToast]);

        // Auto dismiss after duration
        setTimeout(() => {
            removeToast(id);
        }, toast.duration || 5000);

        return id;
    }, []);

    const removeToast = useCallback((id) => {
        setToasts(prev => prev.filter(t => t.id !== id));
    }, []);

    // Convenience methods
    const success = (message, options = {}) => addToast({ type: 'success', message, ...options });
    const error = (message, options = {}) => addToast({ type: 'error', message, ...options });
    const info = (message, options = {}) => addToast({ type: 'info', message, ...options });
    const pr = (exercise, weight, options = {}) => addToast({
        type: 'pr',
        message: `New PR: ${exercise}`,
        subtext: `${weight} kg`,
        ...options
    });
    const achievement = (name, description, options = {}) => addToast({
        type: 'achievement',
        message: name,
        subtext: description,
        duration: 7000,
        ...options
    });
    const streak = (days, options = {}) => addToast({
        type: 'streak',
        message: `${days} Day Streak!`,
        subtext: 'Keep up the great work!',
        ...options
    });

    return (
        <ToastContext.Provider value={{ addToast, removeToast, success, error, info, pr, achievement, streak }}>
            {children}
            <ToastContainer toasts={toasts} removeToast={removeToast} />
        </ToastContext.Provider>
    );
};

// Toast Container
const ToastContainer = ({ toasts, removeToast }) => (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-sm">
        <AnimatePresence>
            {toasts.map(toast => (
                <Toast key={toast.id} toast={toast} onDismiss={() => removeToast(toast.id)} />
            ))}
        </AnimatePresence>
    </div>
);

// Individual Toast
const Toast = ({ toast, onDismiss }) => {
    const getIcon = () => {
        switch (toast.type) {
            case 'success': return <CheckCircle className="h-5 w-5 text-green-500" />;
            case 'error': return <AlertCircle className="h-5 w-5 text-red-500" />;
            case 'pr': return <Trophy className="h-5 w-5 text-yellow-500" />;
            case 'achievement': return <Trophy className="h-5 w-5 text-purple-500" />;
            case 'streak': return <Flame className="h-5 w-5 text-orange-500" />;
            default: return <Info className="h-5 w-5 text-blue-500" />;
        }
    };

    const getBgColor = () => {
        switch (toast.type) {
            case 'success': return 'bg-green-50 dark:bg-green-900/30 border-green-200 dark:border-green-800';
            case 'error': return 'bg-red-50 dark:bg-red-900/30 border-red-200 dark:border-red-800';
            case 'pr': return 'bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/30 dark:to-orange-900/30 border-yellow-300 dark:border-yellow-700';
            case 'achievement': return 'bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/30 dark:to-pink-900/30 border-purple-300 dark:border-purple-700';
            case 'streak': return 'bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-900/30 dark:to-red-900/30 border-orange-300 dark:border-orange-700';
            default: return 'bg-blue-50 dark:bg-blue-900/30 border-blue-200 dark:border-blue-800';
        }
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 50, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, x: 100, scale: 0.9 }}
            className={`p-4 rounded-xl border shadow-lg ${getBgColor()} min-w-[280px]`}
        >
            <div className="flex items-start gap-3">
                <div className="flex-shrink-0 mt-0.5">
                    {toast.icon || getIcon()}
                </div>
                <div className="flex-1 min-w-0">
                    <p className="font-semibold text-gray-900 dark:text-white">{toast.message}</p>
                    {toast.subtext && (
                        <p className="text-sm text-gray-600 dark:text-gray-300 mt-0.5">{toast.subtext}</p>
                    )}
                </div>
                <button
                    onClick={onDismiss}
                    className="flex-shrink-0 p-1 rounded-lg hover:bg-black/5 dark:hover:bg-white/5"
                >
                    <X className="h-4 w-4 text-gray-400" />
                </button>
            </div>

            {/* Progress bar for auto-dismiss */}
            <motion.div
                initial={{ width: '100%' }}
                animate={{ width: '0%' }}
                transition={{ duration: (toast.duration || 5000) / 1000, ease: 'linear' }}
                className="h-1 bg-current opacity-20 rounded-full mt-3"
            />
        </motion.div>
    );
};

export default ToastContext;

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, X, ExternalLink, Youtube } from 'lucide-react';

/**
 * Exercise Demo Video Component
 * Shows exercise demonstration videos/GIFs from online sources
 */
const ExerciseDemo = ({ exercise, onClose }) => {
    const [loading, setLoading] = useState(true);

    // Map exercise names to YouTube video IDs (short form demos)
    const videoMap = {
        'bench press': 'rT7DgCr-3pg',
        'squat': 'Dy28eq2PjcQ',
        'deadlift': 'op9kVnSso6Q',
        'overhead press': 'CnBmiBqp-AI',
        'barbell row': 'FWJR5Ve8bnQ',
        'pull-up': 'eGo4IYlbE5g',
        'lat pulldown': '43jJtPc_bCY',
        'bicep curl': 'ykJmrZ5v0Oo',
        'tricep extension': '2-LAMcpzODU',
        'leg press': 'IZxyjW7MPJQ',
        'leg curl': 'ELOCsoDSmrg',
        'leg extension': 'YyvSfVjQeL0',
        'calf raise': 'gwLzBJYoWlI',
        'plank': 'ASdvN_XEl_c',
        'crunch': 'Xyd_fa5zoEU',
        'russian twist': 'wkD8rjkodUI',
        'dumbbell fly': 'eozdVDA78K0',
        'lateral raise': 'XPPfnSEATJA',
        'front raise': 'gzDsm1FhBdQ',
        'face pull': 'rep-qVOkqgk',
        'shrug': 'cJRVVxmytaM',
        'hip thrust': 'SEdqd1n0p7Y',
        'lunges': 'QOVaHwm-Q6U',
        'romanian deadlift': 'hCDzSR6bW10',
        'incline bench press': '8iPEnn-ltC8',
        'dips': 'yN6Q1UI_xkE',
        'push-up': 'IODxDxX7oi4',
        'cable crossover': 'taI4XduLpTk',
    };

    const getVideoId = (name) => {
        const lowerName = name?.toLowerCase() || '';
        return videoMap[lowerName] || Object.entries(videoMap).find(([key]) =>
            lowerName.includes(key) || key.includes(lowerName)
        )?.[1];
    };

    const videoId = getVideoId(exercise?.name);

    return (
        <AnimatePresence>
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4"
                onClick={onClose}
            >
                <motion.div
                    initial={{ scale: 0.9, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale: 0.9, opacity: 0 }}
                    onClick={(e) => e.stopPropagation()}
                    className="bg-white dark:bg-gray-800 rounded-2xl max-w-2xl w-full overflow-hidden"
                >
                    {/* Header */}
                    <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
                        <div className="flex items-center gap-3">
                            <div className="p-2 bg-red-100 dark:bg-red-900/30 rounded-lg">
                                <Youtube className="h-5 w-5 text-red-600 dark:text-red-400" />
                            </div>
                            <div>
                                <h3 className="font-semibold text-gray-900 dark:text-white">{exercise?.name || 'Exercise Demo'}</h3>
                                <p className="text-sm text-gray-500 dark:text-gray-400">Watch proper form</p>
                            </div>
                        </div>
                        <button
                            onClick={onClose}
                            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg"
                        >
                            <X className="h-5 w-5 text-gray-500" />
                        </button>
                    </div>

                    {/* Video */}
                    <div className="aspect-video bg-black relative">
                        {loading && (
                            <div className="absolute inset-0 flex items-center justify-center">
                                <div className="animate-spin h-8 w-8 border-2 border-white border-t-transparent rounded-full"></div>
                            </div>
                        )}
                        {videoId ? (
                            <iframe
                                src={`https://www.youtube.com/embed/${videoId}?autoplay=1&rel=0`}
                                title={`${exercise?.name} Demo`}
                                className="w-full h-full"
                                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                                allowFullScreen
                                onLoad={() => setLoading(false)}
                            />
                        ) : (
                            <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-400">
                                <Play className="h-16 w-16 mb-4 opacity-50" />
                                <p>No demo video available for this exercise</p>
                                <a
                                    href={`https://www.youtube.com/results?search_query=${encodeURIComponent(exercise?.name + ' exercise form')}`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="mt-4 px-4 py-2 bg-red-600 text-white rounded-lg flex items-center gap-2 hover:bg-red-700"
                                >
                                    <ExternalLink className="h-4 w-4" />
                                    Search on YouTube
                                </a>
                            </div>
                        )}
                    </div>

                    {/* Tips */}
                    {exercise?.instructions && (
                        <div className="p-4 bg-gray-50 dark:bg-gray-900">
                            <h4 className="font-medium text-gray-900 dark:text-white mb-2">Key Points</h4>
                            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                                {(Array.isArray(exercise.instructions) ? exercise.instructions : [exercise.instructions])
                                    .slice(0, 4)
                                    .map((tip, i) => (
                                        <li key={i} className="flex items-start gap-2">
                                            <span className="text-blue-500">•</span>
                                            {tip}
                                        </li>
                                    ))}
                            </ul>
                        </div>
                    )}
                </motion.div>
            </motion.div>
        </AnimatePresence>
    );
};

/**
 * Demo Button Component (for use in exercise cards)
 */
export const DemoButton = ({ exercise, className = '' }) => {
    const [showDemo, setShowDemo] = useState(false);

    return (
        <>
            <button
                onClick={() => setShowDemo(true)}
                className={`p-2 rounded-lg bg-red-100 hover:bg-red-200 dark:bg-red-900/30 dark:hover:bg-red-900/50 ${className}`}
                title="Watch demo"
            >
                <Play className="h-4 w-4 text-red-600 dark:text-red-400" />
            </button>

            {showDemo && (
                <ExerciseDemo exercise={exercise} onClose={() => setShowDemo(false)} />
            )}
        </>
    );
};

export default ExerciseDemo;

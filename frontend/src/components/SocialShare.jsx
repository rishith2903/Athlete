import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Share2, Copy, Check, Twitter, MessageCircle, Link } from 'lucide-react';

/**
 * Social Sharing Component
 * Share workout summaries via Web Share API or social links
 */
const SocialShare = ({ workout, onClose }) => {
    const [copied, setCopied] = useState(false);

    const shareText = workout ?
        `🏋️ Just crushed it at the gym!\n\n` +
        `💪 ${workout.name || 'Workout'}\n` +
        `📊 ${workout.totalSets || 0} sets | ${workout.totalReps || 0} reps\n` +
        `🔥 ${workout.totalVolume || 0} kg total volume\n` +
        `⏱️ ${workout.duration || 0} minutes\n\n` +
        `Tracked with AIthlete 🤖\n#AIthlete #Fitness #GymLife`
        : 'Check out my workout on AIthlete!';

    const shareUrl = window.location.origin;

    const handleNativeShare = async () => {
        if (navigator.share) {
            try {
                await navigator.share({
                    title: 'My Workout - AIthlete',
                    text: shareText,
                    url: shareUrl,
                });
            } catch (error) {
                if (error.name !== 'AbortError') {
                    console.error('Share failed:', error);
                }
            }
        }
    };

    const handleCopy = async () => {
        try {
            await navigator.clipboard.writeText(shareText);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch (error) {
            console.error('Copy failed:', error);
        }
    };

    const shareLinks = [
        {
            name: 'Twitter',
            icon: Twitter,
            color: 'bg-sky-500 hover:bg-sky-600',
            url: `https://twitter.com/intent/tweet?text=${encodeURIComponent(shareText)}&url=${encodeURIComponent(shareUrl)}`,
        },
        {
            name: 'WhatsApp',
            icon: MessageCircle,
            color: 'bg-green-500 hover:bg-green-600',
            url: `https://wa.me/?text=${encodeURIComponent(shareText + '\n' + shareUrl)}`,
        },
    ];

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
                className="bg-white dark:bg-gray-800 rounded-2xl max-w-md w-full p-6 space-y-6"
            >
                <div className="flex items-center gap-3">
                    <Share2 className="h-6 w-6 text-blue-500" />
                    <h2 className="text-xl font-bold text-gray-900 dark:text-white">Share Workout</h2>
                </div>

                {/* Preview */}
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 text-sm whitespace-pre-wrap text-gray-700 dark:text-gray-300 max-h-48 overflow-y-auto">
                    {shareText}
                </div>

                {/* Share Buttons */}
                <div className="space-y-3">
                    {/* Native Share (mobile) */}
                    {navigator.share && (
                        <button
                            onClick={handleNativeShare}
                            className="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-semibold flex items-center justify-center gap-2"
                        >
                            <Share2 className="h-5 w-5" />
                            Share
                        </button>
                    )}

                    {/* Social Links */}
                    <div className="flex gap-3">
                        {shareLinks.map(link => (
                            <a
                                key={link.name}
                                href={link.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className={`flex-1 py-3 text-white rounded-xl font-semibold flex items-center justify-center gap-2 ${link.color}`}
                            >
                                <link.icon className="h-5 w-5" />
                                {link.name}
                            </a>
                        ))}
                    </div>

                    {/* Copy Button */}
                    <button
                        onClick={handleCopy}
                        className="w-full py-3 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-xl font-medium flex items-center justify-center gap-2 hover:bg-gray-200 dark:hover:bg-gray-600"
                    >
                        {copied ? (
                            <>
                                <Check className="h-5 w-5 text-green-500" />
                                Copied!
                            </>
                        ) : (
                            <>
                                <Copy className="h-5 w-5" />
                                Copy to Clipboard
                            </>
                        )}
                    </button>
                </div>

                {/* Close */}
                <button
                    onClick={onClose}
                    className="w-full py-2 text-gray-500 dark:text-gray-400 text-sm"
                >
                    Close
                </button>
            </motion.div>
        </motion.div>
    );
};

/**
 * Share Button Component (for inline use)
 */
export const ShareButton = ({ workout, className = '' }) => {
    const [showModal, setShowModal] = useState(false);

    return (
        <>
            <button
                onClick={() => setShowModal(true)}
                className={`p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 ${className}`}
                title="Share workout"
            >
                <Share2 className="h-5 w-5 text-gray-500 dark:text-gray-400" />
            </button>

            <AnimatePresence>
                {showModal && (
                    <SocialShare workout={workout} onClose={() => setShowModal(false)} />
                )}
            </AnimatePresence>
        </>
    );
};

export default SocialShare;

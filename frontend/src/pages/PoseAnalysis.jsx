import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Camera,
  CameraOff,
  Play,
  Pause,
  RotateCcw,
  CheckCircle,
  AlertCircle,
  XCircle,
  Activity,
  Target,
  AlertTriangle
} from 'lucide-react';

// Direct AI Service URL
const AI_SERVICE_URL = 'http://localhost:8000';

const PoseAnalysis = () => {
  const [isStreaming, setIsStreaming] = useState(false);
  const [selectedExercise, setSelectedExercise] = useState('squat');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [error, setError] = useState(null);
  const [sessionHistory, setSessionHistory] = useState([]); // Track all analyses
  const [showReport, setShowReport] = useState(false); // Show final report
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const analysisIntervalRef = useRef(null);

  const exerciseCategories = {
    'Legs': [
      { id: 'squat', name: 'Squat' },
      { id: 'barbell_squat', name: 'Barbell Squat' },
      { id: 'goblet_squat', name: 'Goblet Squat' },
      { id: 'front_squat', name: 'Front Squat' },
      { id: 'lunge', name: 'Lunge' },
      { id: 'walking_lunge', name: 'Walking Lunge' },
      { id: 'bulgarian_split_squat', name: 'Bulgarian Split Squat' },
      { id: 'leg_press', name: 'Leg Press' },
      { id: 'leg_curl', name: 'Leg Curl' },
      { id: 'leg_extension', name: 'Leg Extension' },
      { id: 'calf_raise', name: 'Calf Raise' },
      { id: 'romanian_deadlift', name: 'Romanian Deadlift' },
      { id: 'hip_thrust', name: 'Hip Thrust' },
      { id: 'step_up', name: 'Step Up' },
    ],
    'Chest': [
      { id: 'pushup', name: 'Push-up' },
      { id: 'bench_press', name: 'Bench Press' },
      { id: 'incline_bench_press', name: 'Incline Bench Press' },
      { id: 'decline_bench_press', name: 'Decline Bench Press' },
      { id: 'dumbbell_press', name: 'Dumbbell Press' },
      { id: 'dumbbell_fly', name: 'Dumbbell Fly' },
      { id: 'cable_crossover', name: 'Cable Crossover' },
      { id: 'chest_dip', name: 'Chest Dip' },
      { id: 'diamond_pushup', name: 'Diamond Push-up' },
    ],
    'Back': [
      { id: 'deadlift', name: 'Deadlift' },
      { id: 'barbell_row', name: 'Barbell Row' },
      { id: 'bent_over_row', name: 'Bent Over Row' },
      { id: 'pull_up', name: 'Pull-up' },
      { id: 'chin_up', name: 'Chin-up' },
      { id: 'lat_pulldown', name: 'Lat Pulldown' },
      { id: 'seated_row', name: 'Seated Row' },
      { id: 'face_pull', name: 'Face Pull' },
      { id: 'superman', name: 'Superman' },
      { id: 'back_extension', name: 'Back Extension' },
    ],
    'Shoulders': [
      { id: 'overhead_press', name: 'Overhead Press' },
      { id: 'shoulder_press', name: 'Shoulder Press' },
      { id: 'lateral_raise', name: 'Lateral Raise' },
      { id: 'front_raise', name: 'Front Raise' },
      { id: 'rear_delt_fly', name: 'Rear Delt Fly' },
      { id: 'arnold_press', name: 'Arnold Press' },
      { id: 'upright_row', name: 'Upright Row' },
      { id: 'shrug', name: 'Shrug' },
    ],
    'Arms': [
      { id: 'bicep_curl', name: 'Bicep Curl' },
      { id: 'hammer_curl', name: 'Hammer Curl' },
      { id: 'preacher_curl', name: 'Preacher Curl' },
      { id: 'tricep_dip', name: 'Tricep Dip' },
      { id: 'tricep_pushdown', name: 'Tricep Pushdown' },
      { id: 'skull_crusher', name: 'Skull Crusher' },
      { id: 'close_grip_bench', name: 'Close Grip Bench' },
      { id: 'concentration_curl', name: 'Concentration Curl' },
    ],
    'Core': [
      { id: 'plank', name: 'Plank' },
      { id: 'side_plank', name: 'Side Plank' },
      { id: 'crunch', name: 'Crunch' },
      { id: 'sit_up', name: 'Sit-up' },
      { id: 'russian_twist', name: 'Russian Twist' },
      { id: 'leg_raise', name: 'Leg Raise' },
      { id: 'hanging_leg_raise', name: 'Hanging Leg Raise' },
      { id: 'mountain_climber', name: 'Mountain Climber' },
      { id: 'dead_bug', name: 'Dead Bug' },
      { id: 'bicycle_crunch', name: 'Bicycle Crunch' },
      { id: 'ab_rollout', name: 'Ab Rollout' },
    ],
    'Full Body': [
      { id: 'burpee', name: 'Burpee' },
      { id: 'clean', name: 'Clean' },
      { id: 'snatch', name: 'Snatch' },
      { id: 'thruster', name: 'Thruster' },
      { id: 'kettlebell_swing', name: 'Kettlebell Swing' },
      { id: 'turkish_getup', name: 'Turkish Get-up' },
    ],
  };

  const [selectedCategory, setSelectedCategory] = useState('Legs');
  const exercises = exerciseCategories[selectedCategory] || [];

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  const startCamera = async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' }
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setIsStreaming(true);
      }
    } catch (err) {
      console.error('Camera error:', err);
      setError('Unable to access camera. Please check permissions.');
    }
  };

  const stopCamera = () => {
    if (analysisIntervalRef.current) {
      clearInterval(analysisIntervalRef.current);
      analysisIntervalRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
    setIsAnalyzing(false);
  };

  const captureAndAnalyze = async () => {
    if (!videoRef.current || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const video = videoRef.current;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    return new Promise((resolve) => {
      canvas.toBlob(async (blob) => {
        if (!blob) {
          resolve();
          return;
        }

        try {
          const formData = new FormData();
          formData.append('file', blob, 'frame.jpg');
          formData.append('exercise_type', selectedExercise);

          console.log('Sending to AI service...');
          const response = await fetch(`${AI_SERVICE_URL}/pose/analyze`, {
            method: 'POST',
            body: formData,
          });

          if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
          }

          const data = await response.json();
          console.log('AI Response:', data);

          if (data.success) {
            setAnalysisResult(data);
            // Add to session history
            setSessionHistory(prev => [...prev, {
              timestamp: new Date(),
              score: data.analysis?.overallScore || 0,
              formAnalysis: data.analysis?.formAnalysis || {},
              corrections: data.analysis?.corrections || [],
              injuryRisk: data.analysis?.injuryRisk || 'unknown'
            }]);
            setError(null);
          } else {
            setError('Analysis failed');
          }
        } catch (err) {
          console.error('Analysis error:', err);
          setError(`Analysis error: ${err.message}`);
        }
        resolve();
      }, 'image/jpeg', 0.8);
    });
  };

  const startAnalysis = () => {
    setIsAnalyzing(true);
    setError(null);
    setShowReport(false);
    setSessionHistory([]); // Reset history for new session
    // Analyze immediately and then every 2 seconds
    captureAndAnalyze();
    analysisIntervalRef.current = setInterval(captureAndAnalyze, 2000);
  };

  const stopAnalysis = () => {
    if (analysisIntervalRef.current) {
      clearInterval(analysisIntervalRef.current);
      analysisIntervalRef.current = null;
    }
    setIsAnalyzing(false);
    // Show report if we have any history
    if (sessionHistory.length > 0) {
      setShowReport(true);
    }
  };

  const resetSession = () => {
    stopAnalysis();
    setAnalysisResult(null);
    setError(null);
    setSessionHistory([]);
    setShowReport(false);
  };

  // Calculate session summary
  const getSessionSummary = () => {
    if (sessionHistory.length === 0) return null;

    const scores = sessionHistory.map(h => h.score);
    const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
    const minScore = Math.min(...scores);
    const maxScore = Math.max(...scores);

    // Count all corrections
    const allCorrections = sessionHistory.flatMap(h => h.corrections);
    const correctionCounts = allCorrections.reduce((acc, c) => {
      acc[c] = (acc[c] || 0) + 1;
      return acc;
    }, {});

    // Get most common corrections
    const topCorrections = Object.entries(correctionCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([correction, count]) => ({ correction, count }));

    // Get injury risk stats
    const riskCounts = sessionHistory.reduce((acc, h) => {
      acc[h.injuryRisk] = (acc[h.injuryRisk] || 0) + 1;
      return acc;
    }, {});

    return {
      totalReps: sessionHistory.length,
      avgScore: avgScore.toFixed(1),
      minScore: minScore.toFixed(1),
      maxScore: maxScore.toFixed(1),
      topCorrections,
      riskCounts,
      improvement: scores.length > 1 ? (scores[scores.length - 1] - scores[0]).toFixed(1) : 0
    };
  };

  const getScoreColor = (score) => {
    if (score >= 80) return 'text-green-500';
    if (score >= 60) return 'text-yellow-500';
    return 'text-red-500';
  };

  const getScoreBg = (score) => {
    if (score >= 80) return 'bg-green-100 dark:bg-green-900/30';
    if (score >= 60) return 'bg-yellow-100 dark:bg-yellow-900/30';
    return 'bg-red-100 dark:bg-red-900/30';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">AI Form Check</h1>
        <p className="mt-1 text-gray-600 dark:text-gray-400">Get real-time feedback on your exercise form</p>
      </div>

      {/* Exercise Selector */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
          Select Exercise ({Object.values(exerciseCategories).flat().length} exercises available)
        </label>

        {/* Category Tabs */}
        <div className="flex flex-wrap gap-2 mb-4 pb-3 border-b border-gray-200 dark:border-gray-700">
          {Object.keys(exerciseCategories).map(category => (
            <button
              key={category}
              onClick={() => {
                setSelectedCategory(category);
                setSelectedExercise(exerciseCategories[category][0]?.id);
              }}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${selectedCategory === category
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200'
                }`}
            >
              {category}
            </button>
          ))}
        </div>

        {/* Exercise Buttons */}
        <div className="flex flex-wrap gap-2">
          {exercises.map(ex => (
            <button
              key={ex.id}
              onClick={() => setSelectedExercise(ex.id)}
              className={`px-3 py-2 rounded-lg text-sm font-medium transition-all ${selectedExercise === ex.id
                ? 'bg-green-600 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300'
                }`}
            >
              {ex.name}
            </button>
          ))}
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Camera Section */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm overflow-hidden">
          <div className="relative aspect-video bg-gray-900">
            {/* Always render video element */}
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={`w-full h-full object-cover ${!isStreaming ? 'hidden' : ''}`}
            />
            <canvas ref={canvasRef} className="hidden" />

            {/* Camera off state */}
            {!isStreaming && (
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <Camera className="h-16 w-16 text-gray-500 mb-4" />
                <p className="text-gray-400 mb-4">Camera is off</p>
              </div>
            )}

            {/* Analyzing indicator */}
            {isAnalyzing && (
              <div className="absolute top-4 right-4 bg-red-500 text-white px-3 py-1 rounded-full text-sm flex items-center gap-2">
                <span className="w-2 h-2 bg-white rounded-full animate-pulse"></span>
                Analyzing...
              </div>
            )}

            {/* Exercise label */}
            {isStreaming && (
              <div className="absolute top-4 left-4 bg-black/60 text-white px-3 py-1 rounded-lg text-sm">
                {exercises.find(e => e.id === selectedExercise)?.name}
              </div>
            )}
          </div>

          {/* Controls */}
          <div className="p-4 bg-gray-50 dark:bg-gray-700 flex items-center justify-center gap-3">
            {!isStreaming ? (
              <button
                onClick={startCamera}
                className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700"
              >
                <Camera className="h-5 w-5" />
                Start Camera
              </button>
            ) : (
              <>
                <button
                  onClick={isAnalyzing ? stopAnalysis : startAnalysis}
                  className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium ${isAnalyzing
                    ? 'bg-red-600 text-white hover:bg-red-700'
                    : 'bg-green-600 text-white hover:bg-green-700'
                    }`}
                >
                  {isAnalyzing ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5" />}
                  {isAnalyzing ? 'Stop' : 'Analyze'}
                </button>
                <button
                  onClick={resetSession}
                  className="flex items-center gap-2 px-4 py-3 bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-white rounded-lg hover:bg-gray-300"
                >
                  <RotateCcw className="h-5 w-5" />
                  Reset
                </button>
                <button
                  onClick={stopCamera}
                  className="flex items-center gap-2 px-4 py-3 bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-white rounded-lg hover:bg-gray-300"
                >
                  <CameraOff className="h-5 w-5" />
                  Stop
                </button>
              </>
            )}
          </div>
        </div>

        {/* Results Section */}
        <div className="space-y-4">
          {/* Analyzing Indicator */}
          {isAnalyzing && !analysisResult && (
            <div className="bg-blue-50 dark:bg-blue-900/30 rounded-xl p-4 text-center">
              <div className="animate-pulse">
                <Activity className="h-8 w-8 mx-auto text-blue-500 mb-2" />
                <p className="text-blue-700 dark:text-blue-300">Analyzing your pose...</p>
                <p className="text-xs text-blue-500 mt-1">Make sure your full body is visible</p>
              </div>
            </div>
          )}

          {/* Error Display */}
          {error && (
            <div className="bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-xl p-4">
              <div className="flex items-center gap-2 text-red-600 dark:text-red-400">
                <XCircle className="h-5 w-5" />
                <span>{error}</span>
              </div>
            </div>
          )}

          {/* Score Card */}
          {analysisResult?.analysis && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className={`rounded-xl p-6 ${getScoreBg(analysisResult.analysis.overallScore)}`}
            >
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Form Score</h3>
                <Activity className={`h-6 w-6 ${getScoreColor(analysisResult.analysis.overallScore)}`} />
              </div>
              <div className={`text-5xl font-bold ${getScoreColor(analysisResult.analysis.overallScore)}`}>
                {analysisResult.analysis.overallScore}
                <span className="text-2xl text-gray-500">/100</span>
              </div>
              <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
                Injury Risk: <span className="font-medium capitalize">{analysisResult.analysis.injuryRisk}</span>
              </p>
            </motion.div>
          )}

          {/* Form Analysis Details */}
          {analysisResult?.analysis?.formAnalysis && (
            <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                <Target className="h-5 w-5 text-blue-500" />
                Form Analysis
              </h3>
              <div className="space-y-3">
                {Object.entries(analysisResult.analysis.formAnalysis).map(([key, value]) => (
                  <div key={key} className="flex justify-between items-center py-2 border-b border-gray-100 dark:border-gray-700 last:border-0">
                    <span className="text-gray-600 dark:text-gray-400 capitalize">
                      {key.replace(/([A-Z])/g, ' $1').trim()}
                    </span>
                    <span className={`font-medium ${value === 'good' || value === 'aligned' || (typeof value === 'number' && value >= 85)
                      ? 'text-green-500'
                      : typeof value === 'number' && value >= 70
                        ? 'text-yellow-500'
                        : value === 'needs_improvement'
                          ? 'text-yellow-500'
                          : 'text-gray-900 dark:text-white'
                      }`}>
                      {typeof value === 'number' ? value.toFixed(1) : value.replace('_', ' ')}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Corrections */}
          {analysisResult?.analysis?.corrections?.length > 0 && (
            <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                <AlertTriangle className="h-5 w-5 text-yellow-500" />
                Feedback
              </h3>
              <ul className="space-y-2">
                {analysisResult.analysis.corrections.map((correction, idx) => (
                  <li key={idx} className="flex items-start gap-2">
                    {correction.includes('Great') ? (
                      <CheckCircle className="h-5 w-5 text-green-500 mt-0.5 flex-shrink-0" />
                    ) : (
                      <AlertCircle className="h-5 w-5 text-yellow-500 mt-0.5 flex-shrink-0" />
                    )}
                    <span className="text-gray-700 dark:text-gray-300">{correction}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Instructions when no results */}
          {!analysisResult && !error && !showReport && !isAnalyzing && (
            <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-xl p-4">
              <h4 className="font-medium text-blue-900 dark:text-blue-100 mb-2">How to use:</h4>
              <ol className="text-sm text-blue-700 dark:text-blue-300 space-y-1 list-decimal list-inside">
                <li>Select your exercise type above</li>
                <li>Click "Start Camera" to enable webcam</li>
                <li>Position yourself in frame</li>
                <li>Click "Analyze" for real-time feedback</li>
              </ol>
            </div>
          )}
        </div>
      </div>

      {/* Session Report Modal */}
      {showReport && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
          onClick={() => setShowReport(false)}
        >
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="bg-white dark:bg-gray-800 rounded-2xl p-6 max-w-lg w-full max-h-[90vh] overflow-y-auto"
            onClick={e => e.stopPropagation()}
          >
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
              <Activity className="h-7 w-7 text-blue-500" />
              Session Report
            </h2>

            {(() => {
              const summary = getSessionSummary();
              if (!summary) return <p>No data recorded</p>;

              return (
                <div className="space-y-6">
                  {/* Main Stats */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-blue-50 dark:bg-blue-900/30 rounded-xl p-4 text-center">
                      <p className="text-3xl font-bold text-blue-600">{summary.totalReps}</p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Total Reps Analyzed</p>
                    </div>
                    <div className={`rounded-xl p-4 text-center ${getScoreBg(parseFloat(summary.avgScore))}`}>
                      <p className={`text-3xl font-bold ${getScoreColor(parseFloat(summary.avgScore))}`}>
                        {summary.avgScore}
                      </p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Average Score</p>
                    </div>
                  </div>

                  {/* Score Range */}
                  <div className="bg-gray-50 dark:bg-gray-700/50 rounded-xl p-4">
                    <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Score Range</h3>
                    <div className="flex justify-between items-center">
                      <div className="text-center">
                        <p className="text-xl font-bold text-red-500">{summary.minScore}</p>
                        <p className="text-xs text-gray-500">Lowest</p>
                      </div>
                      <div className="flex-1 mx-4 h-2 bg-gradient-to-r from-red-400 via-yellow-400 to-green-400 rounded-full" />
                      <div className="text-center">
                        <p className="text-xl font-bold text-green-500">{summary.maxScore}</p>
                        <p className="text-xs text-gray-500">Highest</p>
                      </div>
                    </div>
                    {parseFloat(summary.improvement) !== 0 && (
                      <p className={`text-center mt-3 text-sm font-medium ${parseFloat(summary.improvement) > 0 ? 'text-green-500' : 'text-red-500'
                        }`}>
                        {parseFloat(summary.improvement) > 0 ? '📈' : '📉'}
                        {parseFloat(summary.improvement) > 0 ? '+' : ''}{summary.improvement} point change from start to end
                      </p>
                    )}
                  </div>

                  {/* Top Issues */}
                  {summary.topCorrections.length > 0 && (
                    <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-xl p-4">
                      <h3 className="text-sm font-medium text-yellow-800 dark:text-yellow-200 mb-3">
                        Areas to Improve
                      </h3>
                      <ul className="space-y-2">
                        {summary.topCorrections.map((item, idx) => (
                          <li key={idx} className="flex items-start gap-2 text-sm">
                            <span className="bg-yellow-200 dark:bg-yellow-800 text-yellow-800 dark:text-yellow-200 px-2 py-0.5 rounded text-xs font-medium">
                              {item.count}x
                            </span>
                            <span className="text-gray-700 dark:text-gray-300">{item.correction}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Injury Risk Summary */}
                  <div className="bg-gray-50 dark:bg-gray-700/50 rounded-xl p-4">
                    <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Injury Risk Assessment</h3>
                    <div className="flex gap-2">
                      {Object.entries(summary.riskCounts).map(([risk, count]) => (
                        <span key={risk} className={`px-3 py-1 rounded-full text-sm font-medium ${risk === 'low' ? 'bg-green-100 text-green-700' :
                          risk === 'medium' ? 'bg-yellow-100 text-yellow-700' :
                            'bg-red-100 text-red-700'
                          }`}>
                          {risk}: {count}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Close Button */}
                  <button
                    onClick={() => setShowReport(false)}
                    className="w-full py-3 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700 transition-colors"
                  >
                    Close Report
                  </button>
                </div>
              );
            })()}
          </motion.div>
        </motion.div>
      )}
    </div>
  );
};

export default PoseAnalysis;
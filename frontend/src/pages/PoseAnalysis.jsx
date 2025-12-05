import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Camera, 
  CameraOff, 
  Play, 
  Pause, 
  RotateCcw,
  CheckCircle,
  AlertCircle,
  XCircle,
  Info
} from 'lucide-react';
import { poseAPI } from '../services/api';

const PoseAnalysis = () => {
  const [isStreaming, setIsStreaming] = useState(false);
  const [selectedExercise, setSelectedExercise] = useState('squat');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [feedback, setFeedback] = useState(null);
  const [repCount, setRepCount] = useState(0);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  const exercises = [
    { id: 'squat', name: 'Squat', category: 'Lower Body' },
    { id: 'pushup', name: 'Push-up', category: 'Upper Body' },
    { id: 'plank', name: 'Plank', category: 'Core' },
    { id: 'lunge', name: 'Lunge', category: 'Lower Body' },
    { id: 'deadlift', name: 'Deadlift', category: 'Full Body' },
    { id: 'shoulder_press', name: 'Shoulder Press', category: 'Upper Body' }
  ];

  const formTips = {
    squat: [
      'Keep your chest up and core engaged',
      'Knees should track over toes',
      'Hip crease below parallel',
      'Drive through heels to stand'
    ],
    pushup: [
      'Body in straight line from head to heels',
      'Hands shoulder-width apart',
      'Lower chest to floor',
      'Full extension at top'
    ],
    plank: [
      'Straight line from head to heels',
      'Engage core throughout',
      'Neutral spine position',
      'Breathe normally'
    ]
  };

  useEffect(() => {
    return () => {
      // Cleanup stream on unmount
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: 1280, 
          height: 720,
          facingMode: 'user'
        } 
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsStreaming(true);
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      alert('Unable to access camera. Please check permissions.');
    }
  };

  const stopCamera = () => {
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

  const toggleAnalysis = () => {
    if (isAnalyzing) {
      setIsAnalyzing(false);
      setFeedback(null);
    } else {
      setIsAnalyzing(true);
      startAnalysis();
    }
  };

  const analyzeSnapshot = async () => {
    if (!videoRef.current || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const video = videoRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    return new Promise((resolve) => {
      canvas.toBlob(async (blob) => {
        try {
          if (!blob) return resolve();
          const res = await poseAPI.analyzePose(blob, selectedExercise);
          const data = res.data || {};
          const score = typeof data.formScore === 'number' ? data.formScore : 0;
          const type = score >= 0.8 ? 'good' : score >= 0.7 ? 'warning' : 'error';
          setFeedback({ type, message: data.feedback || 'Analysis complete' });
          if (typeof data.repCount === 'number') {
            setRepCount(data.repCount);
          }
        } catch (e) {
          // Fallback: no-op on errors
        } finally {
          resolve();
        }
      }, 'image/jpeg', 0.9);
    });
  };

  const startAnalysis = () => {
    const analysisInterval = setInterval(async () => {
      if (!isAnalyzing) {
        clearInterval(analysisInterval);
        return;
      }
      await analyzeSnapshot();
    }, 2000);
  };

  const resetSession = () => {
    setRepCount(0);
    setFeedback(null);
    setIsAnalyzing(false);
  };

  const getFeedbackIcon = () => {
    if (!feedback) return null;
    
    switch (feedback.type) {
      case 'good':
        return <CheckCircle className="h-6 w-6 text-green-500" />;
      case 'warning':
        return <AlertCircle className="h-6 w-6 text-yellow-500" />;
      case 'error':
        return <XCircle className="h-6 w-6 text-red-500" />;
      default:
        return null;
    }
  };

  const getFeedbackColor = () => {
    if (!feedback) return 'bg-gray-100';
    
    switch (feedback.type) {
      case 'good':
        return 'bg-green-50 border-green-200';
      case 'warning':
        return 'bg-yellow-50 border-yellow-200';
      case 'error':
        return 'bg-red-50 border-red-200';
      default:
        return 'bg-gray-100';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Form Analysis</h1>
        <p className="mt-1 text-gray-600">Get real-time feedback on your exercise form</p>
      </div>

      {/* Exercise Selection */}
      <div className="bg-white rounded-xl p-6 shadow-sm">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Select Exercise</h2>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          {exercises.map(exercise => (
            <button
              key={exercise.id}
              onClick={() => setSelectedExercise(exercise.id)}
              className={`px-4 py-3 rounded-lg font-medium transition-colors ${
                selectedExercise === exercise.id
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              {exercise.name}
            </button>
          ))}
        </div>
      </div>

      {/* Camera View */}
      <div className="grid lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <div className="bg-white rounded-xl shadow-sm overflow-hidden">
            <div className="relative aspect-video bg-gray-900">
              {!isStreaming ? (
                <div className="absolute inset-0 flex flex-col items-center justify-center text-white">
                  <Camera className="h-16 w-16 mb-4 text-gray-400" />
                  <p className="text-lg mb-4">Camera is off</p>
                  <button
                    onClick={startCamera}
                    className="btn-primary"
                  >
                    Start Camera
                  </button>
                </div>
              ) : (
                <>
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="w-full h-full object-cover"
                  />
                  <canvas
                    ref={canvasRef}
                    className="absolute inset-0 w-full h-full"
                  />
                  
                  {/* Overlay UI */}
                  <div className="absolute top-4 left-4 right-4 flex justify-between">
                    <div className="bg-black/50 backdrop-blur rounded-lg px-4 py-2">
                      <p className="text-white text-sm font-medium">
                        {exercises.find(e => e.id === selectedExercise)?.name}
                      </p>
                    </div>
                    <div className="bg-black/50 backdrop-blur rounded-lg px-4 py-2">
                      <p className="text-white text-sm font-medium">
                        Reps: {repCount}
                      </p>
                    </div>
                  </div>

                  {/* Feedback Overlay */}
                  {feedback && isAnalyzing && (
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="absolute bottom-4 left-4 right-4"
                    >
                      <div className={`flex items-center space-x-3 p-4 rounded-lg border ${getFeedbackColor()}`}>
                        {getFeedbackIcon()}
                        <p className="font-medium">{feedback.message}</p>
                      </div>
                    </motion.div>
                  )}
                </>
              )}
            </div>

            {/* Controls */}
            {isStreaming && (
              <div className="p-4 bg-gray-50 flex items-center justify-center space-x-4">
                <button
                  onClick={toggleAnalysis}
                  className={`flex items-center space-x-2 px-6 py-3 rounded-lg font-medium transition-colors ${
                    isAnalyzing 
                      ? 'bg-red-600 text-white hover:bg-red-700'
                      : 'bg-primary-600 text-white hover:bg-primary-700'
                  }`}
                >
                  {isAnalyzing ? (
                    <>
                      <Pause className="h-5 w-5" />
                      <span>Stop Analysis</span>
                    </>
                  ) : (
                    <>
                      <Play className="h-5 w-5" />
                      <span>Start Analysis</span>
                    </>
                  )}
                </button>
                <button
                  onClick={resetSession}
                  className="flex items-center space-x-2 px-6 py-3 bg-gray-200 text-gray-700 rounded-lg font-medium hover:bg-gray-300 transition-colors"
                >
                  <RotateCcw className="h-5 w-5" />
                  <span>Reset</span>
                </button>
                <button
                  onClick={stopCamera}
                  className="flex items-center space-x-2 px-6 py-3 bg-gray-200 text-gray-700 rounded-lg font-medium hover:bg-gray-300 transition-colors"
                >
                  <CameraOff className="h-5 w-5" />
                  <span>Stop Camera</span>
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Tips and Stats */}
        <div className="space-y-6">
          {/* Form Tips */}
          <div className="bg-white rounded-xl p-6 shadow-sm">
            <div className="flex items-center space-x-2 mb-4">
              <Info className="h-5 w-5 text-primary-600" />
              <h3 className="text-lg font-semibold text-gray-900">Form Tips</h3>
            </div>
            <ul className="space-y-2">
              {(formTips[selectedExercise] || formTips.squat).map((tip, index) => (
                <li key={index} className="flex items-start">
                  <span className="text-primary-600 mr-2">•</span>
                  <span className="text-sm text-gray-600">{tip}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Session Stats */}
          <div className="bg-white rounded-xl p-6 shadow-sm">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Session Stats</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Total Reps</span>
                <span className="text-2xl font-bold text-gray-900">{repCount}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Duration</span>
                <span className="font-medium text-gray-900">--:--</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Form Score</span>
                <span className="font-medium text-green-600">Good</span>
              </div>
            </div>
          </div>

          {/* Instructions */}
          <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
            <h4 className="font-medium text-blue-900 mb-2">How it works:</h4>
            <ol className="text-sm text-blue-700 space-y-1">
              <li>1. Select your exercise</li>
              <li>2. Position yourself in frame</li>
              <li>3. Click "Start Analysis"</li>
              <li>4. Perform your exercise</li>
              <li>5. Get real-time feedback</li>
            </ol>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PoseAnalysis;
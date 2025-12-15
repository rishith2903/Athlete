import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthContext';
import { ThemeProvider } from './contexts/ThemeContext';
import { SettingsProvider } from './contexts/SettingsContext';
import { ToastProvider } from './contexts/ToastContext';
import ProtectedRoute from './components/ProtectedRoute';
import ErrorBoundary from './components/ErrorBoundary';
import Layout from './components/Layout';
import WorkoutGenerate from './pages/WorkoutGenerate';
import WorkoutDetail from './pages/WorkoutDetail';

// Auth Pages
import Login from './pages/auth/Login';
import Signup from './pages/auth/Signup';

// Main Pages
import Landing from './pages/Landing';
import Dashboard from './pages/Dashboard';
import Workouts from './pages/Workouts';
import Chatbot from './pages/Chatbot';
import PoseAnalysis from './pages/PoseAnalysis';
import Nutrition from './pages/Nutrition';
import ExerciseLibrary from './pages/ExerciseLibrary';
import WorkoutLogger from './pages/WorkoutLogger';
import Templates from './pages/Templates';
import Progress from './pages/Progress';
import Calculators from './pages/Calculators';
import BodyMeasurements from './pages/BodyMeasurements';
import Achievements from './pages/Achievements';
import Settings from './pages/Settings';
import Analytics from './pages/Analytics';

// Placeholder components
const Goals = () => <div className="text-center py-8"><h2 className="text-2xl font-bold">Goals Page</h2><p className="mt-2 text-gray-600">Coming soon...</p></div>;

function App() {
  return (
    <ThemeProvider>
      <SettingsProvider>
        <ToastProvider>
          <Router>
            <AuthProvider>
              <Routes>
                {/* Public Routes */}
                <Route path="/" element={<Landing />} />
                <Route path="/login" element={<Login />} />
                <Route path="/signup" element={<Signup />} />

                {/* Protected Routes */}
                <Route
                  path="/*"
                  element={
                    <ProtectedRoute>
                      <ErrorBoundary>
                        <Layout />
                      </ErrorBoundary>
                    </ProtectedRoute>
                  }
                >
                  <Route path="dashboard" element={<Dashboard />} />
                  <Route path="workouts" element={<Workouts />} />
                  <Route path="workouts/generate" element={<WorkoutGenerate />} />
                  <Route path="workouts/log" element={<WorkoutLogger />} />
                  <Route path="workouts/history" element={<Workouts />} />
                  <Route path="workouts/:id" element={<WorkoutDetail />} />
                  <Route path="exercises" element={<ExerciseLibrary />} />
                  <Route path="templates" element={<Templates />} />
                  <Route path="nutrition" element={<Nutrition />} />
                  <Route path="progress" element={<Progress />} />
                  <Route path="chatbot" element={<Chatbot />} />
                  <Route path="pose-analysis" element={<PoseAnalysis />} />
                  <Route path="calculators" element={<Calculators />} />
                  <Route path="measurements" element={<BodyMeasurements />} />
                  <Route path="achievements" element={<Achievements />} />
                  <Route path="settings" element={<Settings />} />
                  <Route path="analytics" element={<Analytics />} />
                  <Route path="goals" element={<Goals />} />

                  {/* Default redirect */}
                  <Route path="*" element={<Navigate to="/dashboard" replace />} />
                </Route>
              </Routes>
            </AuthProvider>
          </Router>
        </ToastProvider>
      </SettingsProvider>
    </ThemeProvider>
  );
}

export default App

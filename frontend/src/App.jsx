import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';
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

// Placeholder components for pages we haven't created yet
const Progress = () => <div className="text-center py-8"><h2 className="text-2xl font-bold">Progress Page</h2><p className="mt-2 text-gray-600">Coming soon...</p></div>;
const Goals = () => <div className="text-center py-8"><h2 className="text-2xl font-bold">Goals Page</h2><p className="mt-2 text-gray-600">Coming soon...</p></div>;
const Schedule = () => <div className="text-center py-8"><h2 className="text-2xl font-bold">Schedule Page</h2><p className="mt-2 text-gray-600">Coming soon...</p></div>;
const Profile = () => <div className="text-center py-8"><h2 className="text-2xl font-bold">Profile Page</h2><p className="mt-2 text-gray-600">Coming soon...</p></div>;

function App() {
  return (
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
                <Layout />
              </ProtectedRoute>
            }
          >
            <Route path="dashboard" element={<Dashboard />} />
            <Route path="workouts" element={<Workouts />} />
            <Route path="workouts/generate" element={<WorkoutGenerate />} />
            <Route path="workouts/:id" element={<WorkoutDetail />} />
            <Route path="nutrition" element={<Nutrition />} />
            <Route path="progress" element={<Progress />} />
            <Route path="chatbot" element={<Chatbot />} />
            <Route path="pose-analysis" element={<PoseAnalysis />} />
            <Route path="goals" element={<Goals />} />
            <Route path="schedule" element={<Schedule />} />
            <Route path="profile" element={<Profile />} />
            
            {/* Default redirect */}
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Route>
        </Routes>
      </AuthProvider>
    </Router>
  );
}

export default App

import axios from 'axios';

// Point to Spring Boot backend. The backend will call Python AI services.
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080/api';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor to handle errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Unauthorized - clear token and redirect to login
      localStorage.removeItem('authToken');
      localStorage.removeItem('user');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Helpers
const deriveUsername = (name, email) => {
  if (name) {
    return name.trim().toLowerCase().replace(/[^a-z0-9]+/g, '.').replace(/^\.|\.$/g, '').slice(0, 24) || (email ? email.split('@')[0] : undefined);
  }
  return email ? email.split('@')[0] : undefined;
};

// Authentication endpoints (Spring: /api/auth/*)
export const authAPI = {
  login: (credentials) => {
    const payload = {
      usernameOrEmail: credentials?.usernameOrEmail ?? credentials?.email ?? credentials?.username,
      password: credentials?.password,
    };
    return api.post('/auth/login', payload);
  },
  signup: (userData) => {
    const username = userData?.username || deriveUsername(userData?.name, userData?.email);
    const names = (userData?.name || '').trim().split(' ');
    const firstName = userData?.firstName || names[0] || username || '';
    const lastName = userData?.lastName || names.slice(1).join(' ') || '';
    const payload = {
      username,
      email: userData?.email,
      password: userData?.password,
      firstName,
      lastName,
      phoneNumber: userData?.phone,
      dateOfBirth: userData?.dateOfBirth || null,
      fitnessGoal: userData?.fitnessGoal || null,
      // Optional fields accepted by backend; sending null or defaults
      height: userData?.height ?? null,
      weight: userData?.weight ?? null,
      gender: userData?.gender ?? null,
      activityLevel: userData?.activityLevel ?? null,
      targetWeight: userData?.targetWeight ?? null,
      targetDate: userData?.targetDate ?? null,
      medicalConditions: userData?.medicalConditions ?? [],
      allergies: userData?.allergies ?? [],
      dietaryRestrictions: userData?.dietaryRestrictions ?? [],
      preferredExercises: userData?.preferredExercises ?? [],
      equipmentAvailable: userData?.equipmentAvailable ?? [],
      workoutDuration: userData?.workoutDuration ?? null,
      workoutsPerWeek: userData?.workoutsPerWeek ?? null,
    };
    return api.post('/auth/register', payload);
  },
  logout: () => api.post('/auth/logout'),
};

// Workout endpoints (Spring: /api/workout)
export const workoutAPI = {
  getWorkouts: () => api.get('/workout'),
  getWorkout: (id) => api.get(`/workout/${id}`),
  createWorkout: (data) => api.post('/workout', data),
  updateWorkout: (id, data) => api.put(`/workout/${id}`, data),
  deleteWorkout: (id) => api.delete(`/workout/${id}`),
  generateAIWorkout: (preferences) => api.post('/workout/ai-generate', preferences),
  startWorkout: (id) => api.post(`/workout/${id}/start`),
  completeWorkout: (id, completionData) => api.post(`/workout/${id}/complete`, completionData || {}),
  getStatistics: (days = 30) => api.get(`/workout/statistics?days=${days}`),
};

// Nutrition endpoints (Spring: /api/nutrition)
export const nutritionAPI = {
  generateAIDietPlan: (preferences) => api.post('/nutrition/ai-plan', preferences),
};

// Chatbot endpoints (Spring: /api/chatbot)
export const chatbotAPI = {
  sendMessage: ({ message, context, sessionId }) => api.post('/chatbot', { message, context, sessionId }),
  getSuggestions: () => api.get('/chatbot/suggestions'),
};

// Pose analysis endpoints (Spring: /api/pose)
export const poseAPI = {
  analyzePose: async (file, exerciseType) => {
    const form = new FormData();
    form.append('file', file);
    form.append('exerciseType', exerciseType);
    return api.post('/pose/check', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
};

// Progress tracking endpoints (Spring likely: /api/progress)
export const progressAPI = {
  getProgress: () => api.get('/progress'),
  recordWeight: (data) => api.post('/progress/weight', data),
  recordMeasurements: (data) => api.post('/progress/measurements', data),
  getProgressCharts: () => api.get('/progress/charts'),
  getGoals: () => api.get('/progress/goals'),
  setGoal: (data) => api.post('/progress/goals', data),
};

export default api;

import React, { createContext, useContext, useState, useEffect } from 'react';
import { authAPI } from '../services/api';

const AuthContext = createContext({});

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Initialize from localStorage (no profile endpoint on backend yet)
    const token = localStorage.getItem('authToken');
    const savedUser = localStorage.getItem('user');

    if (token && savedUser) {
      setUser(JSON.parse(savedUser));
    }
    setLoading(false);
  }, []);

  const toUserObject = (jwtResponse) => {
    if (!jwtResponse) return null;
    const {
      id, username, email, firstName, lastName, roles,
      subscriptionPlan, lastLoginAt
    } = jwtResponse;
    return {
      id, username, email, firstName, lastName, roles,
      subscriptionPlan, lastLoginAt,
      name: [firstName, lastName].filter(Boolean).join(' ').trim() || username || email,
    };
  };

  const login = async (emailOrUsername, password) => {
    try {
      setError(null);
      setLoading(true);
      const response = await authAPI.login({ usernameOrEmail: emailOrUsername, password });
      const data = response.data;

      const accessToken = data.accessToken || data.token;
      const userObj = toUserObject(data);

      if (accessToken) localStorage.setItem('authToken', accessToken);
      if (userObj) localStorage.setItem('user', JSON.stringify(userObj));
      setUser(userObj);

      return { success: true };
    } catch (err) {
      const errorMessage = err.response?.data?.message || 'Login failed';
      setError(errorMessage);
      return { success: false, error: errorMessage };
    } finally {
      setLoading(false);
    }
  };

  const signup = async (userData) => {
    try {
      setError(null);
      setLoading(true);
      const response = await authAPI.signup(userData);
      const data = response.data;

      const accessToken = data.accessToken || data.token;
      const userObj = toUserObject(data);

      if (accessToken) localStorage.setItem('authToken', accessToken);
      if (userObj) localStorage.setItem('user', JSON.stringify(userObj));
      setUser(userObj);

      return { success: true };
    } catch (err) {
      const errorMessage = err.response?.data?.message || 'Signup failed';
      setError(errorMessage);
      return { success: false, error: errorMessage };
    } finally {
      setLoading(false);
    }
  };

  const logout = async () => {
    try {
      await authAPI.logout();
    } catch (err) {
      // Ignore logout errors
    } finally {
      localStorage.removeItem('authToken');
      localStorage.removeItem('user');
      setUser(null);
      setError(null);
    }
  };

  // Local-only profile update until backend endpoint exists
  const updateProfile = async (partial) => {
    try {
      const current = JSON.parse(localStorage.getItem('user') || '{}');
      const updated = { ...current, ...partial };
      localStorage.setItem('user', JSON.stringify(updated));
      setUser(updated);
      return { success: true };
    } catch (err) {
      const errorMessage = 'Profile update failed';
      setError(errorMessage);
      return { success: false, error: errorMessage };
    }
  };

  const value = {
    user,
    loading,
    error,
    login,
    signup,
    logout,
    updateProfile,
    isAuthenticated: !!user,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

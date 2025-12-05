import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Eye, EyeOff, Mail, Lock, Loader2, Play } from 'lucide-react';
import { useAuth } from '../../contexts/AuthContext';
import { cn } from '../../lib/utils';

// Demo credentials
const DEMO_CREDENTIALS = {
  email: 'demo@athlete.com',
  password: 'Demo@123'
};

const Login = () => {
  const navigate = useNavigate();
  const { login, loading } = useAuth();
  const [showPassword, setShowPassword] = useState(false);
  const [isDemoLoading, setIsDemoLoading] = useState(false);
  const [formData, setFormData] = useState({
    email: '',
    password: '',
  });
  const [errors, setErrors] = useState({});

  const validateForm = () => {
    const newErrors = {};

    if (!formData.email) {
      newErrors.email = 'Email is required';
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = 'Email is invalid';
    }

    if (!formData.password) {
      newErrors.password = 'Password is required';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!validateForm()) return;

    const result = await login(formData.email, formData.password);

    if (result.success) {
      navigate('/dashboard');
    } else {
      setErrors({ general: result.error });
    }
  };

  const handleDemoLogin = async () => {
    setIsDemoLoading(true);
    setErrors({});

    // Fill in demo credentials with animation effect
    setFormData({
      email: DEMO_CREDENTIALS.email,
      password: DEMO_CREDENTIALS.password,
    });

    // Small delay to show the filled form
    await new Promise(resolve => setTimeout(resolve, 500));

    const result = await login(DEMO_CREDENTIALS.email, DEMO_CREDENTIALS.password);

    setIsDemoLoading(false);

    if (result.success) {
      navigate('/dashboard');
    } else {
      setErrors({ general: result.error });
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    // Clear error for this field
    if (errors[name]) {
      setErrors(prev => ({ ...prev, [name]: '' }));
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100 px-4 sm:px-6 lg:px-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="max-w-md w-full space-y-8"
      >
        <div className="bg-white rounded-2xl shadow-xl p-8">
          <div className="text-center">
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="mx-auto h-16 w-16 bg-blue-600 rounded-full flex items-center justify-center"
            >
              <span className="text-white text-2xl font-bold">FIT</span>
            </motion.div>
            <h2 className="mt-6 text-3xl font-bold text-gray-900">
              Welcome Back
            </h2>
            <p className="mt-2 text-sm text-gray-600">
              Sign in to continue your fitness journey
            </p>
          </div>

          <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
            {errors.general && (
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg"
              >
                {errors.general}
              </motion.div>
            )}

            <div className="space-y-4">
              <div>
                <label htmlFor="email" className="label">
                  Email Address
                </label>
                <div className="relative flex items-center">
                  <span className="absolute left-3 flex items-center justify-center pointer-events-none">
                    <Mail className="text-gray-400 h-5 w-5" />
                  </span>
                  <input
                    id="email"
                    name="email"
                    type="email"
                    autoComplete="email"
                    required
                    value={formData.email}
                    onChange={handleChange}
                    className={cn(
                      "input pl-11",
                      errors.email && "border-red-500 focus:ring-red-500"
                    )}
                    placeholder="Enter your email"
                  />
                </div>
                {errors.email && (
                  <p className="mt-1 text-sm text-red-600">{errors.email}</p>
                )}
              </div>

              <div>
                <label htmlFor="password" className="label">
                  Password
                </label>
                <div className="relative flex items-center">
                  <span className="absolute left-3 flex items-center justify-center pointer-events-none">
                    <Lock className="text-gray-400 h-5 w-5" />
                  </span>
                  <input
                    id="password"
                    name="password"
                    type={showPassword ? "text" : "password"}
                    autoComplete="current-password"
                    required
                    value={formData.password}
                    onChange={handleChange}
                    className={cn(
                      "input pl-11 pr-11",
                      errors.password && "border-red-500 focus:ring-red-500"
                    )}
                    placeholder="Enter your password"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 flex items-center justify-center text-gray-400 hover:text-gray-600 transition-colors"
                  >
                    {showPassword ? (
                      <EyeOff className="h-5 w-5" />
                    ) : (
                      <Eye className="h-5 w-5" />
                    )}
                  </button>
                </div>
                {errors.password && (
                  <p className="mt-1 text-sm text-red-600">{errors.password}</p>
                )}
              </div>
            </div>

            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <input
                  id="remember-me"
                  name="remember-me"
                  type="checkbox"
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <label htmlFor="remember-me" className="ml-2 block text-sm text-gray-700">
                  Remember me
                </label>
              </div>

              <div className="text-sm">
                <Link to="/forgot-password" className="font-medium text-blue-600 hover:text-blue-500">
                  Forgot password?
                </Link>
              </div>
            </div>

            <div className="space-y-3">
              <button
                type="submit"
                disabled={loading || isDemoLoading}
                className={cn(
                  "w-full flex justify-center items-center btn-primary",
                  (loading || isDemoLoading) && "opacity-50 cursor-not-allowed"
                )}
              >
                {loading ? (
                  <>
                    <Loader2 className="animate-spin h-5 w-5 mr-2" />
                    Signing in...
                  </>
                ) : (
                  'Sign In'
                )}
              </button>

              {/* Demo Login Button */}
              <button
                type="button"
                onClick={handleDemoLogin}
                disabled={loading || isDemoLoading}
                className={cn(
                  "w-full flex justify-center items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all duration-200",
                  "bg-gradient-to-r from-emerald-500 to-teal-500 text-white",
                  "hover:from-emerald-600 hover:to-teal-600 shadow-md hover:shadow-lg",
                  (loading || isDemoLoading) && "opacity-50 cursor-not-allowed"
                )}
              >
                {isDemoLoading ? (
                  <>
                    <Loader2 className="animate-spin h-5 w-5" />
                    Starting Demo...
                  </>
                ) : (
                  <>
                    <Play className="h-5 w-5" />
                    Try Demo Account
                  </>
                )}
              </button>

              <p className="text-center text-xs text-gray-500">
                Use demo account to explore features without signing up
              </p>
            </div>

            <div className="text-center">
              <span className="text-sm text-gray-600">
                Don't have an account?{' '}
                <Link to="/signup" className="font-medium text-blue-600 hover:text-blue-500">
                  Sign up
                </Link>
              </span>
            </div>
          </form>
        </div>
      </motion.div>
    </div>
  );
};

export default Login;
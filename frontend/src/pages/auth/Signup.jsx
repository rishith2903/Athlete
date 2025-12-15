import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Eye, EyeOff, Mail, Lock, User, Loader2, Phone, Calendar, Ruler, Scale, Sun, Moon } from 'lucide-react';
import { useAuth } from '../../contexts/AuthContext';
import { useTheme } from '../../contexts/ThemeContext';
import { cn } from '../../lib/utils';

const Signup = () => {
  const navigate = useNavigate();
  const { signup, loading } = useAuth();
  const { isDark, toggleTheme } = useTheme();
  const [showPassword, setShowPassword] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    confirmPassword: '',
    phone: '',
    dateOfBirth: '',
    weight: '',
    height: '',
  });
  const [errors, setErrors] = useState({});

  const validateForm = () => {
    const newErrors = {};

    if (!formData.name) {
      newErrors.name = 'Name is required';
    }

    if (!formData.email) {
      newErrors.email = 'Email is required';
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = 'Email is invalid';
    }

    if (!formData.password) {
      newErrors.password = 'Password is required';
    } else if (formData.password.length < 8) {
      newErrors.password = 'Password must be at least 8 characters';
    }

    if (!formData.confirmPassword) {
      newErrors.confirmPassword = 'Please confirm your password';
    } else if (formData.password !== formData.confirmPassword) {
      newErrors.confirmPassword = 'Passwords do not match';
    }

    if (!formData.dateOfBirth) {
      newErrors.dateOfBirth = 'Date of birth is required';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!validateForm()) return;

    const { confirmPassword, ...userData } = formData;
    const result = await signup(userData);

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
    <div className={cn(
      "min-h-screen flex items-center justify-center px-4 sm:px-6 lg:px-8 py-12 relative transition-colors duration-300",
      isDark ? "bg-gray-900" : "bg-gradient-to-br from-blue-50 to-indigo-100"
    )}>
      {/* Theme Toggle */}
      <button
        onClick={toggleTheme}
        className="absolute top-4 right-4 p-3 rounded-full bg-white shadow-lg hover:shadow-xl transition-all duration-200 text-gray-600 hover:text-gray-800"
        aria-label="Toggle theme"
      >
        {isDark ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
      </button>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="max-w-md w-full space-y-8"
      >
        <div className="bg-white rounded-2xl shadow-xl p-8 text-gray-900" style={{ backgroundColor: 'white' }}>
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
              Create Account
            </h2>
            <p className="mt-2 text-sm text-gray-600">
              Start your fitness journey today
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
                <label htmlFor="name" className="label">
                  Full Name <span className="text-red-500">*</span>
                </label>
                <div className="flex">
                  <div className={cn(
                    "flex items-center justify-center px-3 border border-r-0 rounded-l-lg",
                    isDark ? "bg-gray-800 border-gray-600" : "bg-white border-gray-200"
                  )}>
                    <User className="text-gray-500 h-4 w-4" />
                  </div>
                  <input
                    id="name"
                    name="name"
                    type="text"
                    autoComplete="name"
                    required
                    value={formData.name}
                    onChange={handleChange}
                    className={cn(
                      "input rounded-l-none flex-1",
                      errors.name && "border-red-500 focus:ring-red-500"
                    )}
                    placeholder="Enter your full name"
                  />
                </div>
                {errors.name && (
                  <p className="mt-1 text-sm text-red-600">{errors.name}</p>
                )}
              </div>

              <div>
                <label htmlFor="email" className="label">
                  Email Address <span className="text-red-500">*</span>
                </label>
                <div className="flex">
                  <div className={cn(
                    "flex items-center justify-center px-3 border border-r-0 rounded-l-lg",
                    isDark ? "bg-gray-800 border-gray-600" : "bg-white border-gray-200"
                  )}>
                    <Mail className="text-gray-500 h-4 w-4" />
                  </div>
                  <input
                    id="email"
                    name="email"
                    type="email"
                    autoComplete="email"
                    required
                    value={formData.email}
                    onChange={handleChange}
                    className={cn(
                      "input rounded-l-none flex-1",
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
                <label htmlFor="phone" className="label">
                  Phone Number
                </label>
                <div className="flex">
                  <div className={cn(
                    "flex items-center justify-center px-3 border border-r-0 rounded-l-lg",
                    isDark ? "bg-gray-800 border-gray-600" : "bg-white border-gray-200"
                  )}>
                    <Phone className="text-gray-500 h-4 w-4" />
                  </div>
                  <input
                    id="phone"
                    name="phone"
                    type="tel"
                    autoComplete="tel"
                    value={formData.phone}
                    onChange={handleChange}
                    className="input rounded-l-none flex-1"
                    placeholder="Enter your phone number"
                  />
                </div>
              </div>

              <div>
                <label htmlFor="dateOfBirth" className="label">
                  Date of Birth <span className="text-red-500">*</span>
                </label>
                <div className="flex">
                  <div className={cn(
                    "flex items-center justify-center px-3 border border-r-0 rounded-l-lg",
                    isDark ? "bg-gray-800 border-gray-600" : "bg-white border-gray-200"
                  )}>
                    <Calendar className="text-gray-500 h-4 w-4" />
                  </div>
                  <input
                    id="dateOfBirth"
                    name="dateOfBirth"
                    type="date"
                    required
                    value={formData.dateOfBirth}
                    onChange={handleChange}
                    className={cn(
                      "input rounded-l-none flex-1",
                      errors.dateOfBirth && "border-red-500 focus:ring-red-500"
                    )}
                  />
                </div>
                {errors.dateOfBirth && (
                  <p className="mt-1 text-sm text-red-600">{errors.dateOfBirth}</p>
                )}
              </div>

              {/* Weight and Height - Optional */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label htmlFor="weight" className="label">
                    Weight (kg)
                  </label>
                  <div className="flex">
                    <div className={cn(
                      "flex items-center justify-center px-3 border border-r-0 rounded-l-lg",
                      isDark ? "bg-gray-800 border-gray-600" : "bg-white border-gray-200"
                    )}>
                      <Scale className="text-gray-500 h-4 w-4" />
                    </div>
                    <input
                      id="weight"
                      name="weight"
                      type="number"
                      step="0.1"
                      min="0"
                      value={formData.weight}
                      onChange={handleChange}
                      className="input rounded-l-none flex-1"
                      placeholder="e.g. 70"
                    />
                  </div>
                </div>

                <div>
                  <label htmlFor="height" className="label">
                    Height (cm)
                  </label>
                  <div className="flex">
                    <div className={cn(
                      "flex items-center justify-center px-3 border border-r-0 rounded-l-lg",
                      isDark ? "bg-gray-800 border-gray-600" : "bg-white border-gray-200"
                    )}>
                      <Ruler className="text-gray-500 h-4 w-4" />
                    </div>
                    <input
                      id="height"
                      name="height"
                      type="number"
                      step="0.1"
                      min="0"
                      value={formData.height}
                      onChange={handleChange}
                      className="input rounded-l-none flex-1"
                      placeholder="e.g. 175"
                    />
                  </div>
                </div>
              </div>

              <div>
                <label htmlFor="password" className="label">
                  Password <span className="text-red-500">*</span>
                </label>
                <div className="flex">
                  <div className={cn(
                    "flex items-center justify-center px-3 border border-r-0 rounded-l-lg",
                    isDark ? "bg-gray-800 border-gray-600" : "bg-white border-gray-200"
                  )}>
                    <Lock className="text-gray-500 h-4 w-4" />
                  </div>
                  <div className="relative flex-1">
                    <input
                      id="password"
                      name="password"
                      type={showPassword ? "text" : "password"}
                      autoComplete="new-password"
                      required
                      value={formData.password}
                      onChange={handleChange}
                      className={cn(
                        "input rounded-l-none w-full pr-12",
                        errors.password && "border-red-500 focus:ring-red-500"
                      )}
                      placeholder="Create a password"
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
                    >
                      {showPassword ? (
                        <EyeOff className="h-5 w-5" />
                      ) : (
                        <Eye className="h-5 w-5" />
                      )}
                    </button>
                  </div>
                </div>
                {errors.password && (
                  <p className="mt-1 text-sm text-red-600">{errors.password}</p>
                )}
              </div>

              <div>
                <label htmlFor="confirmPassword" className="label">
                  Confirm Password <span className="text-red-500">*</span>
                </label>
                <div className="flex">
                  <div className={cn(
                    "flex items-center justify-center px-3 border border-r-0 rounded-l-lg",
                    isDark ? "bg-gray-800 border-gray-600" : "bg-white border-gray-200"
                  )}>
                    <Lock className="text-gray-500 h-4 w-4" />
                  </div>
                  <input
                    id="confirmPassword"
                    name="confirmPassword"
                    type={showPassword ? "text" : "password"}
                    autoComplete="new-password"
                    required
                    value={formData.confirmPassword}
                    onChange={handleChange}
                    className={cn(
                      "input rounded-l-none flex-1",
                      errors.confirmPassword && "border-red-500 focus:ring-red-500"
                    )}
                    placeholder="Confirm your password"
                  />
                </div>
                {errors.confirmPassword && (
                  <p className="mt-1 text-sm text-red-600">{errors.confirmPassword}</p>
                )}
              </div>
            </div>

            <div className="flex items-center">
              <input
                id="terms"
                name="terms"
                type="checkbox"
                required
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
              <label htmlFor="terms" className="ml-2 block text-sm text-gray-700">
                I agree to the{' '}
                <Link to="/terms" className="text-blue-600 hover:text-blue-500">
                  Terms and Conditions
                </Link>
              </label>
            </div>

            <div>
              <button
                type="submit"
                disabled={loading}
                className={cn(
                  "w-full flex justify-center items-center btn-primary",
                  loading && "opacity-50 cursor-not-allowed"
                )}
              >
                {loading ? (
                  <>
                    <Loader2 className="animate-spin h-5 w-5 mr-2" />
                    Creating account...
                  </>
                ) : (
                  'Sign Up'
                )}
              </button>
            </div>

            <div className="text-center">
              <span className="text-sm text-gray-600">
                Already have an account?{' '}
                <Link to="/login" className="font-medium text-blue-600 hover:text-blue-500">
                  Sign in
                </Link>
              </span>
            </div>
          </form>
        </div>
      </motion.div>
    </div>
  );
};

export default Signup;
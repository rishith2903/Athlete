import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { motion, useScroll, useTransform } from 'framer-motion';
import {
  ArrowRight,
  CheckCircle,
  Dumbbell,
  Utensils,
  Brain,
  Camera,
  TrendingUp,
  Users,
  Star,
  Play,
  Zap,
  Target,
  MessageSquare,
  Shield,
  ChevronRight,
  Activity,
  Award,
  Sparkles,
  Menu,
  X
} from 'lucide-react';

const Landing = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [activeTestimonial, setActiveTestimonial] = useState(0);
  const { scrollYProgress } = useScroll();
  const opacity = useTransform(scrollYProgress, [0, 0.2], [1, 0]);

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveTestimonial((prev) => (prev + 1) % testimonials.length);
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const features = [
    {
      icon: Brain,
      title: 'AI Workout Generation',
      description: 'Deep reinforcement learning creates personalized routines based on your goals, equipment, and fitness level.',
      gradient: 'from-blue-500 to-cyan-500',
      delay: 0
    },
    {
      icon: Utensils,
      title: 'Smart Nutrition Plans',
      description: 'Intelligent meal recommendations tailored to your dietary preferences and fitness objectives.',
      gradient: 'from-green-500 to-emerald-500',
      delay: 0.1
    },
    {
      icon: Camera,
      title: 'Real-time Pose Analysis',
      description: 'Advanced computer vision provides instant feedback on form, technique, and injury prevention.',
      gradient: 'from-purple-500 to-pink-500',
      delay: 0.2
    },
    {
      icon: MessageSquare,
      title: 'AI Fitness Chatbot',
      description: 'Get intelligent answers to your fitness questions with our LangChain-powered assistant.',
      gradient: 'from-orange-500 to-red-500',
      delay: 0.3
    }
  ];

  const howItWorks = [
    {
      step: '01',
      title: 'Create Your Profile',
      description: 'Tell us about your fitness level, goals, and available equipment.',
      icon: Users
    },
    {
      step: '02',
      title: 'Get AI Recommendations',
      description: 'Receive personalized workout and nutrition plans powered by machine learning.',
      icon: Brain
    },
    {
      step: '03',
      title: 'Train with Guidance',
      description: 'Use real-time pose analysis to perfect your form and prevent injuries.',
      icon: Camera
    },
    {
      step: '04',
      title: 'Track Progress',
      description: 'Monitor your improvements with detailed analytics and insights.',
      icon: TrendingUp
    }
  ];

  const testimonials = [
    {
      name: 'Sarah Johnson',
      role: 'Fitness Enthusiast',
      avatar: 'SJ',
      content: 'AIthlete completely transformed my workout routine. The AI-generated plans are perfectly tailored to my goals, and the pose analysis helped me fix my squat form!',
      rating: 5
    },
    {
      name: 'Mike Chen',
      role: 'Personal Trainer',
      avatar: 'MC',
      content: 'As a PT, I recommend AIthlete to all my clients. The form analysis feature catches things even I might miss. It\'s like having a training partner available 24/7.',
      rating: 5
    },
    {
      name: 'Emma Williams',
      role: 'Nutritionist',
      avatar: 'EW',
      content: 'The nutrition planning is incredibly sophisticated. It considers macros, dietary restrictions, and actually tastes good. My clients love the meal suggestions!',
      rating: 5
    }
  ];

  const stats = [
    { value: '50K+', label: 'Active Users', icon: Users },
    { value: '1M+', label: 'Workouts Completed', icon: Dumbbell },
    { value: '94%', label: 'Pose Accuracy', icon: Target },
    { value: '4.9', label: 'App Rating', icon: Star }
  ];

  const technologies = [
    'Deep Reinforcement Learning',
    'MediaPipe Pose Estimation',
    'LangChain AI',
    'Computer Vision',
    'Transformer Models',
    'Real-time Analysis'
  ];

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 via-white to-slate-50 dark:from-slate-950 dark:via-slate-900 dark:to-slate-950 overflow-hidden">
      {/* Animated Background Elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-blue-500/10 rounded-full blur-3xl animate-pulse" />
        <div className="absolute top-1/2 -left-40 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }} />
        <div className="absolute bottom-20 right-1/4 w-72 h-72 bg-cyan-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }} />
      </div>

      {/* Navigation */}
      <nav className="fixed top-0 w-full bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl z-50 border-b border-gray-100 dark:border-slate-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <motion.div
              className="flex items-center"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
            >
              <div className="h-10 w-10 bg-gradient-to-br from-blue-600 to-blue-700 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/25">
                <Dumbbell className="h-5 w-5 text-white" />
              </div>
              <span className="ml-3 text-xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 dark:from-white dark:to-gray-300 bg-clip-text text-transparent">
                AIthlete
              </span>
            </motion.div>

            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center space-x-8">
              <a href="#features" className="text-gray-600 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 font-medium transition-colors">
                Features
              </a>
              <a href="#how-it-works" className="text-gray-600 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 font-medium transition-colors">
                How it Works
              </a>
              <a href="#testimonials" className="text-gray-600 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 font-medium transition-colors">
                Reviews
              </a>
            </div>

            <motion.div
              className="hidden md:flex items-center space-x-4"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
            >
              <Link
                to="/login"
                className="text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 font-medium transition-colors"
              >
                Login
              </Link>
              <Link
                to="/signup"
                className="inline-flex items-center bg-gradient-to-r from-blue-600 to-blue-700 text-white px-5 py-2.5 rounded-xl font-medium hover:from-blue-700 hover:to-blue-800 transition-all duration-300 shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40 hover:-translate-y-0.5"
              >
                Get Started
                <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </motion.div>

            {/* Mobile Menu Button */}
            <button
              className="md:hidden p-2"
              onClick={() => setIsMenuOpen(!isMenuOpen)}
            >
              {isMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        {isMenuOpen && (
          <motion.div
            className="md:hidden bg-white dark:bg-slate-900 border-t border-gray-100 dark:border-slate-800"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <div className="px-4 py-4 space-y-3">
              <a href="#features" className="block text-gray-600 dark:text-gray-300 hover:text-blue-600 font-medium">Features</a>
              <a href="#how-it-works" className="block text-gray-600 dark:text-gray-300 hover:text-blue-600 font-medium">How it Works</a>
              <a href="#testimonials" className="block text-gray-600 dark:text-gray-300 hover:text-blue-600 font-medium">Reviews</a>
              <hr className="border-gray-100 dark:border-slate-800" />
              <Link to="/login" className="block text-gray-600 dark:text-gray-300 hover:text-blue-600 font-medium">Login</Link>
              <Link to="/signup" className="block bg-blue-600 text-white px-4 py-2 rounded-lg text-center font-medium">Get Started</Link>
            </div>
          </motion.div>
        )}
      </nav>

      {/* Hero Section */}
      <section className="pt-32 pb-20 px-4 sm:px-6 lg:px-8 relative">
        <div className="max-w-7xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, ease: 'easeOut' }}
            >
              {/* Badge */}
              <motion.div
                className="inline-flex items-center bg-blue-50 dark:bg-blue-500/10 rounded-full px-4 py-1.5 mb-6"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.2 }}
              >
                <Sparkles className="h-4 w-4 text-blue-600 dark:text-blue-400 mr-2" />
                <span className="text-sm font-medium text-blue-700 dark:text-blue-300">Powered by Advanced AI</span>
              </motion.div>

              <h1 className="text-5xl lg:text-7xl font-bold leading-tight">
                <span className="bg-gradient-to-r from-gray-900 via-gray-800 to-gray-900 dark:from-white dark:via-gray-200 dark:to-white bg-clip-text text-transparent">
                  Transform Your
                </span>
                <br />
                <span className="bg-gradient-to-r from-blue-600 via-blue-500 to-cyan-500 bg-clip-text text-transparent">
                  Fitness Journey
                </span>
              </h1>

              <p className="mt-6 text-xl text-gray-600 dark:text-gray-400 leading-relaxed max-w-xl">
                Experience personalized workouts, smart nutrition plans, and real-time form analysis powered by cutting-edge machine learning and computer vision.
              </p>

              <div className="mt-10 flex flex-col sm:flex-row gap-4">
                <Link
                  to="/signup"
                  className="group inline-flex items-center justify-center bg-gradient-to-r from-blue-600 to-blue-700 text-white px-8 py-4 rounded-2xl font-semibold text-lg hover:from-blue-700 hover:to-blue-800 transition-all duration-300 shadow-xl shadow-blue-500/25 hover:shadow-blue-500/40 hover:-translate-y-1"
                >
                  Start Free Trial
                  <ArrowRight className="ml-2 h-5 w-5 group-hover:translate-x-1 transition-transform" />
                </Link>
                <button className="group inline-flex items-center justify-center bg-white dark:bg-slate-800 text-gray-900 dark:text-white px-8 py-4 rounded-2xl font-semibold text-lg border border-gray-200 dark:border-slate-700 hover:border-blue-300 dark:hover:border-blue-500 transition-all duration-300 hover:-translate-y-1 shadow-lg">
                  <Play className="mr-2 h-5 w-5 text-blue-600" />
                  Watch Demo
                </button>
              </div>

              <div className="mt-10 flex flex-wrap items-center gap-8">
                <div className="flex items-center">
                  <CheckCircle className="h-5 w-5 text-green-500 mr-2" />
                  <span className="text-gray-600 dark:text-gray-400">No credit card required</span>
                </div>
                <div className="flex items-center">
                  <CheckCircle className="h-5 w-5 text-green-500 mr-2" />
                  <span className="text-gray-600 dark:text-gray-400">14-day free trial</span>
                </div>
                <div className="flex items-center">
                  <CheckCircle className="h-5 w-5 text-green-500 mr-2" />
                  <span className="text-gray-600 dark:text-gray-400">Cancel anytime</span>
                </div>
              </div>
            </motion.div>

            {/* Hero Visual */}
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.8, delay: 0.3 }}
              className="relative"
            >
              <div className="relative">
                {/* Main Card */}
                <div className="relative bg-gradient-to-br from-slate-900 to-slate-800 rounded-3xl p-8 shadow-2xl overflow-hidden">
                  {/* Animated Grid Background */}
                  <div className="absolute inset-0 bg-[linear-gradient(rgba(59,130,246,0.1)_1px,transparent_1px),linear-gradient(90deg,rgba(59,130,246,0.1)_1px,transparent_1px)] bg-[size:40px_40px] opacity-30" />

                  <div className="relative z-10">
                    {/* Dashboard Preview Header */}
                    <div className="flex items-center justify-between mb-6">
                      <div className="flex items-center gap-3">
                        <div className="h-10 w-10 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl flex items-center justify-center">
                          <Dumbbell className="h-5 w-5 text-white" />
                        </div>
                        <div>
                          <div className="text-white font-semibold">Today's Workout</div>
                          <div className="text-gray-400 text-sm">Upper Body Strength</div>
                        </div>
                      </div>
                      <span className="text-xs font-medium text-green-400 bg-green-500/20 px-3 py-1 rounded-full">Active</span>
                    </div>

                    {/* Stats Row */}
                    <div className="grid grid-cols-3 gap-4 mb-6">
                      <div className="bg-white/5 backdrop-blur rounded-2xl p-4">
                        <div className="text-2xl font-bold text-white">450</div>
                        <div className="text-gray-400 text-sm">Calories</div>
                      </div>
                      <div className="bg-white/5 backdrop-blur rounded-2xl p-4">
                        <div className="text-2xl font-bold text-cyan-400">45</div>
                        <div className="text-gray-400 text-sm">Minutes</div>
                      </div>
                      <div className="bg-white/5 backdrop-blur rounded-2xl p-4">
                        <div className="text-2xl font-bold text-green-400">85%</div>
                        <div className="text-gray-400 text-sm">Complete</div>
                      </div>
                    </div>

                    {/* Progress Bar */}
                    <div className="space-y-4">
                      <div>
                        <div className="flex justify-between text-sm mb-2">
                          <span className="text-gray-300">Workout Progress</span>
                          <span className="text-white font-medium">85%</span>
                        </div>
                        <div className="h-3 bg-white/10 rounded-full overflow-hidden">
                          <motion.div
                            className="h-full bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full"
                            initial={{ width: 0 }}
                            animate={{ width: '85%' }}
                            transition={{ duration: 1.5, delay: 0.8, ease: 'easeOut' }}
                          />
                        </div>
                      </div>
                      <div>
                        <div className="flex justify-between text-sm mb-2">
                          <span className="text-gray-300">Form Score</span>
                          <span className="text-white font-medium">94%</span>
                        </div>
                        <div className="h-3 bg-white/10 rounded-full overflow-hidden">
                          <motion.div
                            className="h-full bg-gradient-to-r from-green-500 to-emerald-500 rounded-full"
                            initial={{ width: 0 }}
                            animate={{ width: '94%' }}
                            transition={{ duration: 1.5, delay: 1, ease: 'easeOut' }}
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Floating Elements */}
                <motion.div
                  className="absolute -top-6 -right-6 bg-white dark:bg-slate-800 rounded-2xl p-4 shadow-xl border border-gray-100 dark:border-slate-700"
                  animate={{ y: [0, -10, 0] }}
                  transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
                >
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 bg-green-100 dark:bg-green-500/20 rounded-xl flex items-center justify-center">
                      <Activity className="h-5 w-5 text-green-600 dark:text-green-400" />
                    </div>
                    <div>
                      <div className="text-sm font-medium text-gray-900 dark:text-white">Form Detected</div>
                      <div className="text-xs text-green-600 dark:text-green-400">Posture: Excellent</div>
                    </div>
                  </div>
                </motion.div>

                <motion.div
                  className="absolute -bottom-4 -left-8 bg-white dark:bg-slate-800 rounded-2xl p-4 shadow-xl border border-gray-100 dark:border-slate-700"
                  animate={{ y: [0, 10, 0] }}
                  transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
                >
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 bg-purple-100 dark:bg-purple-500/20 rounded-xl flex items-center justify-center">
                      <Brain className="h-5 w-5 text-purple-600 dark:text-purple-400" />
                    </div>
                    <div>
                      <div className="text-sm font-medium text-gray-900 dark:text-white">AI Suggestion</div>
                      <div className="text-xs text-gray-600 dark:text-gray-400">Increase weight by 5%</div>
                    </div>
                  </div>
                </motion.div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Tech Stack Marquee */}
      <section className="py-8 bg-gradient-to-r from-blue-600 to-blue-700 overflow-hidden">
        <div className="flex animate-marquee whitespace-nowrap">
          {[...technologies, ...technologies].map((tech, index) => (
            <span key={index} className="text-white/90 font-medium mx-8 flex items-center">
              <Zap className="h-4 w-4 mr-2 text-cyan-300" />
              {tech}
            </span>
          ))}
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-24 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <motion.div
            className="text-center max-w-3xl mx-auto mb-16"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
          >
            <span className="text-blue-600 dark:text-blue-400 font-semibold text-sm uppercase tracking-wider">Features</span>
            <h2 className="mt-4 text-4xl lg:text-5xl font-bold text-gray-900 dark:text-white">
              Everything You Need to
              <span className="bg-gradient-to-r from-blue-600 to-cyan-500 bg-clip-text text-transparent"> Succeed</span>
            </h2>
            <p className="mt-6 text-xl text-gray-600 dark:text-gray-400">
              Comprehensive AI-powered tools designed to help you achieve your fitness goals faster and smarter.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: feature.delay }}
                  viewport={{ once: true }}
                  className="group relative bg-white dark:bg-slate-800/50 rounded-3xl p-8 shadow-lg hover:shadow-2xl transition-all duration-500 border border-gray-100 dark:border-slate-700 hover:border-blue-200 dark:hover:border-blue-500/30 overflow-hidden"
                >
                  {/* Gradient Overlay on Hover */}
                  <div className={`absolute inset-0 bg-gradient-to-br ${feature.gradient} opacity-0 group-hover:opacity-5 transition-opacity duration-500`} />

                  <div className={`relative z-10 h-14 w-14 bg-gradient-to-br ${feature.gradient} rounded-2xl flex items-center justify-center mb-6 shadow-lg group-hover:scale-110 transition-transform duration-300`}>
                    <Icon className="h-7 w-7 text-white" />
                  </div>
                  <h3 className="relative z-10 text-xl font-bold text-gray-900 dark:text-white mb-3 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                    {feature.title}
                  </h3>
                  <p className="relative z-10 text-gray-600 dark:text-gray-400 leading-relaxed">
                    {feature.description}
                  </p>
                  <div className="relative z-10 mt-6 flex items-center text-blue-600 dark:text-blue-400 font-medium opacity-0 group-hover:opacity-100 transition-opacity">
                    Learn more
                    <ChevronRight className="h-4 w-4 ml-1 group-hover:translate-x-1 transition-transform" />
                  </div>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-20 bg-gradient-to-r from-blue-600 via-blue-700 to-blue-600 relative overflow-hidden">
        {/* Animated Background Pattern */}
        <div className="absolute inset-0 opacity-10">
          <div className="absolute inset-0 bg-[linear-gradient(45deg,transparent_45%,rgba(255,255,255,0.1)_50%,transparent_55%)] bg-[length:20px_20px]" />
        </div>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <div className="grid md:grid-cols-4 gap-8">
            {stats.map((stat, index) => {
              const Icon = stat.icon;
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, scale: 0.8 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="text-center group"
                >
                  <div className="inline-flex items-center justify-center h-14 w-14 bg-white/20 backdrop-blur rounded-2xl mb-4 group-hover:scale-110 transition-transform">
                    <Icon className="h-7 w-7 text-white" />
                  </div>
                  <div className="text-5xl font-bold text-white mb-2">{stat.value}</div>
                  <div className="text-blue-100 text-lg">{stat.label}</div>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="py-24 px-4 sm:px-6 lg:px-8 bg-gradient-to-b from-slate-50 to-white dark:from-slate-900 dark:to-slate-800">
        <div className="max-w-7xl mx-auto">
          <motion.div
            className="text-center max-w-3xl mx-auto mb-16"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
          >
            <span className="text-blue-600 dark:text-blue-400 font-semibold text-sm uppercase tracking-wider">How It Works</span>
            <h2 className="mt-4 text-4xl lg:text-5xl font-bold text-gray-900 dark:text-white">
              Your Path to
              <span className="bg-gradient-to-r from-blue-600 to-cyan-500 bg-clip-text text-transparent"> Success</span>
            </h2>
            <p className="mt-6 text-xl text-gray-600 dark:text-gray-400">
              Get started in minutes. Our AI handles the complexity so you can focus on your workouts.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 relative">
            {/* Connection Line - centered through icons (h-16 = 64px, so top-8 = 32px = center) */}
            <div className="hidden lg:block absolute top-8 left-[12.5%] right-[12.5%] h-0.5 bg-gradient-to-r from-blue-500 via-purple-500 to-blue-500 z-0" />

            {howItWorks.map((item, index) => {
              const Icon = item.icon;
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="relative text-center"
                >
                  {/* Step Number Background - clearly visible */}
                  <div className="absolute -top-4 left-1/2 -translate-x-1/2 text-9xl font-black text-blue-100 dark:text-slate-700 select-none pointer-events-none" style={{ zIndex: 0 }}>
                    {item.step}
                  </div>
                  {/* Icon Badge */}
                  <div className="inline-flex items-center justify-center h-16 w-16 bg-gradient-to-br from-blue-600 to-blue-700 rounded-2xl shadow-lg shadow-blue-500/25 mb-6 relative z-10">
                    <Icon className="h-8 w-8 text-white" />
                  </div>
                  <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-3 relative z-10">
                    {item.title}
                  </h3>
                  <p className="text-gray-600 dark:text-gray-400 relative z-10">
                    {item.description}
                  </p>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>


      {/* Testimonials Section */}
      <section id="testimonials" className="py-24 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <motion.div
            className="text-center max-w-3xl mx-auto mb-16"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
          >
            <span className="text-blue-600 dark:text-blue-400 font-semibold text-sm uppercase tracking-wider">Testimonials</span>
            <h2 className="mt-4 text-4xl lg:text-5xl font-bold text-gray-900 dark:text-white">
              Loved by
              <span className="bg-gradient-to-r from-blue-600 to-cyan-500 bg-clip-text text-transparent"> Fitness Enthusiasts</span>
            </h2>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8">
            {testimonials.map((testimonial, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="bg-white dark:bg-slate-800/50 rounded-3xl p-8 shadow-lg border border-gray-100 dark:border-slate-700 hover:shadow-xl transition-shadow"
              >
                {/* Stars */}
                <div className="flex mb-6">
                  {[...Array(testimonial.rating)].map((_, i) => (
                    <Star key={i} className="h-5 w-5 text-yellow-400 fill-current" />
                  ))}
                </div>

                <p className="text-gray-700 dark:text-gray-300 text-lg leading-relaxed mb-8">
                  "{testimonial.content}"
                </p>

                <div className="flex items-center">
                  <div className="h-12 w-12 bg-gradient-to-br from-blue-500 to-blue-600 rounded-full flex items-center justify-center text-white font-bold">
                    {testimonial.avatar}
                  </div>
                  <div className="ml-4">
                    <div className="font-semibold text-gray-900 dark:text-white">{testimonial.name}</div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">{testimonial.role}</div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 px-4 sm:px-6 lg:px-8">
        <motion.div
          className="max-w-5xl mx-auto relative overflow-hidden rounded-[3rem] bg-gradient-to-r from-blue-600 via-blue-700 to-blue-800 p-12 lg:p-20 shadow-2xl"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
        >
          {/* Background Elements */}
          <div className="absolute top-0 right-0 w-96 h-96 bg-white/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2" />
          <div className="absolute bottom-0 left-0 w-64 h-64 bg-cyan-500/20 rounded-full blur-2xl translate-y-1/2 -translate-x-1/2" />

          <div className="relative z-10 text-center">
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              viewport={{ once: true }}
            >
              <Award className="h-16 w-16 text-white/80 mx-auto mb-6" />
              <h2 className="text-4xl lg:text-5xl font-bold text-white mb-6">
                Ready to Transform Your Fitness?
              </h2>
              <p className="text-xl text-blue-100 mb-10 max-w-2xl mx-auto">
                Join thousands of users who are already achieving their fitness goals with AI-powered training.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Link
                  to="/signup"
                  className="group inline-flex items-center justify-center bg-white text-blue-700 px-8 py-4 rounded-2xl font-semibold text-lg hover:bg-blue-50 transition-all duration-300 shadow-xl hover:-translate-y-1"
                >
                  Start Your Free Trial
                  <ArrowRight className="ml-2 h-5 w-5 group-hover:translate-x-1 transition-transform" />
                </Link>
                <Link
                  to="/login"
                  className="inline-flex items-center justify-center border-2 border-white/30 text-white px-8 py-4 rounded-2xl font-semibold text-lg hover:bg-white/10 transition-all duration-300"
                >
                  Sign In
                </Link>
              </div>
            </motion.div>
          </div>
        </motion.div>
      </section>

      {/* Footer */}
      <footer className="bg-slate-900 dark:bg-black text-white py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid md:grid-cols-5 gap-12">
            {/* Brand */}
            <div className="md:col-span-2">
              <div className="flex items-center mb-6">
                <div className="h-12 w-12 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg">
                  <Dumbbell className="h-6 w-6 text-white" />
                </div>
                <span className="ml-3 text-2xl font-bold">AIthlete</span>
              </div>
              <p className="text-gray-400 mb-6 leading-relaxed">
                Your AI-powered fitness companion for personalized training, smart nutrition, and real-time form analysis.
              </p>
              <div className="flex space-x-4">
                <Shield className="h-6 w-6 text-gray-500" />
                <span className="text-gray-500 text-sm">Secure & Private</span>
              </div>
            </div>

            {/* Links */}
            <div>
              <h3 className="font-semibold text-lg mb-4">Product</h3>
              <ul className="space-y-3 text-gray-400">
                <li><a href="#features" className="hover:text-white transition-colors">Features</a></li>
                <li><a href="#how-it-works" className="hover:text-white transition-colors">How it Works</a></li>
                <li><Link to="/pricing" className="hover:text-white transition-colors">Pricing</Link></li>
              </ul>
            </div>

            <div>
              <h3 className="font-semibold text-lg mb-4">Company</h3>
              <ul className="space-y-3 text-gray-400">
                <li><Link to="/about" className="hover:text-white transition-colors">About</Link></li>
                <li><Link to="/blog" className="hover:text-white transition-colors">Blog</Link></li>
                <li><Link to="/careers" className="hover:text-white transition-colors">Careers</Link></li>
              </ul>
            </div>

            <div>
              <h3 className="font-semibold text-lg mb-4">Support</h3>
              <ul className="space-y-3 text-gray-400">
                <li><Link to="/help" className="hover:text-white transition-colors">Help Center</Link></li>
                <li><Link to="/contact" className="hover:text-white transition-colors">Contact</Link></li>
                <li><Link to="/privacy" className="hover:text-white transition-colors">Privacy Policy</Link></li>
              </ul>
            </div>
          </div>

          <div className="mt-16 pt-8 border-t border-slate-800 flex flex-col md:flex-row justify-between items-center">
            <p className="text-gray-500 text-sm">
              © {new Date().getFullYear()} AIthlete. All rights reserved.
            </p>
            <p className="text-gray-500 text-sm mt-4 md:mt-0">
              Built with ❤️ by Rishith Kumar Pachipulusu
            </p>
          </div>
        </div>
      </footer>

      {/* CSS for Marquee Animation */}
      <style>{`
        @keyframes marquee {
          0% { transform: translateX(0); }
          100% { transform: translateX(-50%); }
        }
        .animate-marquee {
          animation: marquee 20s linear infinite;
        }
      `}</style>
    </div>
  );
};

export default Landing;
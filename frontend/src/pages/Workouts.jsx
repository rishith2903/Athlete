import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Dumbbell, 
  Clock, 
  Flame, 
  Plus, 
  Filter,
  Search,
  Play,
  Heart,
  TrendingUp,
  ChevronRight
} from 'lucide-react';
import { Link } from 'react-router-dom';
import { workoutAPI } from '../services/api';

const Workouts = () => {
  const [workouts, setWorkouts] = useState([]);

  const [selectedCategory, setSelectedCategory] = useState('All');
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchWorkouts = async () => {
      try {
        setLoading(true);
        const res = await workoutAPI.getWorkouts();
        const page = res.data;
        setWorkouts(page?.content || []);
      } catch (e) {
        // keep empty
      } finally {
        setLoading(false);
      }
    };
    fetchWorkouts();
  }, []);

  const categories = ['All', 'HIIT', 'Strength', 'Cardio', 'Yoga', 'Core', 'Flexibility'];

  const filteredWorkouts = workouts.filter(workout => {
    const category = workout.category || workout.type || 'Mixed';
    const matchesCategory = selectedCategory === 'All' || category.toLowerCase() === selectedCategory.toLowerCase();
    const matchesSearch = (workout.name || '').toLowerCase().includes(searchQuery.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  const titleCase = (s) => (s || '').toLowerCase().replace(/(^|_|-)([a-z])/g, (_, __, c) => ` ${c.toUpperCase()}`).trim();

  const getDifficultyColor = (difficulty) => {
    const d = (difficulty || '').toUpperCase();
    switch(d) {
      case 'BEGINNER': return 'bg-green-100 text-green-700';
      case 'INTERMEDIATE': return 'bg-yellow-100 text-yellow-700';
      case 'ADVANCED': return 'bg-red-100 text-red-700';
      case 'EXPERT': return 'bg-purple-100 text-purple-700';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Workouts</h1>
          <p className="mt-1 text-gray-600">Choose from our AI-powered workout plans</p>
        </div>
        <Link to="/workouts" className="mt-4 sm:mt-0 inline-flex items-center btn-primary" onClick={async (e)=>{ e.preventDefault(); try{ const res = await workoutAPI.generateAIWorkout({}); const planName = res.data?.workout?.name || 'AI Workout'; alert(`Generated: ${planName}`);}catch(err){ alert('Failed to generate AI workout'); } }}>
          <Plus className="h-5 w-5 mr-2" />
          Generate AI Workout
        </Link>
      </div>

      {/* Search and Filters */}
      <div className="bg-white rounded-xl p-4 shadow-sm">
        <div className="flex flex-col lg:flex-row gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search workouts..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
          </div>
          <div className="flex gap-2 overflow-x-auto pb-2 lg:pb-0">
            {categories.map(category => (
              <button
                key={category}
                onClick={() => setSelectedCategory(category)}
                className={`px-4 py-2 rounded-lg font-medium whitespace-nowrap transition-colors ${
                  selectedCategory === category
                    ? 'bg-primary-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {category}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Workouts Grid */}
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
        {loading && (
          <div className="col-span-full text-center text-gray-600">Loading workouts…</div>
        )}
        {!loading && filteredWorkouts.length === 0 && (
          <div className="col-span-full text-center text-gray-600">No workouts found.</div>
        )}
        {!loading && filteredWorkouts.map((workout, index) => (
          <motion.div
            key={workout.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="bg-white rounded-xl shadow-sm hover:shadow-lg transition-shadow overflow-hidden"
          >
              <div className="h-48 bg-gradient-to-br from-primary-500 to-primary-700 relative">
                <div className="absolute inset-0 flex items-center justify-center">
                  <Dumbbell className="h-24 w-24 text-white/20" />
                </div>
                <div className="absolute top-4 right-4">
                  <span className={`px-3 py-1 rounded-full text-xs font-medium ${getDifficultyColor(workout.difficulty)}`}>
                    {titleCase(workout.difficulty)}
                  </span>
                </div>
              </div>
              <div className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">{workout.name}</h3>
                <p className="text-sm text-gray-600 mb-4">{workout.description || 'Personalized workout plan'}</p>
              
              <div className="flex items-center gap-4 mb-4 text-sm text-gray-500">
                <div className="flex items-center">
                  <Clock className="h-4 w-4 mr-1" />
                  {workout.duration || 0} min
                </div>
                <div className="flex items-center">
                  <Flame className="h-4 w-4 mr-1" />
                  {workout.caloriesBurned ?? '—'} cal
                </div>
              </div>

              <div className="flex flex-wrap gap-2 mb-4">
                {(workout.exercises || []).slice(0,3).map((ex, i) => (
                  <span key={i} className="px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded">
                    {ex.name}
                  </span>
                ))}
              </div>

              <Link 
                to={`/workouts/${workout.id}`}
                className="w-full btn-primary flex items-center justify-center"
              >
                View Details
                <Play className="h-4 w-4 ml-2" />
              </Link>
            </div>
          </motion.div>
        ))}
      </div>

      {/* AI Recommendation */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gradient-to-r from-primary-50 to-primary-100 rounded-xl p-6"
      >
        <div className="flex items-start space-x-4">
          <div className="h-12 w-12 bg-primary-600 rounded-lg flex items-center justify-center flex-shrink-0">
            <TrendingUp className="h-6 w-6 text-white" />
          </div>
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">AI Recommendation</h3>
            <p className="text-gray-600 mb-4">
              Based on your recent performance, we recommend trying the "Progressive Strength" program to help you reach your muscle-building goals faster.
            </p>
            <Link to="/workouts/recommendations" className="inline-flex items-center text-primary-600 font-medium hover:text-primary-700">
              View Personalized Plans
              <ChevronRight className="h-4 w-4 ml-1" />
            </Link>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default Workouts;
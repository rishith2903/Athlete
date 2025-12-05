import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Activity,
  TrendingUp,
  Calendar,
  Target,
  Clock,
  Flame,
  Award,
  ChevronRight,
  Plus,
  Camera
} from 'lucide-react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { progressAPI, workoutAPI, nutritionAPI } from '../services/api';

const Dashboard = () => {
  const { user } = useAuth();
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    caloriesBurned: 2450,
    workoutsCompleted: 12,
    currentStreak: 7,
    weeklyGoal: 5,
    weeklyProgress: 3
  });

  const [progressData, setProgressData] = useState([
    { name: 'Mon', calories: 400, weight: 155 },
    { name: 'Tue', calories: 300, weight: 154.5 },
    { name: 'Wed', calories: 520, weight: 154 },
    { name: 'Thu', calories: 450, weight: 154 },
    { name: 'Fri', calories: 380, weight: 153.5 },
    { name: 'Sat', calories: 400, weight: 153.5 },
    { name: 'Sun', calories: 0, weight: 153 }
  ]);

  const [nutritionData, setNutritionData] = useState([
    { name: 'Protein', value: 35, color: '#3b82f6' },
    { name: 'Carbs', value: 45, color: '#10b981' },
    { name: 'Fats', value: 20, color: '#f59e0b' }
  ]);

  const [recentWorkouts, setRecentWorkouts] = useState([]);
  const [workoutStats, setWorkoutStats] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch recent workouts list
        const list = await workoutAPI.getWorkouts();
        const page = list.data;
        const items = (page?.content || []).slice(0, 5).map(w => ({
          id: w.id,
          name: w.name,
          duration: `${w.duration || 0} min`,
          calories: w.caloriesBurned ?? '—',
          date: w.completedAt ? new Date(w.completedAt).toLocaleDateString() : (w.startedAt ? 'In progress' : 'Planned'),
        }));
        setRecentWorkouts(items);
        
        // Fetch workout statistics for the last 30 days
        try {
          const statsResponse = await workoutAPI.getStatistics(30);
          setWorkoutStats(statsResponse.data);
          
          // Update the stats display with real data if available
          if (statsResponse.data) {
            setStats(prev => ({
              ...prev,
              caloriesBurned: statsResponse.data.totalCalories || prev.caloriesBurned,
              workoutsCompleted: statsResponse.data.totalWorkouts || prev.workoutsCompleted
            }));
          }
        } catch (statsError) {
          console.log('Could not fetch workout statistics');
        }
      } catch (e) {
        // keep fallback empty state
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  const StatCard = ({ icon: Icon, title, value, change, color }) => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow"
    >
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-600 mb-1">{title}</p>
          <p className="text-2xl font-bold text-gray-900">{value}</p>
          {change && (
            <p className={`text-sm mt-2 flex items-center ${change > 0 ? 'text-green-600' : 'text-red-600'}`}>
              <TrendingUp className="h-4 w-4 mr-1" />
              {change > 0 ? '+' : ''}{change}%
            </p>
          )}
        </div>
        <div className={`h-12 w-12 rounded-lg ${color} flex items-center justify-center`}>
          <Icon className="h-6 w-6 text-white" />
        </div>
      </div>
    </motion.div>
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="spinner"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Welcome Section */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gradient-to-r from-primary-600 to-primary-700 rounded-2xl p-8 text-white"
      >
        <h1 className="text-3xl font-bold mb-2">Welcome back, {user?.name?.split(' ')[0]}! 👋</h1>
        <p className="text-primary-100 mb-6">You're doing great! Keep up the momentum.</p>
        <div className="flex flex-wrap gap-4">
          <Link to="/workouts/new" className="inline-flex items-center bg-white text-primary-600 px-6 py-3 rounded-lg font-medium hover:bg-gray-100 transition-colors">
            <Plus className="h-5 w-5 mr-2" />
            Start Workout
          </Link>
          <Link to="/nutrition/track" className="inline-flex items-center bg-primary-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-primary-400 transition-colors">
            Track Meal
          </Link>
        </div>
      </motion.div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          icon={Flame}
          title="Calories Burned"
          value={stats.caloriesBurned.toLocaleString()}
          change={12}
          color="bg-orange-500"
        />
        <StatCard
          icon={Activity}
          title="Workouts"
          value={stats.workoutsCompleted}
          change={8}
          color="bg-primary-600"
        />
        <StatCard
          icon={Award}
          title="Current Streak"
          value={`${stats.currentStreak} days`}
          color="bg-green-500"
        />
        <StatCard
          icon={Target}
          title="Weekly Goal"
          value={`${stats.weeklyProgress}/${stats.weeklyGoal}`}
          color="bg-purple-500"
        />
      </div>

      {/* Recent Activity Summary */}
      {workoutStats && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
          className="bg-white rounded-xl p-6 shadow-sm"
        >
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-900">Recent Activity (Last 30 Days)</h2>
            <Link to="/workouts" className="text-sm text-primary-600 hover:text-primary-700 font-medium">
              View Details
            </Link>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600">Total Workouts</span>
                <Activity className="h-4 w-4 text-gray-400" />
              </div>
              <p className="text-2xl font-bold text-gray-900">{workoutStats.totalWorkouts}</p>
              <p className="text-xs text-gray-500 mt-1">
                {workoutStats.averageWorkoutsPerWeek?.toFixed(1)} per week
              </p>
            </div>
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600">Total Minutes</span>
                <Clock className="h-4 w-4 text-gray-400" />
              </div>
              <p className="text-2xl font-bold text-gray-900">{workoutStats.totalMinutes}</p>
              <p className="text-xs text-gray-500 mt-1">
                {(workoutStats.totalMinutes / 60).toFixed(1)} hours
              </p>
            </div>
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600">Calories Burned</span>
                <Flame className="h-4 w-4 text-gray-400" />
              </div>
              <p className="text-2xl font-bold text-gray-900">{workoutStats.totalCalories?.toLocaleString()}</p>
              <p className="text-xs text-gray-500 mt-1">
                {(workoutStats.totalCalories / 30).toFixed(0)} per day avg
              </p>
            </div>
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600">Consistency</span>
                <Award className="h-4 w-4 text-gray-400" />
              </div>
              <p className="text-2xl font-bold text-gray-900">
                {((workoutStats.totalWorkouts / 30) * 100).toFixed(0)}%
              </p>
              <p className="text-xs text-gray-500 mt-1">
                {workoutStats.totalWorkouts} of 30 days
              </p>
            </div>
          </div>
        </motion.div>
      )}

      {/* Charts Section */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Weight Progress Chart */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-white rounded-xl p-6 shadow-sm"
        >
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-900">Weight Progress</h2>
            <select className="text-sm border border-gray-200 rounded-lg px-3 py-1">
              <option>This Week</option>
              <option>This Month</option>
              <option>This Year</option>
            </select>
          </div>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={progressData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="name" stroke="#9ca3af" />
              <YAxis stroke="#9ca3af" />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="weight"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={{ fill: '#3b82f6', r: 4 }}
                activeDot={{ r: 6 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Calories Burned Chart */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-white rounded-xl p-6 shadow-sm"
        >
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-900">Calories Burned</h2>
            <select className="text-sm border border-gray-200 rounded-lg px-3 py-1">
              <option>This Week</option>
              <option>This Month</option>
            </select>
          </div>
          <ResponsiveContainer width="100%" height={250}>
            <AreaChart data={progressData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="name" stroke="#9ca3af" />
              <YAxis stroke="#9ca3af" />
              <Tooltip />
              <Area
                type="monotone"
                dataKey="calories"
                stroke="#10b981"
                fill="#10b981"
                fillOpacity={0.2}
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      {/* Nutrition and Recent Workouts */}
      <div className="grid lg:grid-cols-3 gap-6">
        {/* Nutrition Breakdown */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="bg-white rounded-xl p-6 shadow-sm"
        >
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Today's Nutrition</h2>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={nutritionData}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={80}
                paddingAngle={5}
                dataKey="value"
              >
                {nutritionData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
          <div className="mt-4 space-y-2">
            {nutritionData.map((item, index) => (
              <div key={index} className="flex items-center justify-between">
                <div className="flex items-center">
                  <div className={`h-3 w-3 rounded-full mr-2`} style={{ backgroundColor: item.color }}></div>
                  <span className="text-sm text-gray-600">{item.name}</span>
                </div>
                <span className="text-sm font-medium text-gray-900">{item.value}%</span>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Recent Workouts */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="lg:col-span-2 bg-white rounded-xl p-6 shadow-sm"
        >
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-900">Recent Workouts</h2>
            <Link to="/workouts" className="text-sm text-primary-600 hover:text-primary-700 font-medium">
              View All
            </Link>
          </div>
          <div className="space-y-3">
            {recentWorkouts.map((workout) => (
              <div key={workout.id} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
                <div className="flex items-center space-x-4">
                  <div className="h-10 w-10 bg-primary-100 rounded-lg flex items-center justify-center">
                    <Activity className="h-5 w-5 text-primary-600" />
                  </div>
                  <div>
                    <p className="font-medium text-gray-900">{workout.name}</p>
                    <p className="text-sm text-gray-500">{workout.date}</p>
                  </div>
                </div>
                <div className="flex items-center space-x-4">
                  <div className="text-right">
                    <p className="text-sm font-medium text-gray-900">{workout.calories} cal</p>
                    <p className="text-xs text-gray-500">{workout.duration}</p>
                  </div>
                  <ChevronRight className="h-5 w-5 text-gray-400" />
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Quick Actions */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="bg-gradient-to-r from-gray-50 to-gray-100 rounded-xl p-6"
      >
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Link to="/pose-analysis" className="flex flex-col items-center p-4 bg-white rounded-lg hover:shadow-md transition-shadow">
            <div className="h-12 w-12 bg-blue-100 rounded-lg flex items-center justify-center mb-2">
              <Camera className="h-6 w-6 text-blue-600" />
            </div>
            <span className="text-sm font-medium text-gray-900">Check Form</span>
          </Link>
          <Link to="/chatbot" className="flex flex-col items-center p-4 bg-white rounded-lg hover:shadow-md transition-shadow">
            <div className="h-12 w-12 bg-purple-100 rounded-lg flex items-center justify-center mb-2">
              <Activity className="h-6 w-6 text-purple-600" />
            </div>
            <span className="text-sm font-medium text-gray-900">AI Coach</span>
          </Link>
          <Link to="/progress" className="flex flex-col items-center p-4 bg-white rounded-lg hover:shadow-md transition-shadow">
            <div className="h-12 w-12 bg-green-100 rounded-lg flex items-center justify-center mb-2">
              <TrendingUp className="h-6 w-6 text-green-600" />
            </div>
            <span className="text-sm font-medium text-gray-900">View Progress</span>
          </Link>
          <Link to="/schedule" className="flex flex-col items-center p-4 bg-white rounded-lg hover:shadow-md transition-shadow">
            <div className="h-12 w-12 bg-orange-100 rounded-lg flex items-center justify-center mb-2">
              <Calendar className="h-6 w-6 text-orange-600" />
            </div>
            <span className="text-sm font-medium text-gray-900">Schedule</span>
          </Link>
        </div>
      </motion.div>
    </div>
  );
};

export default Dashboard;
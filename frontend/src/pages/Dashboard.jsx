import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Activity,
  Flame,
  Award,
  Target,
  ChevronRight
} from 'lucide-react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts';
import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { workoutAPI } from '../services/api';

// Dashboard Components
import { StatCard, WelcomeBanner, ActivitySummary, QuickActions } from '../components/dashboard';

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

  const [progressData] = useState([
    { name: 'Mon', calories: 400, weight: 155 },
    { name: 'Tue', calories: 300, weight: 154.5 },
    { name: 'Wed', calories: 520, weight: 154 },
    { name: 'Thu', calories: 450, weight: 154 },
    { name: 'Fri', calories: 380, weight: 153.5 },
    { name: 'Sat', calories: 400, weight: 153.5 },
    { name: 'Sun', calories: 0, weight: 153 }
  ]);

  const [nutritionData] = useState([
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

  return (
    <div className="space-y-6">
      {/* Welcome Section */}
      <WelcomeBanner userName={user?.name} />

      {/* Stats Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6">
        <StatCard
          icon={Flame}
          title="Calories Burned"
          value={stats.caloriesBurned.toLocaleString()}
          change={12}
          color="bg-orange-500"
          loading={loading}
        />
        <StatCard
          icon={Activity}
          title="Workouts"
          value={stats.workoutsCompleted}
          change={8}
          color="bg-primary-600"
          loading={loading}
        />
        <StatCard
          icon={Award}
          title="Current Streak"
          value={`${stats.currentStreak} days`}
          color="bg-green-500"
          loading={loading}
        />
        <StatCard
          icon={Target}
          title="Weekly Goal"
          value={`${stats.weeklyProgress}/${stats.weeklyGoal}`}
          color="bg-purple-500"
          loading={loading}
        />
      </div>

      {/* Recent Activity Summary */}
      <ActivitySummary stats={workoutStats} loading={loading} />

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
            <select className="text-sm border border-gray-200 rounded-lg px-3 py-1 focus:outline-none focus:ring-2 focus:ring-primary-500">
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
            <select className="text-sm border border-gray-200 rounded-lg px-3 py-1 focus:outline-none focus:ring-2 focus:ring-primary-500">
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
                  <div className="h-3 w-3 rounded-full mr-2" style={{ backgroundColor: item.color }}></div>
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
          {recentWorkouts.length === 0 ? (
            <div className="text-center py-8">
              <div className="h-12 w-12 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-3">
                <Activity className="h-6 w-6 text-gray-400" />
              </div>
              <p className="text-gray-500">No workouts yet. Start your fitness journey!</p>
              <Link to="/workouts/generate" className="text-sm text-primary-600 hover:text-primary-700 font-medium mt-2 inline-block">
                Generate a Workout
              </Link>
            </div>
          ) : (
            <div className="space-y-3">
              {recentWorkouts.map((workout) => (
                <Link
                  key={workout.id}
                  to={`/workouts/${workout.id}`}
                  className="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
                >
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
                </Link>
              ))}
            </div>
          )}
        </motion.div>
      </div>

      {/* Quick Actions */}
      <QuickActions />
    </div>
  );
};

export default Dashboard;
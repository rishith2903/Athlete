import React, { useEffect, useState } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { workoutAPI } from '../services/api';
import { Loader2, ArrowLeft, Play, CheckCircle2, Clock, Flame, Dumbbell } from 'lucide-react';
import { motion } from 'framer-motion';

const WorkoutDetail = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [workout, setWorkout] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [acting, setActing] = useState(false);

  const fetchWorkout = async () => {
    try {
      setLoading(true);
      const res = await workoutAPI.getWorkout(id);
      setWorkout(res.data);
    } catch (e) {
      setError(e.response?.data?.message || 'Failed to load workout');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchWorkout();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  const start = async () => {
    try {
      setActing(true);
      const res = await workoutAPI.startWorkout(id);
      setWorkout(res.data);
    } catch (e) {
      alert('Failed to start workout');
    } finally {
      setActing(false);
    }
  };

  const [rating, setRating] = useState(5);
  const [exertion, setExertion] = useState(7);
  const [notes, setNotes] = useState('');

  const complete = async () => {
    try {
      setActing(true);
      const completionData = { rating, perceivedExertion: exertion, notes };
      const res = await workoutAPI.completeWorkout(id, completionData);
      setWorkout(res.data);
    } catch (e) {
      alert('Failed to complete workout');
    } finally {
      setActing(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="h-6 w-6 animate-spin text-gray-500" />
      </div>
    );
  }
  if (error) {
    return (
      <div className="max-w-3xl mx-auto">
        <button onClick={() => navigate(-1)} className="inline-flex items-center text-primary-600 hover:text-primary-700 mb-4">
          <ArrowLeft className="h-4 w-4 mr-2" /> Back
        </button>
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">{error}</div>
      </div>
    );
  }
  if (!workout) return null;

  const status = (workout.status || 'PLANNED').toUpperCase();

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <button onClick={() => navigate(-1)} className="inline-flex items-center text-primary-600 hover:text-primary-700">
          <ArrowLeft className="h-4 w-4 mr-2" /> Back
        </button>
        <Link className="text-sm text-gray-600 hover:text-gray-800" to="/workouts">All Workouts</Link>
      </div>

      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="bg-white rounded-xl p-6 shadow-sm">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">{workout.name}</h1>
            <p className="text-gray-600 mt-1">{workout.description || 'Personalized workout plan'}</p>
          </div>
          <div className="flex items-center gap-3 text-sm">
            <div className="px-3 py-2 rounded-lg bg-gray-100 text-gray-800 inline-flex items-center">
              <Clock className="h-4 w-4 mr-2" /> {workout.duration || 0} min
            </div>
            <div className="px-3 py-2 rounded-lg bg-orange-100 text-orange-800 inline-flex items-center">
              <Flame className="h-4 w-4 mr-2" /> {workout.caloriesBurned ?? '—'} kcal
            </div>
            <div className="px-3 py-2 rounded-lg bg-blue-100 text-blue-800 inline-flex items-center">
              <Dumbbell className="h-4 w-4 mr-2" /> {workout.difficulty}
            </div>
          </div>
        </div>

        <div className="mt-4 flex items-center gap-3">
          <span className={`px-3 py-1 rounded-full text-xs font-medium ${status === 'COMPLETED' ? 'bg-green-100 text-green-700' : status === 'IN_PROGRESS' ? 'bg-yellow-100 text-yellow-700' : 'bg-gray-100 text-gray-700'}`}>
            {status}
          </span>
          {workout.startedAt && <span className="text-xs text-gray-500">Started: {new Date(workout.startedAt).toLocaleString()}</span>}
          {workout.completedAt && <span className="text-xs text-gray-500">Completed: {new Date(workout.completedAt).toLocaleString()}</span>}
        </div>

        <div className="mt-6 flex gap-3">
          <button
            onClick={start}
            disabled={acting || status !== 'PLANNED'}
            className={`inline-flex items-center px-4 py-2 rounded-lg font-medium ${status === 'PLANNED' ? 'bg-primary-600 text-white hover:bg-primary-700' : 'bg-gray-200 text-gray-600 cursor-not-allowed'}`}
          >
            {acting && status === 'PLANNED' ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Play className="h-4 w-4 mr-2" />}
            Start
          </button>
          <button
            onClick={complete}
            disabled={acting || status !== 'IN_PROGRESS'}
            className={`inline-flex items-center px-4 py-2 rounded-lg font-medium ${status === 'IN_PROGRESS' ? 'bg-green-600 text-white hover:bg-green-700' : 'bg-gray-200 text-gray-600 cursor-not-allowed'}`}
          >
            {acting && status === 'IN_PROGRESS' ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <CheckCircle2 className="h-4 w-4 mr-2" />}
            Complete
          </button>
        </div>
      </motion.div>

      {/* Completion Form */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="bg-white rounded-xl p-6 shadow-sm">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Completion Feedback</h2>
        <div className="grid md:grid-cols-3 gap-4">
          <div>
            <label className="label">Rating (1-5)</label>
            <select className="input" value={rating} onChange={(e)=>setRating(Number(e.target.value))}>
              {[1,2,3,4,5].map(n=> <option key={n} value={n}>{n}</option>)}
            </select>
          </div>
          <div>
            <label className="label">Perceived Exertion (1-10)</label>
            <select className="input" value={exertion} onChange={(e)=>setExertion(Number(e.target.value))}>
              {[1,2,3,4,5,6,7,8,9,10].map(n=> <option key={n} value={n}>{n}</option>)}
            </select>
          </div>
          <div className="md:col-span-3">
            <label className="label">Notes</label>
            <textarea className="input" rows={3} placeholder="How did it go?" value={notes} onChange={(e)=>setNotes(e.target.value)} />
          </div>
        </div>
      </motion.div>

      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="bg-white rounded-xl p-6 shadow-sm">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Exercises</h2>
        <div className="space-y-3">
          {(workout.exercises || []).map((ex, i) => (
            <div key={i} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
              <div>
                <p className="font-medium text-gray-900">{ex.name}</p>
                <p className="text-sm text-gray-500">
                  {ex.sets ? `${ex.sets} sets` : ''} {ex.reps ? `· ${ex.reps} reps` : ''} {ex.duration ? `· ${ex.duration}s` : ''} {ex.restTime ? `· rest ${ex.restTime}s` : ''}
                </p>
              </div>
            </div>
          ))}
        </div>
      </motion.div>

      {workout.aiGenerated && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="bg-blue-50 border border-blue-200 rounded-xl p-4">
          <h3 className="font-medium text-blue-900 mb-2">AI Metadata</h3>
          <pre className="text-xs text-blue-900 whitespace-pre-wrap">{JSON.stringify(workout.aiMetadata, null, 2)}</pre>
        </motion.div>
      )}
    </div>
  );
};

export default WorkoutDetail;
import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Clock, Flame, Dumbbell, ChevronRight, Loader2 } from 'lucide-react';
import { workoutAPI } from '../services/api';

const WorkoutGenerate = () => {
  const [prefs, setPrefs] = useState({
    fitnessGoal: 'general_fitness',
    activityLevel: 'INTERMEDIATE',
    equipment: [],
    duration: 45,
  });
  const [plan, setPlan] = useState(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);

  const toggleEquip = (name) => {
    setPrefs((prev) => {
      const set = new Set(prev.equipment);
      if (set.has(name)) set.delete(name); else set.add(name);
      return { ...prev, equipment: Array.from(set) };
    });
  };

  const generate = async () => {
    try {
      setLoading(true);
      setError(null);
      setPlan(null);
      const res = await workoutAPI.generateAIWorkout({
        fitnessGoal: prefs.fitnessGoal,
        activityLevel: prefs.activityLevel,
        equipment: prefs.equipment,
        workoutDuration: prefs.duration,
      });
      setPlan(res.data);
    } catch (e) {
      setError(e.response?.data?.message || 'Failed to generate workout');
    } finally {
      setLoading(false);
    }
  };

  const mapPlanToWorkout = (p) => {
    const exercises = (p?.workout?.exercises || []).map((ex) => ({
      name: ex.name,
      sets: ex.sets ?? null,
      reps: ex.reps ?? null,
      duration: ex.duration ?? null,
      restTime: ex.rest ?? ex.restTime ?? null,
      equipment: ex.equipment ?? null,
      muscleGroup: Array.isArray(ex.muscleGroups) ? ex.muscleGroups[0] : (ex.muscleGroup ?? null),
      instructions: ex.instructions ?? null,
    }));
    return {
      name: p?.workout?.name || 'AI Workout',
      description: 'AI-generated workout plan',
      type: p?.workout?.type || 'MIXED',
      difficulty: p?.difficulty || p?.workout?.difficulty || 'INTERMEDIATE',
      duration: p?.duration || p?.workout?.duration || 45,
      caloriesBurned: p?.estimatedCalories || 0,
      exercises,
      aiGenerated: true,
      aiMetadata: p,
      status: 'PLANNED',
    };
  };

  const savePlan = async () => {
    if (!plan) return;
    try {
      setSaving(true);
      const workout = mapPlanToWorkout(plan);
      await workoutAPI.createWorkout(workout);
      alert('Workout saved to your plans.');
    } catch (e) {
      alert('Failed to save workout');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Generate AI Workout</h1>
        <p className="mt-1 text-gray-600">Personalized plan based on your preferences</p>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Preferences */}
        <div className="bg-white rounded-xl p-6 shadow-sm space-y-4">
          <div>
            <label className="label">Goal</label>
            <select className="input" value={prefs.fitnessGoal} onChange={(e)=>setPrefs(p=>({...p, fitnessGoal: e.target.value}))}>
              <option value="general_fitness">General Fitness</option>
              <option value="LOSE_WEIGHT">Weight Loss</option>
              <option value="MUSCLE_GAIN">Muscle Gain</option>
              <option value="ENDURANCE">Endurance</option>
              <option value="STRENGTH">Strength</option>
            </select>
          </div>
          <div>
            <label className="label">Activity level</label>
            <select className="input" value={prefs.activityLevel} onChange={(e)=>setPrefs(p=>({...p, activityLevel: e.target.value}))}>
              <option value="BEGINNER">Beginner</option>
              <option value="INTERMEDIATE">Intermediate</option>
              <option value="ADVANCED">Advanced</option>
            </select>
          </div>
          <div>
            <label className="label">Duration (minutes)</label>
            <input type="number" className="input" min={15} max={120} value={prefs.duration} onChange={(e)=>setPrefs(p=>({...p, duration: Number(e.target.value)}))} />
          </div>
          <div>
            <label className="label">Equipment</label>
            <div className="flex flex-wrap gap-2">
              {['dumbbells','barbell','bench','mat','kettlebell'].map((eq)=> (
                <button key={eq} type="button" onClick={()=>toggleEquip(eq)} className={`px-3 py-1 rounded-full text-sm ${prefs.equipment.includes(eq) ? 'bg-primary-600 text-white' : 'bg-gray-100 text-gray-700'}`}>{eq}</button>
              ))}
            </div>
          </div>
          <div className="flex gap-3">
            <button onClick={generate} disabled={loading} className={`flex-1 btn-primary inline-flex items-center justify-center ${loading ? 'opacity-70' : ''}`}>
              {loading ? <Loader2 className="h-5 w-5 animate-spin mr-2" /> : <Dumbbell className="h-5 w-5 mr-2" />}
              Generate Workout
            </button>
            <button onClick={savePlan} disabled={!plan || saving} className={`flex-1 bg-gray-200 text-gray-800 rounded-lg font-medium px-4 py-2 hover:bg-gray-300 transition-colors ${(!plan || saving) ? 'opacity-60 cursor-not-allowed' : ''}`}>
              {saving ? 'Saving…' : 'Save Plan'}
            </button>
          </div>
          {error && <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">{error}</div>}
        </div>

        {/* Plan */}
        <div className="lg:col-span-2 space-y-6">
          {!plan && !loading && (
            <div className="bg-gray-50 rounded-xl p-6 text-center text-gray-600">Choose preferences and generate a plan.</div>
          )}
          {plan && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
              <div className="bg-white rounded-xl p-6 shadow-sm">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-lg font-semibold text-gray-900">{plan.workout?.name || 'AI Workout Plan'}</h2>
                    <p className="text-sm text-gray-600">Difficulty: {plan.difficulty || plan.workout?.difficulty || '—'}</p>
                  </div>
                  <div className="flex items-center gap-3 text-sm">
                    <div className="px-4 py-2 rounded-lg bg-gray-100 text-gray-800 inline-flex items-center"><Clock className="h-4 w-4 mr-2" />{plan.duration || plan.workout?.duration || 0} min</div>
                    <div className="px-4 py-2 rounded-lg bg-orange-100 text-orange-800 inline-flex items-center"><Flame className="h-4 w-4 mr-2" />{plan.estimatedCalories || '—'} kcal</div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-xl p-6 shadow-sm space-y-3">
                {(plan.workout?.exercises || []).map((ex, i) => (
                  <div key={i} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                    <div>
                      <p className="font-medium text-gray-900">{ex.name}</p>
                      <p className="text-sm text-gray-500">{ex.sets ? `${ex.sets} sets` : ''} {ex.reps ? `· ${ex.reps} reps` : ''} {ex.duration ? `· ${ex.duration}s` : ''}</p>
                    </div>
                    <ChevronRight className="h-5 w-5 text-gray-400" />
                  </div>
                ))}
              </div>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
};

export default WorkoutGenerate;

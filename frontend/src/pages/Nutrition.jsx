import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Loader2, Utensils, Flame } from 'lucide-react';
import { nutritionAPI } from '../services/api';

const Nutrition = () => {
  const [preferences, setPreferences] = useState({
    mealsPerDay: 4,
    dietType: 'balanced',
    allergies: [],
    restrictions: [],
  });
  const [plan, setPlan] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const toggleItem = (key, value) => {
    setPreferences((prev) => {
      const arr = new Set(prev[key] || []);
      if (arr.has(value)) arr.delete(value); else arr.add(value);
      return { ...prev, [key]: Array.from(arr) };
    });
  };

  const generatePlan = async () => {
    try {
      setLoading(true);
      setError(null);
      setPlan(null);
      const res = await nutritionAPI.generateAIDietPlan({
        mealsPerDay: preferences.mealsPerDay,
        dietType: preferences.dietType,
        allergies: preferences.allergies,
        restrictions: preferences.restrictions,
      });
      setPlan(res.data);
    } catch (e) {
      setError(e.response?.data?.message || 'Failed to generate nutrition plan');
    } finally {
      setLoading(false);
    }
  };

  const MacroPill = ({ label, value, color }) => (
    <div className={`px-4 py-2 rounded-lg text-white font-medium ${color}`}>
      {label}: {value}g
    </div>
  );

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Nutrition Plan</h1>
        <p className="mt-1 text-gray-600">Generate an AI meal plan tailored to your goals</p>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Preferences */}
        <div className="bg-white rounded-xl p-6 shadow-sm">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Preferences</h2>

          <div className="space-y-4">
            <div>
              <label className="label">Meals per day</label>
              <select
                className="input"
                value={preferences.mealsPerDay}
                onChange={(e) => setPreferences((p) => ({ ...p, mealsPerDay: Number(e.target.value) }))}
              >
                {[3,4,5,6].map((n) => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="label">Diet type</label>
              <select
                className="input"
                value={preferences.dietType}
                onChange={(e) => setPreferences((p) => ({ ...p, dietType: e.target.value }))}
              >
                <option value="balanced">Balanced</option>
                <option value="high_protein">High Protein</option>
                <option value="low_carb">Low Carb</option>
                <option value="pescatarian">Pescatarian</option>
                <option value="vegetarian">Vegetarian</option>
              </select>
            </div>

            <div>
              <label className="label">Allergies</label>
              <div className="flex flex-wrap gap-2">
                {['peanuts','dairy','gluten','soy','shellfish'].map((a) => (
                  <button
                    key={a}
                    type="button"
                    onClick={() => toggleItem('allergies', a)}
                    className={`px-3 py-1 rounded-full text-sm ${preferences.allergies.includes(a) ? 'bg-primary-600 text-white' : 'bg-gray-100 text-gray-700'}`}
                  >
                    {a}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="label">Restrictions</label>
              <div className="flex flex-wrap gap-2">
                {['no_pork','no_beef','no_fish','no_eggs'].map((r) => (
                  <button
                    key={r}
                    type="button"
                    onClick={() => toggleItem('restrictions', r)}
                    className={`px-3 py-1 rounded-full text-sm ${preferences.restrictions.includes(r) ? 'bg-primary-600 text-white' : 'bg-gray-100 text-gray-700'}`}
                  >
                    {r.replace('no_','no ')}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <button
                onClick={generatePlan}
                disabled={loading}
                className={`w-full btn-primary inline-flex items-center justify-center ${loading ? 'opacity-70' : ''}`}
              >
                {loading ? <Loader2 className="h-5 w-5 animate-spin mr-2" /> : <Utensils className="h-5 w-5 mr-2" />}
                Generate Plan
              </button>
            </div>

            {error && (
              <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">{error}</div>
            )}
          </div>
        </div>

        {/* Plan Output */}
        <div className="lg:col-span-2 space-y-6">
          {!plan && !loading && (
            <div className="bg-gray-50 rounded-xl p-6 text-center text-gray-600">
              Choose your preferences and click Generate Plan to see results here.
            </div>
          )}

          {plan && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
              <div className="bg-white rounded-xl p-6 shadow-sm">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-lg font-semibold text-gray-900">Daily Targets</h2>
                    <p className="text-sm text-gray-600">Generated at {new Date(plan.generatedAt).toLocaleString()}</p>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="px-4 py-2 rounded-lg bg-orange-100 text-orange-800 font-medium inline-flex items-center">
                      <Flame className="h-4 w-4 mr-2" /> {plan.totalCalories} kcal
                    </div>
                    <MacroPill label="Protein" value={plan.macros?.protein ?? 0} color="bg-blue-600" />
                    <MacroPill label="Carbs" value={plan.macros?.carbs ?? 0} color="bg-green-600" />
                    <MacroPill label="Fat" value={plan.macros?.fat ?? 0} color="bg-yellow-600" />
                  </div>
                </div>
              </div>

              {['breakfast','lunch','dinner','snacks'].map((slot) => (
                <div key={slot} className="bg-white rounded-xl p-6 shadow-sm">
                  <div className="flex items-center mb-4">
                    <Utensils className="h-5 w-5 text-primary-600 mr-2" />
                    <h3 className="text-lg font-semibold text-gray-900 capitalize">{slot}</h3>
                  </div>
                  <div className="space-y-3">
                    {(plan.meals?.[slot] || []).map((item, idx) => (
                      <div key={idx} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                        <div>
                          <p className="font-medium text-gray-900">{item.name}</p>
                          <p className="text-sm text-gray-500">{item.quantity || ''}</p>
                        </div>
                        <div className="text-right text-sm text-gray-600">
                          <div>{item.calories} kcal</div>
                          <div>Protein {item.protein}g · Carbs {item.carbs}g · Fat {item.fat}g</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}

              {Array.isArray(plan.notes) && plan.notes.length > 0 && (
                <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
                  <h4 className="font-medium text-blue-900 mb-2">Notes</h4>
                  <ul className="text-sm text-blue-800 list-disc pl-4">
                    {plan.notes.map((n, i) => <li key={i}>{n}</li>)}
                  </ul>
                </div>
              )}
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Nutrition;

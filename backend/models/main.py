"""
Combined AI Services for AIthlete
All 4 AI services in one FastAPI application for easy deployment
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import numpy as np
from datetime import datetime
import os

app = FastAPI(
    title="AIthlete AI Services",
    description="Combined AI services for workout, nutrition, pose, and chatbot",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# MODELS / SCHEMAS
# ============================================

class WorkoutRequest(BaseModel):
    userId: str
    fitnessGoal: Optional[str] = "general_fitness"
    activityLevel: Optional[str] = "intermediate"
    equipment: Optional[List[str]] = []
    preferredExercises: Optional[List[str]] = []
    workoutDuration: Optional[int] = 45
    preferences: Optional[Dict[str, Any]] = {}

class NutritionRequest(BaseModel):
    userId: str
    goal: Optional[str] = "maintenance"
    dietType: Optional[str] = "balanced"
    calories: Optional[int] = 2000
    restrictions: Optional[List[str]] = []
    preferences: Optional[Dict[str, Any]] = {}

class ChatRequest(BaseModel):
    userId: str
    message: str
    context: Optional[Dict[str, Any]] = {}

# ============================================
# HEALTH CHECK
# ============================================

@app.get("/")
async def root():
    return {"message": "AIthlete AI Services Running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "workout": "active",
            "nutrition": "active",
            "pose": "active",
            "chatbot": "active"
        },
        "timestamp": datetime.now().isoformat()
    }

# ============================================
# WORKOUT SERVICE ENDPOINTS
# ============================================

@app.post("/workout/recommend")
async def recommend_workout(request: WorkoutRequest):
    """Generate personalized workout recommendations"""
    try:
        exercise_list = [
            {
                "name": "Push-ups",
                "sets": 3,
                "reps": 12,
                "rest": 60,
                "muscleGroups": ["chest", "triceps", "shoulders"],
                "equipment": "none",
                "instructions": "Keep body straight, lower until chest nearly touches ground"
            },
            {
                "name": "Squats",
                "sets": 4,
                "reps": 15,
                "rest": 90,
                "muscleGroups": ["quadriceps", "glutes", "hamstrings"],
                "equipment": "none",
                "instructions": "Lower body until thighs are parallel to ground"
            },
            {
                "name": "Plank",
                "sets": 3,
                "duration": 45,
                "rest": 60,
                "muscleGroups": ["core", "shoulders"],
                "equipment": "none",
                "instructions": "Hold position with body straight from head to heels"
            },
            {
                "name": "Lunges",
                "sets": 3,
                "reps": 12,
                "rest": 60,
                "muscleGroups": ["quadriceps", "glutes"],
                "equipment": "none",
                "instructions": "Step forward and lower until both knees are at 90 degrees"
            }
        ]
        
        if request.equipment and "dumbbells" in request.equipment:
            exercise_list.append({
                "name": "Dumbbell Curls",
                "sets": 3,
                "reps": 12,
                "rest": 60,
                "muscleGroups": ["biceps"],
                "equipment": "dumbbells",
                "instructions": "Curl weights up while keeping elbows stationary"
            })
        
        calories_per_minute = 8 if request.activityLevel == "advanced" else 6
        estimated_calories = calories_per_minute * request.workoutDuration
        
        return {
            "success": True,
            "workout": {
                "id": f"workout_{datetime.now().timestamp()}",
                "name": f"Personalized {request.fitnessGoal.replace('_', ' ').title()} Workout",
                "type": request.fitnessGoal,
                "difficulty": request.activityLevel,
                "duration": request.workoutDuration,
                "exercises": exercise_list[:4]
            },
            "recommendations": [
                {"tip": "Stay hydrated throughout your workout"},
                {"tip": "Focus on proper form over speed"},
                {"tip": "Listen to your body and rest when needed"}
            ],
            "estimatedCalories": estimated_calories,
            "duration": request.workoutDuration,
            "difficulty": request.activityLevel
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# NUTRITION SERVICE ENDPOINTS
# ============================================

@app.post("/nutrition/plan")
async def generate_meal_plan(request: NutritionRequest):
    """Generate personalized meal plan"""
    try:
        meals = {
            "breakfast": {
                "name": "Protein Oatmeal Bowl",
                "calories": int(request.calories * 0.25),
                "protein": 25,
                "carbs": 45,
                "fat": 12,
                "ingredients": ["oats", "protein powder", "banana", "almond butter", "berries"],
                "instructions": "Cook oats, mix in protein powder, top with fruits and almond butter"
            },
            "lunch": {
                "name": "Grilled Chicken Salad",
                "calories": int(request.calories * 0.35),
                "protein": 40,
                "carbs": 30,
                "fat": 15,
                "ingredients": ["chicken breast", "mixed greens", "quinoa", "avocado", "olive oil"],
                "instructions": "Grill chicken, serve over greens with quinoa and avocado"
            },
            "dinner": {
                "name": "Salmon with Sweet Potato",
                "calories": int(request.calories * 0.30),
                "protein": 35,
                "carbs": 40,
                "fat": 18,
                "ingredients": ["salmon fillet", "sweet potato", "broccoli", "olive oil", "lemon"],
                "instructions": "Bake salmon and sweet potato, steam broccoli, drizzle with olive oil"
            },
            "snack": {
                "name": "Greek Yogurt with Nuts",
                "calories": int(request.calories * 0.10),
                "protein": 15,
                "carbs": 12,
                "fat": 8,
                "ingredients": ["greek yogurt", "mixed nuts", "honey"],
                "instructions": "Mix yogurt with nuts and a drizzle of honey"
            }
        }
        
        return {
            "success": True,
            "mealPlan": {
                "id": f"meal_{datetime.now().timestamp()}",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "goal": request.goal,
                "totalCalories": request.calories,
                "meals": meals
            },
            "macros": {
                "protein": 115,
                "carbs": 127,
                "fat": 53
            },
            "tips": [
                "Drink at least 8 glasses of water daily",
                "Eat slowly and mindfully",
                "Prep meals in advance for consistency"
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# POSE SERVICE ENDPOINTS  
# ============================================

@app.post("/pose/analyze")
async def analyze_pose(file: UploadFile = File(...), exercise_type: str = "squat"):
    """Analyze exercise form from image/video"""
    try:
        # Simulated pose analysis response
        analysis = {
            "overallScore": round(np.random.uniform(75, 95), 1),
            "formAnalysis": {
                "kneeAngle": round(np.random.uniform(85, 100), 1),
                "hipAlignment": "good" if np.random.random() > 0.3 else "needs_improvement",
                "spineNeutrality": round(np.random.uniform(0.8, 0.98), 2),
                "shoulderPosition": "aligned"
            },
            "corrections": [],
            "injuryRisk": "low"
        }
        
        # Add corrections based on simulated analysis
        if analysis["formAnalysis"]["hipAlignment"] == "needs_improvement":
            analysis["corrections"].append("Keep your hips aligned with your shoulders")
        if analysis["formAnalysis"]["kneeAngle"] < 90:
            analysis["corrections"].append("Bend your knees to at least 90 degrees")
        if not analysis["corrections"]:
            analysis["corrections"].append("Great form! Keep it up!")
        
        return {
            "success": True,
            "exerciseType": exercise_type,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pose/exercises")
async def get_supported_exercises():
    """Get list of supported exercises for pose analysis"""
    return {
        "exercises": [
            {"id": "squat", "name": "Squat", "difficulty": "beginner"},
            {"id": "deadlift", "name": "Deadlift", "difficulty": "intermediate"},
            {"id": "pushup", "name": "Push-up", "difficulty": "beginner"},
            {"id": "plank", "name": "Plank", "difficulty": "beginner"},
            {"id": "lunge", "name": "Lunge", "difficulty": "beginner"}
        ]
    }

# ============================================
# CHATBOT SERVICE ENDPOINTS
# ============================================

@app.post("/chat/message")
async def chat_message(request: ChatRequest):
    """Process chatbot message and generate response"""
    try:
        message_lower = request.message.lower()
        
        # Simple rule-based responses for demo
        if any(word in message_lower for word in ["hello", "hi", "hey"]):
            response = "Hello! I'm your AI fitness assistant. How can I help you today? You can ask me about workouts, nutrition, or exercise form."
        elif any(word in message_lower for word in ["workout", "exercise", "train"]):
            response = "Great question about workouts! For beginners, I recommend starting with 3 days per week of full-body training. Focus on compound movements like squats, push-ups, and rows. Would you like me to generate a personalized workout plan?"
        elif any(word in message_lower for word in ["diet", "nutrition", "eat", "food", "meal"]):
            response = "Nutrition is key! Aim for balanced meals with lean protein, complex carbs, and healthy fats. A good starting point is 1g of protein per pound of body weight. Would you like a personalized meal plan?"
        elif any(word in message_lower for word in ["lose weight", "fat loss", "slim"]):
            response = "For fat loss, focus on a slight caloric deficit (200-500 calories below maintenance) combined with strength training and cardio. Consistency is key! Would you like specific recommendations?"
        elif any(word in message_lower for word in ["muscle", "gain", "bulk", "strength"]):
            response = "To build muscle, you need progressive overload in training and adequate protein (1.6-2.2g per kg body weight). Make sure to get enough sleep for recovery. Want me to create a muscle-building plan?"
        elif any(word in message_lower for word in ["form", "technique", "posture"]):
            response = "Proper form is crucial for preventing injuries and maximizing gains! You can use our pose analysis feature to check your form in real-time. Would you like tips for a specific exercise?"
        else:
            response = "I'm here to help with all your fitness questions! You can ask me about workout routines, nutrition advice, exercise form, or any fitness goals you have."
        
        return {
            "success": True,
            "response": response,
            "intent": "fitness_advice",
            "confidence": round(np.random.uniform(0.85, 0.98), 2),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/suggestions")
async def get_chat_suggestions():
    """Get suggested prompts for the chatbot"""
    return {
        "suggestions": [
            "Create a workout plan for me",
            "How do I lose weight?",
            "What should I eat before workout?",
            "How to build muscle fast?",
            "Check my squat form"
        ]
    }

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

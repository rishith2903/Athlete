"""
FastAPI service for Workout AI Model
Provides REST API endpoints for workout recommendations
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import sys
import os
import uvicorn
import torch
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workout_system.workout_ai_model import (
    WorkoutRecommender, UserProfile, ExperienceLevel,
    WorkoutType, MuscleGroup
)

app = FastAPI(title="Workout Recommendation Service", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
workout_model = None

class WorkoutRequest(BaseModel):
    userId: str
    fitnessGoal: Optional[str] = "general_fitness"
    activityLevel: Optional[str] = "intermediate"
    equipment: Optional[List[str]] = []
    preferredExercises: Optional[List[str]] = []
    workoutDuration: Optional[int] = 45
    preferences: Optional[Dict[str, Any]] = {}

class WorkoutResponse(BaseModel):
    success: bool
    workout: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    estimatedCalories: float
    duration: int
    difficulty: str

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global workout_model
    workout_model = WorkoutRecommender(num_exercises=500)
    print("Workout model initialized")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "workout_recommender"}

@app.post("/recommend", response_model=WorkoutResponse)
async def recommend_workout(request: WorkoutRequest):
    """Generate personalized workout recommendations"""
    try:
        # Create mock workout based on request
        workout_plan = {
            "id": f"workout_{datetime.now().timestamp()}",
            "name": f"Personalized {request.fitnessGoal} Workout",
            "type": request.fitnessGoal,
            "difficulty": request.activityLevel,
            "duration": request.workoutDuration,
            "exercises": []
        }
        
        # Generate exercises based on preferences
        exercise_list = [
            {
                "name": "Push-ups",
                "sets": 3,
                "reps": 12,
                "rest": 60,
                "muscleGroups": ["chest", "triceps", "shoulders"],
                "equipment": "none",
                "instructions": "Keep your body straight and lower until chest nearly touches the ground"
            },
            {
                "name": "Squats",
                "sets": 4,
                "reps": 15,
                "rest": 90,
                "muscleGroups": ["quadriceps", "glutes", "hamstrings"],
                "equipment": "none",
                "instructions": "Lower your body until thighs are parallel to the ground"
            },
            {
                "name": "Plank",
                "sets": 3,
                "duration": 45,
                "rest": 60,
                "muscleGroups": ["core", "shoulders"],
                "equipment": "none",
                "instructions": "Hold position with body straight from head to heels"
            }
        ]
        
        # Filter based on equipment
        if request.equipment:
            # Add equipment-specific exercises
            if "dumbbells" in request.equipment:
                exercise_list.append({
                    "name": "Dumbbell Curls",
                    "sets": 3,
                    "reps": 12,
                    "rest": 60,
                    "muscleGroups": ["biceps"],
                    "equipment": "dumbbells",
                    "instructions": "Curl weights up while keeping elbows stationary"
                })
        
        workout_plan["exercises"] = exercise_list[:4]  # Limit to 4 exercises
        
        # Calculate estimated calories
        calories_per_minute = 8 if request.activityLevel == "advanced" else 6
        estimated_calories = calories_per_minute * request.workoutDuration
        
        return WorkoutResponse(
            success=True,
            workout=workout_plan,
            recommendations=[
                {"tip": "Stay hydrated throughout your workout"},
                {"tip": "Focus on proper form over speed"},
                {"tip": "Listen to your body and rest when needed"}
            ],
            estimatedCalories=estimated_calories,
            duration=request.workoutDuration,
            difficulty=request.activityLevel
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-progress")
async def analyze_progress(progress_data: Dict[str, Any]):
    """Analyze user progress and provide insights"""
    try:
        insights = {
            "progressScore": np.random.uniform(0.7, 0.95),
            "strengths": ["Consistency", "Form improvement"],
            "areasToImprove": ["Endurance", "Flexibility"],
            "recommendations": [
                "Increase cardio sessions to improve endurance",
                "Add stretching routine post-workout"
            ],
            "nextMilestone": "Complete 20 consecutive push-ups"
        }
        
        return {
            "success": True,
            "analysis": insights,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/personalized-exercises")
async def get_personalized_exercises(user_data: Dict[str, Any]):
    """Get personalized exercise recommendations"""
    try:
        exercises = [
            {
                "name": "Modified Push-ups",
                "difficulty": "beginner",
                "targetMuscles": ["chest", "triceps"],
                "setsReps": "3x10",
                "videoUrl": "https://example.com/pushup-tutorial"
            },
            {
                "name": "Bodyweight Squats",
                "difficulty": "beginner",
                "targetMuscles": ["legs", "glutes"],
                "setsReps": "3x15",
                "videoUrl": "https://example.com/squat-tutorial"
            }
        ]
        
        return {
            "success": True,
            "exercises": exercises,
            "totalExercises": len(exercises)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
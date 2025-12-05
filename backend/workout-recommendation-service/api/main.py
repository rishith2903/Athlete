"""
FastAPI Service for Workout Recommendation Model
Serves personalized workout plans via REST API
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import torch
import joblib
import pandas as pd
import json
import os
import logging
from contextlib import asynccontextmanager

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.workout_recommender import (
    HybridWorkoutRecommender,
    UserProfile,
    ReinforcementLearningAdapter
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
model = None
rl_adapter = None
exercise_database = None
model_metadata = None

# Pydantic models for API
class UserProfileRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    age: int = Field(..., ge=16, le=100, description="User age")
    gender: str = Field(..., pattern="^(male|female|other)$", description="User gender")
    fitness_level: str = Field(..., pattern="^(beginner|intermediate|advanced)$", description="Fitness level")
    goals: List[str] = Field(..., description="Fitness goals (weight_loss, muscle_gain, endurance, strength, flexibility)")
    available_equipment: List[str] = Field(default=["none"], description="Available workout equipment")
    workout_days_per_week: int = Field(..., ge=1, le=7, description="Number of workout days per week")
    session_duration_minutes: int = Field(..., ge=15, le=180, description="Workout session duration in minutes")
    injuries: Optional[List[str]] = Field(default=None, description="Current injuries or limitations")
    preferences: Optional[Dict] = Field(default=None, description="Additional preferences")

class WorkoutPlanResponse(BaseModel):
    user_id: str
    generated_at: str
    program_duration_weeks: int
    weekly_plan: Dict
    progression_strategy: Dict
    notes: List[str]

class ExerciseDetail(BaseModel):
    exercise: str
    sets: int
    reps: Optional[int] = None
    duration: Optional[str] = None
    rest_seconds: int
    muscle_groups: List[str]
    notes: str

class DayWorkout(BaseModel):
    day: str
    workout: List[ExerciseDetail]

class UserFeedback(BaseModel):
    user_id: str
    exercise_id: int
    completed: bool
    rating: int = Field(..., ge=1, le=5)
    felt_good: Optional[bool] = None
    too_hard: Optional[bool] = None
    too_easy: Optional[bool] = None
    injury: Optional[bool] = None
    notes: Optional[str] = None

class ProgressUpdate(BaseModel):
    user_id: str
    workout_id: str
    completed_at: str
    exercises_completed: List[str]
    total_duration_minutes: int
    calories_burned: int
    feedback: Optional[str] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global model, rl_adapter, exercise_database, model_metadata
    
    logger.info("Loading workout recommendation model...")
    
    try:
        # Load model
        model_path = os.getenv("MODEL_PATH", "models/trained/")
        
        # Load metadata
        with open(f"{model_path}/metadata.json", 'r') as f:
            model_metadata = json.load(f)
        
        # Load exercise database
        exercise_database = pd.read_csv(f"{model_path}/exercises.csv")
        
        # Load model weights
        checkpoint = torch.load(
            f"{model_path}/workout_recommender.pth",
            map_location=torch.device('cpu')
        )
        
        # Initialize model
        model = HybridWorkoutRecommender(
            num_exercises=checkpoint['num_exercises'],
            num_users=checkpoint['num_users']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Initialize RL adapter
        rl_adapter = ReinforcementLearningAdapter()
        rl_adapter.q_network.load_state_dict(checkpoint['rl_adapter_state_dict'])
        
        logger.info("Model loaded successfully")
        logger.info(f"Model version: {model_metadata.get('model_version', 'unknown')}")
        
    except Exception as e:
        logger.warning(f"Could not load trained model: {str(e)}")
        logger.info("Initializing new model...")
        
        # Initialize new model with default parameters
        model = HybridWorkoutRecommender(num_exercises=20, num_users=1000)
        rl_adapter = ReinforcementLearningAdapter()
        
        # Load default exercise database
        exercise_database = pd.DataFrame([
            {"exercise_id": i, "name": f"Exercise {i}", "category": "strength"}
            for i in range(20)
        ])
        
        model_metadata = {"model_version": "1.0.0", "trained_at": datetime.now().isoformat()}
    
    yield
    
    # Cleanup
    logger.info("Shutting down workout recommendation service...")

# Create FastAPI app
app = FastAPI(
    title="Workout Recommendation Service",
    description="AI-powered personalized workout plan generation",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Workout Recommendation Service",
        "version": model_metadata.get("model_version", "1.0.0") if model_metadata else "1.0.0",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_status": "loaded" if model is not None else "not_loaded"
    }

@app.post("/api/generate-workout", response_model=WorkoutPlanResponse)
async def generate_workout_plan(request: UserProfileRequest):
    """Generate personalized workout plan for a user"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request to UserProfile
        user_profile = UserProfile(
            user_id=request.user_id,
            age=request.age,
            gender=request.gender,
            fitness_level=request.fitness_level,
            goals=request.goals,
            available_equipment=request.available_equipment,
            workout_days_per_week=request.workout_days_per_week,
            session_duration_minutes=request.session_duration_minutes,
            injuries=request.injuries,
            preferences=request.preferences
        )
        
        # Generate workout plan
        with torch.no_grad():
            workout_plan = model(user_profile)
        
        # Apply RL adaptations if available
        if rl_adapter and hasattr(user_profile, 'user_embedding'):
            adaptive_scores = rl_adapter.get_adaptive_scores(user_profile.user_embedding)
            # Apply adaptive scores to workout plan (simplified)
            # In production, this would modify the workout selection
        
        return WorkoutPlanResponse(**workout_plan)
        
    except Exception as e:
        logger.error(f"Error generating workout plan: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating workout plan: {str(e)}")

@app.get("/api/workout/{user_id}/current")
async def get_current_workout(user_id: str, day: Optional[str] = None):
    """Get current day's workout for a user"""
    
    # In production, this would fetch from database
    # For now, generate a new plan
    try:
        if day is None:
            day = datetime.now().strftime("%A")
        
        # Mock data for demonstration
        return {
            "user_id": user_id,
            "day": day,
            "workout": [
                {
                    "exercise": "Squats",
                    "sets": 4,
                    "reps": 12,
                    "rest_seconds": 60,
                    "muscle_groups": ["quadriceps", "glutes"],
                    "notes": "Keep chest up, knees tracking over toes"
                },
                {
                    "exercise": "Push-ups",
                    "sets": 3,
                    "reps": 15,
                    "rest_seconds": 45,
                    "muscle_groups": ["chest", "triceps"],
                    "notes": "Maintain straight body line"
                },
                {
                    "exercise": "Plank",
                    "sets": 3,
                    "duration": "60s",
                    "rest_seconds": 30,
                    "muscle_groups": ["core"],
                    "notes": "Keep body straight, breathe normally"
                }
            ],
            "estimated_duration": 45,
            "estimated_calories": 350
        }
        
    except Exception as e:
        logger.error(f"Error fetching workout: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching workout")

@app.post("/api/feedback")
async def submit_feedback(feedback: UserFeedback, background_tasks: BackgroundTasks):
    """Submit user feedback for workout/exercise"""
    
    try:
        # Calculate reward for RL
        reward = rl_adapter.calculate_reward(feedback.dict())
        
        # Update RL model in background
        background_tasks.add_task(
            update_rl_model,
            feedback.user_id,
            feedback.exercise_id,
            reward
        )
        
        return {
            "status": "success",
            "message": "Feedback received",
            "reward": reward
        }
        
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing feedback")

@app.post("/api/progress")
async def update_progress(progress: ProgressUpdate):
    """Update user workout progress"""
    
    try:
        # In production, save to database
        # For now, just acknowledge
        return {
            "status": "success",
            "message": "Progress updated",
            "workout_id": progress.workout_id,
            "completion_rate": len(progress.exercises_completed) / 10  # Assuming 10 exercises
        }
        
    except Exception as e:
        logger.error(f"Error updating progress: {str(e)}")
        raise HTTPException(status_code=500, detail="Error updating progress")

@app.get("/api/exercises")
async def get_exercise_database():
    """Get available exercises database"""
    
    if exercise_database is None:
        raise HTTPException(status_code=503, detail="Exercise database not loaded")
    
    try:
        return exercise_database.to_dict(orient="records")
        
    except Exception as e:
        logger.error(f"Error fetching exercises: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching exercises")

@app.get("/api/workout/variations/{exercise_name}")
async def get_exercise_variations(exercise_name: str):
    """Get variations for a specific exercise"""
    
    variations = {
        "push-up": [
            {"name": "Standard Push-up", "difficulty": "beginner"},
            {"name": "Wide-Grip Push-up", "difficulty": "intermediate"},
            {"name": "Diamond Push-up", "difficulty": "advanced"},
            {"name": "Incline Push-up", "difficulty": "beginner"},
            {"name": "Decline Push-up", "difficulty": "intermediate"}
        ],
        "squat": [
            {"name": "Bodyweight Squat", "difficulty": "beginner"},
            {"name": "Goblet Squat", "difficulty": "intermediate"},
            {"name": "Front Squat", "difficulty": "advanced"},
            {"name": "Jump Squat", "difficulty": "intermediate"},
            {"name": "Pistol Squat", "difficulty": "advanced"}
        ],
        "plank": [
            {"name": "Standard Plank", "difficulty": "beginner"},
            {"name": "Side Plank", "difficulty": "intermediate"},
            {"name": "Plank with Leg Lift", "difficulty": "intermediate"},
            {"name": "Plank Jacks", "difficulty": "advanced"},
            {"name": "Plank to Push-up", "difficulty": "advanced"}
        ]
    }
    
    exercise_lower = exercise_name.lower()
    for key in variations:
        if key in exercise_lower:
            return {"exercise": exercise_name, "variations": variations[key]}
    
    return {"exercise": exercise_name, "variations": []}

@app.get("/api/workout/statistics/{user_id}")
async def get_user_statistics(user_id: str):
    """Get user workout statistics"""
    
    # Mock statistics for demonstration
    return {
        "user_id": user_id,
        "total_workouts": 42,
        "current_streak": 7,
        "total_calories_burned": 12500,
        "favorite_exercises": ["Squats", "Push-ups", "Plank"],
        "improvements": {
            "strength": "+15%",
            "endurance": "+20%",
            "flexibility": "+10%"
        },
        "next_milestone": "50 workouts completed",
        "recommendation": "Consider increasing workout intensity by 10%"
    }

async def update_rl_model(user_id: str, exercise_id: int, reward: float):
    """Background task to update RL model"""
    try:
        # In production, this would update the model with new data
        logger.info(f"Updating RL model for user {user_id}, exercise {exercise_id}, reward {reward}")
        
        # Create dummy states for demonstration
        state = torch.randn(64)
        next_state = torch.randn(64)
        
        # Update RL adapter
        rl_adapter.update(state, exercise_id, reward, next_state)
        
    except Exception as e:
        logger.error(f"Error updating RL model: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8001)),
        reload=os.getenv("ENV", "development") == "development"
    )
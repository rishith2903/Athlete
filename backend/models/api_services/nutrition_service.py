"""
FastAPI service for Nutrition AI Model
Provides REST API endpoint for meal plan generation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np

app = FastAPI(title="Nutrition Planner Service", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserNutritionProfile(BaseModel):
    userId: Optional[str] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    activityLevel: Optional[str] = "moderate"
    fitnessGoal: Optional[str] = "general_fitness"
    targetWeight: Optional[float] = None
    allergies: Optional[List[str]] = []
    dietaryRestrictions: Optional[List[str]] = []
    preferences: Optional[Dict[str, Any]] = {}

class MealItem(BaseModel):
    name: str
    calories: int
    protein: int
    carbs: int
    fat: int
    quantity: Optional[str] = None

class MealPlanResponse(BaseModel):
    success: bool
    userId: Optional[str] = None
    generatedAt: str
    totalCalories: int
    macros: Dict[str, int]
    meals: Dict[str, List[MealItem]]
    notes: List[str]

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "nutrition_planner"}

@app.post("/plan", response_model=MealPlanResponse)
async def plan_nutrition(profile: UserNutritionProfile):
    try:
        base_cal = 2200
        if profile.fitnessGoal and "loss" in profile.fitnessGoal.lower():
            base_cal = 1800
        elif profile.fitnessGoal and ("gain" in profile.fitnessGoal.lower() or "muscle" in profile.fitnessGoal.lower()):
            base_cal = 2600

        protein = int(base_cal * 0.3 / 4)
        carbs = int(base_cal * 0.45 / 4)
        fat = int(base_cal * 0.25 / 9)

        sample_meals = {
            "breakfast": [
                MealItem(name="Greek Yogurt Parfait", calories=350, protein=25, carbs=45, fat=8, quantity="1 bowl"),
            ],
            "lunch": [
                MealItem(name="Grilled Chicken Salad", calories=500, protein=40, carbs=35, fat=18, quantity="1 plate"),
            ],
            "dinner": [
                MealItem(name="Salmon, Quinoa, Veggies", calories=650, protein=45, carbs=55, fat=20, quantity="1 plate"),
            ],
            "snacks": [
                MealItem(name="Mixed Nuts", calories=200, protein=6, carbs=8, fat=18, quantity="30g"),
                MealItem(name="Banana", calories=100, protein=1, carbs=27, fat=0, quantity="1"),
            ],
        }

        return MealPlanResponse(
            success=True,
            userId=profile.userId,
            generatedAt=datetime.utcnow().isoformat(),
            totalCalories=base_cal,
            macros={"protein": protein, "carbs": carbs, "fat": fat},
            meals=sample_meals,
            notes=[
                "This is a baseline plan; adjust based on tolerance and progress.",
                "Hydrate well and aim for whole foods.",
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

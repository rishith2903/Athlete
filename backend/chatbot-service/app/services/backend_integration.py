"""
Backend Integration Service - Connects chatbot with Spring Boot backend
"""

import httpx
from typing import Dict, List, Optional
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class BackendIntegrationService:
    def __init__(self, backend_url: str):
        self.backend_url = backend_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=30.0)
        self.auth_token = None
        
    async def get_user_workouts(self, user_id: str, token: str) -> Optional[List[Dict]]:
        """Fetch user's workout plans from backend"""
        try:
            headers = {"Authorization": f"Bearer {token}"}
            response = await self.client.get(
                f"{self.backend_url}/api/workout",
                headers=headers,
                params={"userId": user_id}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to fetch workouts: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching workouts: {str(e)}")
            return None
    
    async def create_ai_workout(self, user_id: str, token: str, preferences: Dict) -> Optional[Dict]:
        """Request AI-generated workout from backend"""
        try:
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "userId": user_id,
                "fitnessLevel": preferences.get("fitness_level", "beginner"),
                "goals": preferences.get("goals", []),
                "equipment": preferences.get("equipment", []),
                "duration": preferences.get("duration", 45),
                "targetMuscles": preferences.get("target_muscles", [])
            }
            
            response = await self.client.post(
                f"{self.backend_url}/api/workout/ai-generate",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to generate workout: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating workout: {str(e)}")
            return None
    
    async def get_nutrition_plan(self, user_id: str, token: str) -> Optional[Dict]:
        """Fetch user's nutrition plan from backend"""
        try:
            headers = {"Authorization": f"Bearer {token}"}
            response = await self.client.get(
                f"{self.backend_url}/api/nutrition",
                headers=headers,
                params={"userId": user_id}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to fetch nutrition plan: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching nutrition: {str(e)}")
            return None
    
    async def create_ai_meal_plan(self, user_id: str, token: str, preferences: Dict) -> Optional[Dict]:
        """Request AI-generated meal plan from backend"""
        try:
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "userId": user_id,
                "dietType": preferences.get("diet_type", "balanced"),
                "calories": preferences.get("daily_calories", 2000),
                "goals": preferences.get("goals", []),
                "allergies": preferences.get("allergies", []),
                "preferences": preferences.get("food_preferences", [])
            }
            
            response = await self.client.post(
                f"{self.backend_url}/api/nutrition/ai-plan",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to generate meal plan: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating meal plan: {str(e)}")
            return None
    
    async def get_user_progress(self, user_id: str, token: str) -> Optional[Dict]:
        """Fetch user's progress data from backend"""
        try:
            headers = {"Authorization": f"Bearer {token}"}
            response = await self.client.get(
                f"{self.backend_url}/api/progress",
                headers=headers,
                params={"userId": user_id}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to fetch progress: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching progress: {str(e)}")
            return None
    
    async def log_workout_completion(self, user_id: str, token: str, workout_data: Dict) -> bool:
        """Log completed workout to backend"""
        try:
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "userId": user_id,
                "workoutId": workout_data.get("workout_id"),
                "duration": workout_data.get("duration"),
                "caloriesBurned": workout_data.get("calories_burned"),
                "exercises": workout_data.get("exercises", []),
                "completedAt": datetime.utcnow().isoformat()
            }
            
            response = await self.client.post(
                f"{self.backend_url}/api/progress",
                headers=headers,
                json=payload
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Error logging workout: {str(e)}")
            return False
    
    async def submit_form_check(self, user_id: str, token: str, video_url: str, exercise: str) -> Optional[Dict]:
        """Submit exercise form for analysis"""
        try:
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "userId": user_id,
                "videoUrl": video_url,
                "exercise": exercise,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            response = await self.client.post(
                f"{self.backend_url}/api/pose/check",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to submit form check: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error submitting form check: {str(e)}")
            return None
    
    async def get_ai_insights(self, user_id: str, token: str) -> Optional[Dict]:
        """Get AI-generated insights based on user progress"""
        try:
            headers = {"Authorization": f"Bearer {token}"}
            response = await self.client.get(
                f"{self.backend_url}/api/progress/insights",
                headers=headers,
                params={"userId": user_id}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to fetch insights: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching insights: {str(e)}")
            return None
    
    async def save_chat_history(self, user_id: str, token: str, messages: List[Dict]) -> bool:
        """Save chat history to backend"""
        try:
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "userId": user_id,
                "messages": messages,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            response = await self.client.post(
                f"{self.backend_url}/api/chatbot/history",
                headers=headers,
                json=payload
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Error saving chat history: {str(e)}")
            return False
    
    async def get_exercise_database(self) -> Optional[List[Dict]]:
        """Fetch exercise database for recommendations"""
        try:
            response = await self.client.get(
                f"{self.backend_url}/api/exercises/database"
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                # Return fallback exercise list
                return self.get_fallback_exercises()
                
        except Exception as e:
            logger.error(f"Error fetching exercise database: {str(e)}")
            return self.get_fallback_exercises()
    
    def get_fallback_exercises(self) -> List[Dict]:
        """Fallback exercise database"""
        return [
            {
                "name": "Push-up",
                "category": "Upper Body",
                "muscles": ["Chest", "Shoulders", "Triceps"],
                "difficulty": "Beginner",
                "equipment": "None"
            },
            {
                "name": "Squat",
                "category": "Lower Body",
                "muscles": ["Quadriceps", "Glutes", "Hamstrings"],
                "difficulty": "Beginner",
                "equipment": "None"
            },
            {
                "name": "Plank",
                "category": "Core",
                "muscles": ["Core", "Shoulders"],
                "difficulty": "Beginner",
                "equipment": "None"
            },
            {
                "name": "Deadlift",
                "category": "Full Body",
                "muscles": ["Back", "Glutes", "Hamstrings"],
                "difficulty": "Intermediate",
                "equipment": "Barbell"
            },
            {
                "name": "Bench Press",
                "category": "Upper Body",
                "muscles": ["Chest", "Shoulders", "Triceps"],
                "difficulty": "Intermediate",
                "equipment": "Barbell"
            },
            {
                "name": "Pull-up",
                "category": "Upper Body",
                "muscles": ["Back", "Biceps"],
                "difficulty": "Intermediate",
                "equipment": "Pull-up Bar"
            },
            {
                "name": "Lunges",
                "category": "Lower Body",
                "muscles": ["Quadriceps", "Glutes"],
                "difficulty": "Beginner",
                "equipment": "None"
            },
            {
                "name": "Burpees",
                "category": "Full Body",
                "muscles": ["Full Body"],
                "difficulty": "Intermediate",
                "equipment": "None"
            }
        ]
    
    async def get_food_database(self) -> Optional[List[Dict]]:
        """Fetch food/nutrition database"""
        try:
            response = await self.client.get(
                f"{self.backend_url}/api/nutrition/foods"
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return self.get_fallback_foods()
                
        except Exception as e:
            logger.error(f"Error fetching food database: {str(e)}")
            return self.get_fallback_foods()
    
    def get_fallback_foods(self) -> List[Dict]:
        """Fallback food database"""
        return [
            {"name": "Chicken Breast", "calories": 165, "protein": 31, "carbs": 0, "fat": 3.6, "per": "100g"},
            {"name": "Brown Rice", "calories": 130, "protein": 2.7, "carbs": 28, "fat": 1, "per": "100g cooked"},
            {"name": "Banana", "calories": 89, "protein": 1.1, "carbs": 23, "fat": 0.3, "per": "100g"},
            {"name": "Eggs", "calories": 155, "protein": 13, "carbs": 1.1, "fat": 11, "per": "100g"},
            {"name": "Salmon", "calories": 208, "protein": 20, "carbs": 0, "fat": 13, "per": "100g"},
            {"name": "Oatmeal", "calories": 68, "protein": 2.4, "carbs": 12, "fat": 1.4, "per": "100g cooked"},
            {"name": "Greek Yogurt", "calories": 59, "protein": 10, "carbs": 3.6, "fat": 0.4, "per": "100g"},
            {"name": "Sweet Potato", "calories": 86, "protein": 1.6, "carbs": 20, "fat": 0.1, "per": "100g"},
            {"name": "Almonds", "calories": 579, "protein": 21, "carbs": 22, "fat": 50, "per": "100g"},
            {"name": "Broccoli", "calories": 34, "protein": 2.8, "carbs": 7, "fat": 0.4, "per": "100g"}
        ]
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()
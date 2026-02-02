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

# Lazy load MediaPipe to avoid startup issues
mp_pose = None
pose_detector = None

def get_pose_detector():
    global mp_pose, pose_detector
    if pose_detector is None:
        import cv2
        import mediapipe as mp
        from io import BytesIO
        from PIL import Image
        mp_pose = mp.solutions.pose
        pose_detector = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5
        )
    return pose_detector

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
    """Analyze exercise form from image using MediaPipe"""
    try:
        import cv2
        from io import BytesIO
        from PIL import Image
        
        # Get lazy-loaded pose detector
        detector = get_pose_detector()
        
        # Read image from upload
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        image_np = np.array(image)
        
        # Convert RGB for MediaPipe
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image_np
        
        # Detect pose
        results = detector.process(image_rgb)
        
        if not results.pose_landmarks:
            return {
                "success": True,
                "exerciseType": exercise_type,
                "analysis": {
                    "overallScore": 0,
                    "formAnalysis": {},
                    "corrections": ["No pose detected. Please ensure your full body is visible."],
                    "injuryRisk": "unknown"
                },
                "timestamp": datetime.now().isoformat()
            }
        
        # Extract landmarks
        landmarks = results.pose_landmarks.landmark
        
        # Calculate joint angles
        def calculate_angle(a, b, c):
            """Calculate angle between three points"""
            a = np.array([a.x, a.y])
            b = np.array([b.x, b.y])
            c = np.array([c.x, c.y])
            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians * 180.0 / np.pi)
            if angle > 180.0:
                angle = 360 - angle
            return round(angle, 1)
        
        # Key joint indices from MediaPipe
        LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 23, 25, 27
        RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE = 24, 26, 28
        LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST = 11, 13, 15
        RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST = 12, 14, 16
        NOSE = 0
        
        # Calculate angles
        left_knee_angle = calculate_angle(
            landmarks[LEFT_HIP], landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE]
        )
        right_knee_angle = calculate_angle(
            landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE]
        )
        left_hip_angle = calculate_angle(
            landmarks[LEFT_SHOULDER], landmarks[LEFT_HIP], landmarks[LEFT_KNEE]
        )
        right_hip_angle = calculate_angle(
            landmarks[RIGHT_SHOULDER], landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE]
        )
        left_elbow_angle = calculate_angle(
            landmarks[LEFT_SHOULDER], landmarks[LEFT_ELBOW], landmarks[LEFT_WRIST]
        )
        right_elbow_angle = calculate_angle(
            landmarks[RIGHT_SHOULDER], landmarks[RIGHT_ELBOW], landmarks[RIGHT_WRIST]
        )
        
        # Average angles
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
        avg_hip_angle = (left_hip_angle + right_hip_angle) / 2
        avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
        
        # Check symmetry
        knee_symmetry = abs(left_knee_angle - right_knee_angle)
        hip_symmetry = abs(left_hip_angle - right_hip_angle)
        
        # Generate feedback based on exercise type and angles
        corrections = []
        score = 100
        injury_risk = "low"
        
        # Exercise-specific analysis
        if exercise_type in ["squat", "barbell_squat", "goblet_squat", "front_squat"]:
            if avg_knee_angle > 160:
                corrections.append("Bend your knees more - aim for 90° at the bottom")
                score -= 20
            elif avg_knee_angle < 70:
                corrections.append("You're going too deep - stop at 90° knee bend")
                score -= 10
            
            if avg_hip_angle > 150:
                corrections.append("Hinge at your hips more")
                score -= 15
            
            if knee_symmetry > 15:
                corrections.append(f"Your knees are uneven ({knee_symmetry:.0f}° difference)")
                score -= 10
                injury_risk = "medium"
                
        elif exercise_type in ["pushup", "diamond_pushup"]:
            if avg_elbow_angle > 160:
                corrections.append("Lower your body - bend elbows to 90°")
                score -= 20
            elif avg_elbow_angle < 60:
                corrections.append("You're going too low")
                score -= 10
                
        elif exercise_type in ["plank", "side_plank"]:
            if avg_hip_angle < 160:
                corrections.append("Keep your body straight - don't let hips sag")
                score -= 15
                injury_risk = "medium"
            elif avg_hip_angle > 200:
                corrections.append("Lower your hips - body should be straight")
                score -= 15
                
        elif exercise_type in ["lunge", "walking_lunge", "bulgarian_split_squat"]:
            if avg_knee_angle > 160:
                corrections.append("Step deeper into the lunge")
                score -= 15
            if knee_symmetry > 20:
                corrections.append("Keep your front knee stable")
                score -= 10
                
        elif exercise_type in ["deadlift", "romanian_deadlift"]:
            if avg_hip_angle > 160:
                corrections.append("Hinge more at the hips")
                score -= 15
            if avg_knee_angle < 150:
                corrections.append("Keep your knees straighter")
                score -= 10
                
        # General checks
        if hip_symmetry > 15:
            corrections.append(f"Your hips are uneven ({hip_symmetry:.0f}° difference)")
            score -= 10
            
        # Ensure score is in range
        score = max(0, min(100, score))
        
        # Set injury risk based on score
        if score < 50:
            injury_risk = "high"
        elif score < 70:
            injury_risk = "medium"
            
        if not corrections:
            corrections.append("Great form! Keep it up!")
        
        analysis = {
            "overallScore": round(score, 1),
            "formAnalysis": {
                "leftKneeAngle": left_knee_angle,
                "rightKneeAngle": right_knee_angle,
                "avgKneeAngle": round(avg_knee_angle, 1),
                "leftHipAngle": left_hip_angle,
                "rightHipAngle": right_hip_angle,
                "avgHipAngle": round(avg_hip_angle, 1),
                "kneeSymmetry": "good" if knee_symmetry < 10 else "needs_improvement",
                "hipSymmetry": "good" if hip_symmetry < 10 else "needs_improvement"
            },
            "corrections": corrections,
            "injuryRisk": injury_risk
        }
        
        return {
            "success": True,
            "exerciseType": exercise_type,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Pose analysis error: {e}")
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

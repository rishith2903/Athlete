"""
Lightweight Pose Service - No TensorFlow dependency
Uses MediaPipe directly for pose detection
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import uvicorn
import numpy as np
import cv2
import json
from pathlib import Path

app = FastAPI(title="Pose Analysis Service", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load exercise rules
rules_path = Path(__file__).parent.parent / "enhanced_exercise_rules.json"
if rules_path.exists():
    with open(rules_path) as f:
        EXERCISE_RULES = json.load(f)
else:
    EXERCISE_RULES = {"exercises": {}}

class DetailedFeedback(BaseModel):
    joint: str
    userAngle: float
    expectedAngle: float
    idealAngle: float
    deviation: float
    isGood: bool
    feedback: str

class EnhancedFormCheckResponse(BaseModel):
    success: bool
    rating: int
    formScore: float
    detectedExercise: str
    exerciseConfidence: float
    skillLevel: str
    feedback: str
    detailedFeedback: List[DetailedFeedback] = []
    suggestions: List[str] = []
    userAngles: Dict[str, float] = {}
    expectedAngles: Dict[str, float] = {}
    deviations: Dict[str, float] = {}
    repCount: int = 0

def calculate_angle(p1, p2, p3):
    """Calculate angle at p2 given three points"""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

def get_rating(score):
    if score >= 0.85: return 5
    elif score >= 0.70: return 4
    elif score >= 0.55: return 3
    elif score >= 0.40: return 2
    return 1

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "pose_checker", "version": "2.0.0"}

@app.post("/analyze", response_model=EnhancedFormCheckResponse)
async def analyze_form(
    file: UploadFile = File(...),
    exercise_type: str = Form(""),
    skill_level: str = Form("beginner")
):
    try:
        content = await file.read()
        if not content:
            raise HTTPException(400, "Empty file")
        
        # Try to use MediaPipe if available
        try:
            import mediapipe as mp
            mp_pose = mp.solutions.pose
            
            nparr = np.frombuffer(content, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise HTTPException(400, "Could not decode image")
            
            with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)
                
                if not results.pose_landmarks:
                    return EnhancedFormCheckResponse(
                        success=True,
                        rating=1,
                        formScore=0.0,
                        detectedExercise="unknown",
                        exerciseConfidence=0.0,
                        skillLevel=skill_level,
                        feedback="No pose detected. Ensure full body is visible.",
                        suggestions=["Step back from camera", "Ensure good lighting"]
                    )
                
                # Extract landmarks
                lm = results.pose_landmarks.landmark
                
                # Calculate key angles
                left_knee = calculate_angle(
                    [lm[23].x, lm[23].y], [lm[25].x, lm[25].y], [lm[27].x, lm[27].y]
                )
                right_knee = calculate_angle(
                    [lm[24].x, lm[24].y], [lm[26].x, lm[26].y], [lm[28].x, lm[28].y]
                )
                left_hip = calculate_angle(
                    [lm[11].x, lm[11].y], [lm[23].x, lm[23].y], [lm[25].x, lm[25].y]
                )
                left_elbow = calculate_angle(
                    [lm[11].x, lm[11].y], [lm[13].x, lm[13].y], [lm[15].x, lm[15].y]
                )
                
                user_angles = {
                    "left_knee": round(left_knee, 1),
                    "right_knee": round(right_knee, 1),
                    "left_hip": round(left_hip, 1),
                    "left_elbow": round(left_elbow, 1)
                }
                
                # Detect exercise
                knee_avg = (left_knee + right_knee) / 2
                if knee_avg < 130 and left_hip < 120:
                    detected = "barbell_squat"
                    confidence = 0.85
                elif left_elbow < 120 and left_hip > 150:
                    detected = "pushup"
                    confidence = 0.80
                elif abs(left_hip - 180) < 20:
                    detected = "plank"
                    confidence = 0.75
                else:
                    detected = "standing"
                    confidence = 0.60
                
                ex_type = exercise_type if exercise_type else detected
                
                # Get rules for exercise
                rules = EXERCISE_RULES.get("exercises", {}).get(ex_type, {})
                skill_rules = rules.get("skill_levels", {}).get(skill_level, {})
                tolerance = skill_rules.get("tolerance", 15)
                
                # Compare angles
                detailed = []
                total_score = 0
                count = 0
                expected_angles = {}
                deviations = {}
                
                if "knee_angle" in skill_rules:
                    ideal = skill_rules["knee_angle"]["ideal"]
                    expected_angles["left_knee"] = ideal
                    dev = left_knee - ideal
                    deviations["left_knee"] = round(dev, 1)
                    is_good = abs(dev) <= tolerance
                    total_score += 1.0 if is_good else 0.5
                    count += 1
                    detailed.append(DetailedFeedback(
                        joint="left_knee",
                        userAngle=round(left_knee, 1),
                        expectedAngle=ideal,
                        idealAngle=ideal,
                        deviation=round(dev, 1),
                        isGood=is_good,
                        feedback=f"Good!" if is_good else f"Knee at {left_knee:.0f}°, should be {ideal}°"
                    ))
                
                if "hip_angle" in skill_rules:
                    ideal = skill_rules["hip_angle"]["ideal"]
                    expected_angles["left_hip"] = ideal
                    dev = left_hip - ideal
                    deviations["left_hip"] = round(dev, 1)
                    is_good = abs(dev) <= tolerance
                    total_score += 1.0 if is_good else 0.5
                    count += 1
                    detailed.append(DetailedFeedback(
                        joint="left_hip",
                        userAngle=round(left_hip, 1),
                        expectedAngle=ideal,
                        idealAngle=ideal,
                        deviation=round(dev, 1),
                        isGood=is_good,
                        feedback=f"Good!" if is_good else f"Hip at {left_hip:.0f}°, should be {ideal}°"
                    ))
                
                form_score = total_score / max(count, 1)
                rating = get_rating(form_score)
                
                # Feedback message
                if rating >= 4:
                    feedback = "Excellent form! Keep it up."
                elif rating == 3:
                    feedback = "Good form with room for improvement."
                else:
                    feedback = "Form needs work. Focus on corrections."
                
                suggestions = rules.get("tips", ["Focus on controlled movement"])[:3]
                
                return EnhancedFormCheckResponse(
                    success=True,
                    rating=rating,
                    formScore=round(form_score, 2),
                    detectedExercise=rules.get("name", ex_type),
                    exerciseConfidence=round(confidence, 2),
                    skillLevel=skill_level,
                    feedback=feedback,
                    detailedFeedback=detailed,
                    suggestions=suggestions,
                    userAngles=user_angles,
                    expectedAngles=expected_angles,
                    deviations=deviations,
                    repCount=0
                )
                
        except ImportError:
            # MediaPipe not available, return mock response
            pass
        
        # Fallback mock response
        return EnhancedFormCheckResponse(
            success=True,
            rating=4,
            formScore=0.78,
            detectedExercise=exercise_type or "squat",
            exerciseConfidence=0.85,
            skillLevel=skill_level,
            feedback="Good form! (Install mediapipe for real analysis)",
            detailedFeedback=[
                DetailedFeedback(joint="left_knee", userAngle=92.0, expectedAngle=90.0, idealAngle=90.0, deviation=2.0, isGood=True, feedback="Good!")
            ],
            suggestions=["Keep core tight", "Drive through heels"],
            userAngles={"left_knee": 92.0, "left_hip": 78.0},
            expectedAngles={"left_knee": 90.0, "left_hip": 80.0},
            deviations={"left_knee": 2.0, "left_hip": -2.0},
            repCount=0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/supported-exercises")
async def get_exercises():
    exercises = EXERCISE_RULES.get("exercises", {})
    return {
        "exercises": [
            {"id": k, "name": v.get("name", k), "category": v.get("category", "other")}
            for k, v in exercises.items()
        ]
    }

@app.get("/skill-levels")
async def get_skill_levels():
    return {
        "levels": [
            {"id": "beginner", "name": "Beginner", "tolerance": "±20°"},
            {"id": "intermediate", "name": "Intermediate", "tolerance": "±10°"},
            {"id": "expert", "name": "Expert", "tolerance": "±5°"}
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
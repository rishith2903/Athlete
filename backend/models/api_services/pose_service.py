"""
FastAPI service for Exercise Form Checker
Provides REST API endpoints for pose analysis
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import sys
import os
import uvicorn
import numpy as np
from datetime import datetime
import cv2
import io
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(title="Pose Analysis Service", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FormCheckResponse(BaseModel):
    success: bool
    formScore: float
    feedback: str
    corrections: Dict[str, str]
    keypoints: Optional[List[Dict[str, float]]] = None
    repCount: Optional[int] = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "pose_checker"}

@app.post("/analyze", response_model=FormCheckResponse)
async def analyze_form(
    file: UploadFile = File(...),
    exercise_type: str = "squat"
):
    """Analyze exercise form from uploaded image or video"""
    try:
        # Read file content
        content = await file.read()
        
        # Basic validation
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Simulate form analysis
        form_score = np.random.uniform(0.65, 0.95)
        
        feedback_messages = {
            "squat": {
                "good": "Great squat form! Keep your back straight and chest up.",
                "corrections": {
                    "knees": "Keep knees aligned with toes",
                    "depth": "Try to go deeper, aim for thighs parallel to ground",
                    "back": "Maintain neutral spine throughout the movement"
                }
            },
            "push_up": {
                "good": "Good push-up form! Maintain body alignment.",
                "corrections": {
                    "elbows": "Keep elbows at 45-degree angle",
                    "core": "Engage core to prevent sagging",
                    "depth": "Lower chest closer to ground"
                }
            },
            "plank": {
                "good": "Excellent plank position! Hold steady.",
                "corrections": {
                    "hips": "Don't let hips sag or pike up",
                    "neck": "Keep neck neutral, look at ground",
                    "shoulders": "Stack shoulders over elbows"
                }
            }
        }
        
        exercise_feedback = feedback_messages.get(
            exercise_type.lower(),
            {"good": "Good form!", "corrections": {"general": "Focus on controlled movement"}}
        )
        
        # Select feedback based on score
        if form_score > 0.8:
            feedback = exercise_feedback["good"]
            corrections = {}
        else:
            feedback = "Form needs improvement. See corrections below."
            corrections = exercise_feedback["corrections"]
        
        # Simulate keypoints detection
        keypoints = [
            {"x": 0.5, "y": 0.3, "confidence": 0.95},  # Head
            {"x": 0.5, "y": 0.5, "confidence": 0.92},  # Torso
            {"x": 0.4, "y": 0.7, "confidence": 0.88},  # Left knee
            {"x": 0.6, "y": 0.7, "confidence": 0.89},  # Right knee
        ]
        
        return FormCheckResponse(
            success=True,
            formScore=round(form_score, 2),
            feedback=feedback,
            corrections=corrections,
            keypoints=keypoints,
            repCount=np.random.randint(5, 15)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/supported-exercises")
async def get_supported_exercises():
    """Get list of supported exercises"""
    return {
        "exercises": [
            {"name": "squat", "description": "Bodyweight or weighted squat"},
            {"name": "push_up", "description": "Standard push-up"},
            {"name": "plank", "description": "Forearm or high plank"},
            {"name": "lunge", "description": "Forward or reverse lunge"},
            {"name": "deadlift", "description": "Romanian or conventional deadlift"},
            {"name": "bicep_curl", "description": "Dumbbell or barbell curl"},
            {"name": "shoulder_press", "description": "Overhead press"},
            {"name": "jumping_jack", "description": "Cardio jumping jacks"}
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
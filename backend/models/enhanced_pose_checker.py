"""
Enhanced Pose Checker with Skill Levels and Detailed Feedback
Supports 100+ exercises with auto-detection, 5-point rating, and pose comparison
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import mediapipe as mp
import cv2

@dataclass
class AngleComparison:
    """Comparison between user angle and expected angle"""
    joint_name: str
    user_angle: float
    expected_angle: float
    ideal_angle: float
    deviation: float
    is_within_tolerance: bool
    feedback: str

@dataclass
class EnhancedPoseResult:
    """Complete pose analysis result"""
    rating: int  # 1-5 stars
    form_score: float  # 0.0 - 1.0
    detected_exercise: str
    exercise_confidence: float
    skill_level: str
    
    # Detailed feedback
    feedback: str  # Main message
    detailed_feedback: List[AngleComparison] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    # Pose comparison
    user_angles: Dict[str, float] = field(default_factory=dict)
    expected_angles: Dict[str, float] = field(default_factory=dict)
    deviations: Dict[str, float] = field(default_factory=dict)
    
    # Rep counting
    rep_count: int = 0

class EnhancedPoseChecker:
    """Enhanced pose checker with skill levels and detailed feedback"""
    
    def __init__(self, rules_path: Optional[str] = None):
        # Load exercise rules
        if rules_path is None:
            rules_path = Path(__file__).parent / "enhanced_exercise_rules.json"
        
        with open(rules_path, 'r') as f:
            self.rules = json.load(f)
        
        self.exercises = self.rules.get("exercises", {})
        self.rating_scale = self.rules.get("metadata", {}).get("rating_scale", {})
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Joint triplets for angle calculation
        self.joint_triplets = {
            'left_elbow': (11, 13, 15),
            'right_elbow': (12, 14, 16),
            'left_knee': (23, 25, 27),
            'right_knee': (24, 26, 28),
            'left_hip': (11, 23, 25),
            'right_hip': (12, 24, 26),
            'left_shoulder': (13, 11, 23),
            'right_shoulder': (14, 12, 24),
            'spine': (11, 23, 25),  # Approximation using left side
            'hip_hinge': (11, 23, 27),  # Shoulder to hip to ankle
        }
        
        # Rep counting state
        self.rep_state = {}
    
    def analyze(
        self,
        frame: np.ndarray,
        skill_level: str = "beginner",
        exercise_type: Optional[str] = None
    ) -> EnhancedPoseResult:
        """
        Analyze a single frame for exercise form
        
        Args:
            frame: BGR image frame
            skill_level: 'beginner', 'intermediate', or 'expert'
            exercise_type: Optional - if None, auto-detect
        
        Returns:
            EnhancedPoseResult with detailed analysis
        """
        # Detect pose
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_detector.process(rgb_frame)
        
        if not results.pose_landmarks:
            return EnhancedPoseResult(
                rating=1,
                form_score=0.0,
                detected_exercise="unknown",
                exercise_confidence=0.0,
                skill_level=skill_level,
                feedback="No pose detected. Please ensure your full body is visible.",
                suggestions=["Step back from camera", "Ensure good lighting"]
            )
        
        # Extract keypoints
        keypoints = self._extract_keypoints(results.pose_landmarks.landmark)
        
        # Calculate all joint angles
        user_angles = self._calculate_all_angles(keypoints)
        
        # Auto-detect exercise if not provided
        if exercise_type is None:
            exercise_type, confidence = self._detect_exercise(user_angles)
        else:
            exercise_type = exercise_type.lower().replace(" ", "_").replace("-", "_")
            confidence = 1.0
        
        # Get exercise rules
        exercise_rules = self.exercises.get(exercise_type)
        if exercise_rules is None:
            return EnhancedPoseResult(
                rating=3,
                form_score=0.5,
                detected_exercise=exercise_type,
                exercise_confidence=confidence,
                skill_level=skill_level,
                feedback=f"Exercise '{exercise_type}' not in database. Showing general analysis.",
                user_angles=user_angles
            )
        
        # Get skill level thresholds
        skill_rules = exercise_rules.get("skill_levels", {}).get(skill_level, {})
        if not skill_rules:
            skill_rules = exercise_rules.get("skill_levels", {}).get("beginner", {})
        
        # Compare angles and generate feedback
        comparisons, deviations, form_score = self._compare_angles(
            user_angles, skill_rules, exercise_rules
        )
        
        # Get expected angles
        expected_angles = {
            k: v.get("ideal", 90) for k, v in skill_rules.items() 
            if isinstance(v, dict) and "ideal" in v
        }
        
        # Generate rating (1-5 stars)
        rating = self._calculate_rating(form_score)
        
        # Generate main feedback and suggestions
        main_feedback, suggestions = self._generate_feedback(
            comparisons, exercise_rules, form_score, skill_level
        )
        
        # Update rep count
        rep_count = self._update_rep_count(exercise_type, user_angles)
        
        return EnhancedPoseResult(
            rating=rating,
            form_score=form_score,
            detected_exercise=exercise_rules.get("name", exercise_type),
            exercise_confidence=confidence,
            skill_level=skill_level,
            feedback=main_feedback,
            detailed_feedback=comparisons,
            suggestions=suggestions,
            user_angles=user_angles,
            expected_angles=expected_angles,
            deviations=deviations,
            rep_count=rep_count
        )
    
    def _extract_keypoints(self, landmarks) -> np.ndarray:
        """Extract keypoints from MediaPipe landmarks"""
        keypoints = np.zeros((33, 4))
        for i, landmark in enumerate(landmarks):
            keypoints[i] = [landmark.x, landmark.y, landmark.z, landmark.visibility]
        return keypoints
    
    def _calculate_all_angles(self, keypoints: np.ndarray) -> Dict[str, float]:
        """Calculate all joint angles"""
        angles = {}
        for joint_name, (p1_idx, p2_idx, p3_idx) in self.joint_triplets.items():
            angles[joint_name] = self._calculate_angle(
                keypoints[p1_idx, :3],
                keypoints[p2_idx, :3],
                keypoints[p3_idx, :3]
            )
        return angles
    
    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle between three points (in degrees)"""
        v1 = p1 - p2
        v2 = p3 - p2
        
        cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        
        return float(angle)
    
    def _detect_exercise(self, angles: Dict[str, float]) -> Tuple[str, float]:
        """Auto-detect exercise based on joint angles"""
        best_match = "unknown"
        best_score = 0.0
        
        # Simple rule-based detection
        knee_angle = (angles.get('left_knee', 180) + angles.get('right_knee', 180)) / 2
        hip_angle = (angles.get('left_hip', 180) + angles.get('right_hip', 180)) / 2
        elbow_angle = (angles.get('left_elbow', 180) + angles.get('right_elbow', 180)) / 2
        
        # Squat detection
        if knee_angle < 130 and hip_angle < 120:
            best_match = "barbell_squat"
            best_score = 0.85
        # Plank detection
        elif abs(hip_angle - 180) < 20 and elbow_angle < 100:
            best_match = "plank"
            best_score = 0.80
        # Push-up detection
        elif elbow_angle < 120 and hip_angle > 150:
            best_match = "pushup"
            best_score = 0.80
        # Standing (deadlift top, curl, etc.)
        elif knee_angle > 160 and hip_angle > 150:
            if elbow_angle < 90:
                best_match = "bicep_curl"
                best_score = 0.75
            else:
                best_match = "standing"
                best_score = 0.60
        
        return best_match, best_score
    
    def _compare_angles(
        self,
        user_angles: Dict[str, float],
        skill_rules: Dict,
        exercise_rules: Dict
    ) -> Tuple[List[AngleComparison], Dict[str, float], float]:
        """Compare user angles to expected angles and generate feedback"""
        comparisons = []
        deviations = {}
        total_score = 0.0
        num_checks = 0
        
        tolerance = skill_rules.get("tolerance", 15)
        
        for angle_name, rules in skill_rules.items():
            if not isinstance(rules, dict) or "ideal" not in rules:
                continue
            
            # Map rule name to joint name
            joint_name = self._map_rule_to_joint(angle_name)
            if joint_name not in user_angles:
                continue
            
            user_val = user_angles[joint_name]
            ideal = rules["ideal"]
            min_val = rules.get("min", ideal - 30)
            max_val = rules.get("max", ideal + 30)
            
            deviation = user_val - ideal
            within_tolerance = min_val <= user_val <= max_val
            
            # Calculate score for this joint
            if within_tolerance:
                joint_score = 1.0 - (abs(deviation) / (tolerance * 2))
            else:
                joint_score = max(0, 0.5 - abs(deviation) / 60)
            
            total_score += joint_score
            num_checks += 1
            deviations[joint_name] = deviation
            
            # Generate feedback
            if within_tolerance:
                feedback = f"Good! {joint_name.replace('_', ' ').title()} at {user_val:.0f}° (ideal: {ideal}°)"
            else:
                direction = "increase" if user_val < ideal else "decrease"
                feedback = f"{joint_name.replace('_', ' ').title()} at {user_val:.0f}°, should be {ideal}° ({direction} by {abs(deviation):.0f}°)"
            
            comparisons.append(AngleComparison(
                joint_name=joint_name,
                user_angle=user_val,
                expected_angle=(min_val + max_val) / 2,
                ideal_angle=ideal,
                deviation=deviation,
                is_within_tolerance=within_tolerance,
                feedback=feedback
            ))
        
        form_score = total_score / max(num_checks, 1)
        return comparisons, deviations, form_score
    
    def _map_rule_to_joint(self, rule_name: str) -> str:
        """Map exercise rule name to joint name"""
        mappings = {
            'knee_angle': 'left_knee',
            'hip_angle': 'left_hip',
            'back_angle': 'spine',
            'elbow_angle_down': 'left_elbow',
            'elbow_angle_top': 'left_elbow',
            'elbow_angle_bottom': 'left_elbow',
            'body_alignment': 'spine',
            'hip_hinge': 'hip_hinge',
            'front_knee': 'left_knee',
            'back_knee': 'right_knee',
            'torso': 'spine',
            'spine_neutral': 'spine',
            'spine_alignment': 'spine',
            'wrist_alignment': 'left_elbow',  # Approximation
            'shoulder_movement': 'left_shoulder',
            'torso_lean': 'spine',
        }
        return mappings.get(rule_name, rule_name)
    
    def _calculate_rating(self, form_score: float) -> int:
        """Convert form score (0-1) to 1-5 star rating"""
        score_pct = form_score * 100
        
        if score_pct >= 85:
            return 5
        elif score_pct >= 70:
            return 4
        elif score_pct >= 55:
            return 3
        elif score_pct >= 40:
            return 2
        else:
            return 1
    
    def _generate_feedback(
        self,
        comparisons: List[AngleComparison],
        exercise_rules: Dict,
        form_score: float,
        skill_level: str
    ) -> Tuple[str, List[str]]:
        """Generate main feedback message and suggestions"""
        # Main feedback based on score
        rating = self._calculate_rating(form_score)
        
        if rating >= 4:
            main_feedback = f"Excellent form! Keep up the great work."
        elif rating == 3:
            main_feedback = "Good form with room for improvement."
        else:
            main_feedback = "Form needs work. Focus on the corrections below."
        
        # Add specific issues
        issues = [c for c in comparisons if not c.is_within_tolerance]
        if issues:
            worst = max(issues, key=lambda x: abs(x.deviation))
            main_feedback += f" Primary issue: {worst.feedback}"
        
        # Suggestions from exercise tips
        suggestions = exercise_rules.get("tips", [])[:3]
        
        # Add skill-level specific suggestion
        if skill_level == "beginner":
            suggestions.append("Focus on form before adding weight.")
        elif skill_level == "expert":
            suggestions.append("Minor adjustments can still improve performance.")
        
        return main_feedback, suggestions
    
    def _update_rep_count(self, exercise_type: str, angles: Dict[str, float]) -> int:
        """Simple rep counter based on angle thresholds"""
        if exercise_type not in self.rep_state:
            self.rep_state[exercise_type] = {"count": 0, "phase": "up"}
        
        state = self.rep_state[exercise_type]
        knee_angle = (angles.get('left_knee', 180) + angles.get('right_knee', 180)) / 2
        
        # Simple squat rep detection
        if "squat" in exercise_type.lower():
            if state["phase"] == "up" and knee_angle < 100:
                state["phase"] = "down"
            elif state["phase"] == "down" and knee_angle > 150:
                state["phase"] = "up"
                state["count"] += 1
        
        return state["count"]
    
    def reset_rep_count(self, exercise_type: Optional[str] = None):
        """Reset rep count for an exercise or all exercises"""
        if exercise_type:
            self.rep_state.pop(exercise_type, None)
        else:
            self.rep_state.clear()
    
    def get_supported_exercises(self) -> List[Dict]:
        """Get list of supported exercises"""
        return [
            {
                "id": key,
                "name": val.get("name", key),
                "category": val.get("category", "other"),
                "primary_muscles": val.get("primary_muscles", [])
            }
            for key, val in self.exercises.items()
        ]

# Convenience function for API usage
def analyze_pose(
    frame: np.ndarray,
    skill_level: str = "beginner",
    exercise_type: Optional[str] = None
) -> Dict:
    """
    Analyze pose and return dictionary result
    
    Args:
        frame: BGR image frame
        skill_level: 'beginner', 'intermediate', or 'expert'
        exercise_type: Optional exercise type
    
    Returns:
        Dictionary with analysis results
    """
    checker = EnhancedPoseChecker()
    result = checker.analyze(frame, skill_level, exercise_type)
    
    return {
        "success": True,
        "rating": result.rating,
        "formScore": result.form_score,
        "detectedExercise": result.detected_exercise,
        "exerciseConfidence": result.exercise_confidence,
        "skillLevel": result.skill_level,
        "feedback": result.feedback,
        "detailedFeedback": [
            {
                "joint": c.joint_name,
                "userAngle": round(c.user_angle, 1),
                "expectedAngle": round(c.expected_angle, 1),
                "idealAngle": round(c.ideal_angle, 1),
                "deviation": round(c.deviation, 1),
                "isGood": c.is_within_tolerance,
                "feedback": c.feedback
            }
            for c in result.detailed_feedback
        ],
        "suggestions": result.suggestions,
        "userAngles": {k: round(v, 1) for k, v in result.user_angles.items()},
        "expectedAngles": {k: round(v, 1) for k, v in result.expected_angles.items()},
        "deviations": {k: round(v, 1) for k, v in result.deviations.items()},
        "repCount": result.rep_count
    }

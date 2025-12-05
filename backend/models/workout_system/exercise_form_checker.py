"""
Advanced Exercise Form Checker with Real-time Pose Estimation
Computer vision system for analyzing and correcting exercise form using deep learning
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import deque
import json
import math
from scipy import signal
from scipy.spatial import distance
import tensorflow as tf
import threading
from queue import Queue

# Try to import TensorFlow Lite for mobile deployment
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False

class ExerciseType(Enum):
    SQUAT = "squat"
    PUSH_UP = "push_up"
    LUNGE = "lunge"
    BICEP_CURL = "bicep_curl"
    PLANK = "plank"
    DEADLIFT = "deadlift"
    SHOULDER_PRESS = "shoulder_press"
    JUMPING_JACK = "jumping_jack"

class FormStatus(Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    WARNING = "warning"
    NOT_DETECTED = "not_detected"

@dataclass
class Keypoint:
    """Represents a single body keypoint"""
    x: float
    y: float
    z: float = 0.0
    confidence: float = 1.0
    name: str = ""

@dataclass
class Pose:
    """Complete pose representation"""
    keypoints: Dict[str, Keypoint]
    timestamp: float
    frame_id: int
    
@dataclass
class ExerciseMetrics:
    """Metrics for exercise analysis"""
    angle_metrics: Dict[str, float]
    position_metrics: Dict[str, float]
    velocity_metrics: Dict[str, float]
    symmetry_score: float
    stability_score: float
    range_of_motion: float
    tempo: float

@dataclass
class FormFeedback:
    """Structured feedback for exercise form"""
    exercise: str
    status: str
    feedback: List[str]
    corrections: List[str]
    score: float
    rep_count: int
    metrics: Optional[ExerciseMetrics] = None

class JointAngleCalculator:
    """Calculates angles between body joints"""
    
    @staticmethod
    def calculate_angle(p1: Keypoint, p2: Keypoint, p3: Keypoint) -> float:
        """Calculate angle at p2 formed by p1-p2-p3"""
        # Vector from p2 to p1
        v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
        # Vector from p2 to p3
        v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
        
        # Calculate angle using dot product
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    @staticmethod
    def calculate_2d_angle(p1: Tuple[float, float], 
                          p2: Tuple[float, float], 
                          p3: Tuple[float, float]) -> float:
        """Calculate 2D angle for simplified calculations"""
        angle = np.degrees(
            np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) -
            np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
        )
        return abs(angle)

class PoseEstimator:
    """Main pose estimation engine using MediaPipe and MoveNet"""
    
    def __init__(self, model_type: str = "mediapipe"):
        self.model_type = model_type
        
        if model_type == "mediapipe":
            # Initialize MediaPipe Pose
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=1,  # 0, 1, or 2
                smooth_landmarks=True,
                enable_segmentation=False
            )
            
            # MediaPipe landmark names
            self.landmark_names = {
                0: 'nose', 1: 'left_eye_inner', 2: 'left_eye', 3: 'left_eye_outer',
                4: 'right_eye_inner', 5: 'right_eye', 6: 'right_eye_outer',
                7: 'left_ear', 8: 'right_ear', 9: 'mouth_left', 10: 'mouth_right',
                11: 'left_shoulder', 12: 'right_shoulder', 13: 'left_elbow',
                14: 'right_elbow', 15: 'left_wrist', 16: 'right_wrist',
                17: 'left_pinky', 18: 'right_pinky', 19: 'left_index',
                20: 'right_index', 21: 'left_thumb', 22: 'right_thumb',
                23: 'left_hip', 24: 'right_hip', 25: 'left_knee',
                26: 'right_knee', 27: 'left_ankle', 28: 'right_ankle',
                29: 'left_heel', 30: 'right_heel', 31: 'left_foot_index',
                32: 'right_foot_index'
            }
            
        elif model_type == "movenet":
            # Load MoveNet model
            self.interpreter = self._load_movenet_model()
            self.input_size = 192
            
    def _load_movenet_model(self):
        """Load MoveNet TFLite model"""
        model_path = "models/movenet_thunder.tflite"
        if TFLITE_AVAILABLE:
            interpreter = tflite.Interpreter(model_path=model_path)
        else:
            interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    
    def detect_pose(self, frame: np.ndarray, frame_id: int = 0) -> Optional[Pose]:
        """Detect pose from a single frame"""
        if self.model_type == "mediapipe":
            return self._detect_mediapipe(frame, frame_id)
        elif self.model_type == "movenet":
            return self._detect_movenet(frame, frame_id)
    
    def _detect_mediapipe(self, frame: np.ndarray, frame_id: int) -> Optional[Pose]:
        """Detect pose using MediaPipe"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None
        
        # Extract keypoints
        keypoints = {}
        h, w = frame.shape[:2]
        
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            name = self.landmark_names.get(idx, f"point_{idx}")
            keypoints[name] = Keypoint(
                x=landmark.x * w,
                y=landmark.y * h,
                z=landmark.z * w,  # Z is in same scale as X
                confidence=landmark.visibility,
                name=name
            )
        
        return Pose(
            keypoints=keypoints,
            timestamp=time.time(),
            frame_id=frame_id
        )
    
    def _detect_movenet(self, frame: np.ndarray, frame_id: int) -> Optional[Pose]:
        """Detect pose using MoveNet"""
        # Preprocess frame
        input_image = cv2.resize(frame, (self.input_size, self.input_size))
        input_image = tf.cast(input_image, dtype=tf.float32)
        input_image = tf.expand_dims(input_image, axis=0)
        
        # Run inference
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        
        self.interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
        self.interpreter.invoke()
        
        # Get output
        keypoints_with_scores = self.interpreter.get_tensor(output_details[0]['index'])
        
        # Parse keypoints
        keypoints = {}
        h, w = frame.shape[:2]
        
        movenet_keypoints = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        for i, name in enumerate(movenet_keypoints):
            y, x, score = keypoints_with_scores[0, 0, i]
            keypoints[name] = Keypoint(
                x=x * w,
                y=y * h,
                z=0,
                confidence=score,
                name=name
            )
        
        return Pose(
            keypoints=keypoints,
            timestamp=time.time(),
            frame_id=frame_id
        )
    
    def draw_pose(self, frame: np.ndarray, pose: Pose) -> np.ndarray:
        """Draw pose landmarks on frame"""
        if self.model_type == "mediapipe" and hasattr(self, 'mp_drawing'):
            # Use MediaPipe drawing utilities
            h, w = frame.shape[:2]
            
            # Convert back to MediaPipe format for drawing
            landmarks = []
            for name in self.landmark_names.values():
                if name in pose.keypoints:
                    kp = pose.keypoints[name]
                    landmark = type('obj', (object,), {
                        'x': kp.x / w,
                        'y': kp.y / h,
                        'z': kp.z / w,
                        'visibility': kp.confidence
                    })()
                    landmarks.append(landmark)
            
            if landmarks:
                pose_landmarks = type('obj', (object,), {'landmark': landmarks})()
                self.mp_drawing.draw_landmarks(
                    frame, pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                )
        else:
            # Custom drawing
            for keypoint in pose.keypoints.values():
                if keypoint.confidence > 0.5:
                    cv2.circle(frame, 
                             (int(keypoint.x), int(keypoint.y)), 
                             5, (0, 255, 0), -1)
        
        return frame

class ExerciseAnalyzer:
    """Analyzes specific exercises and provides form feedback"""
    
    def __init__(self):
        self.angle_calculator = JointAngleCalculator()
        self.exercise_rules = self._initialize_exercise_rules()
        self.rep_counter = RepetitionCounter()
        self.pose_buffer = deque(maxlen=30)  # Store last 30 poses (1 second at 30fps)
        
    def _initialize_exercise_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize rules and thresholds for each exercise"""
        return {
            ExerciseType.SQUAT: {
                'key_angles': {
                    'knee': {'min': 70, 'max': 100, 'optimal': 90},
                    'hip': {'min': 70, 'max': 100, 'optimal': 90},
                    'ankle': {'min': 60, 'max': 90, 'optimal': 75},
                    'back': {'min': 160, 'max': 180, 'optimal': 170}
                },
                'alignment_rules': {
                    'knee_over_toe': True,
                    'back_straight': True,
                    'chest_up': True,
                    'weight_on_heels': True
                },
                'common_errors': {
                    'knee_cave': "Keep knees aligned with toes, don't let them cave inward",
                    'butt_wink': "Maintain neutral spine at bottom of squat",
                    'forward_lean': "Keep chest up and weight balanced",
                    'shallow_depth': "Go deeper - aim for thighs parallel to ground"
                }
            },
            ExerciseType.PUSH_UP: {
                'key_angles': {
                    'elbow': {'min': 70, 'max': 110, 'optimal': 90},
                    'shoulder': {'min': 30, 'max': 60, 'optimal': 45},
                    'body_line': {'min': 170, 'max': 180, 'optimal': 175}
                },
                'alignment_rules': {
                    'straight_body': True,
                    'elbow_position': True,
                    'full_range': True,
                    'controlled_tempo': True
                },
                'common_errors': {
                    'sagging_hips': "Keep hips in line with body - engage core",
                    'flared_elbows': "Keep elbows at 45° angle from body",
                    'partial_range': "Go all the way down until chest nearly touches ground",
                    'neck_position': "Keep neck neutral, don't look up"
                }
            },
            ExerciseType.LUNGE: {
                'key_angles': {
                    'front_knee': {'min': 80, 'max': 100, 'optimal': 90},
                    'back_knee': {'min': 80, 'max': 100, 'optimal': 90},
                    'torso': {'min': 170, 'max': 180, 'optimal': 175}
                },
                'alignment_rules': {
                    'knee_over_ankle': True,
                    'upright_torso': True,
                    'hip_stability': True,
                    'controlled_descent': True
                },
                'common_errors': {
                    'knee_past_toe': "Don't let front knee go past toes",
                    'leaning_forward': "Keep torso upright throughout movement",
                    'narrow_stance': "Maintain hip-width stance for stability",
                    'wobbling': "Engage core for better balance"
                }
            },
            ExerciseType.BICEP_CURL: {
                'key_angles': {
                    'elbow': {'min': 30, 'max': 140, 'optimal': 135},
                    'shoulder': {'min': -10, 'max': 10, 'optimal': 0},
                    'wrist': {'min': 170, 'max': 180, 'optimal': 175}
                },
                'alignment_rules': {
                    'stable_shoulder': True,
                    'controlled_motion': True,
                    'full_range': True,
                    'no_swing': True
                },
                'common_errors': {
                    'swinging': "Don't use momentum - control the weight",
                    'elbow_drift': "Keep elbows at your sides",
                    'partial_range': "Fully extend arms at bottom",
                    'wrist_bend': "Keep wrists straight and neutral"
                }
            },
            ExerciseType.PLANK: {
                'key_angles': {
                    'body_line': {'min': 170, 'max': 180, 'optimal': 175},
                    'elbow': {'min': 85, 'max': 95, 'optimal': 90},
                    'neck': {'min': 160, 'max': 180, 'optimal': 170}
                },
                'alignment_rules': {
                    'straight_line': True,
                    'neutral_spine': True,
                    'engaged_core': True,
                    'proper_breathing': True
                },
                'common_errors': {
                    'sagging_hips': "Lift hips to align with shoulders",
                    'raised_hips': "Lower hips to create straight line",
                    'forward_head': "Keep head in line with spine",
                    'holding_breath': "Breathe normally throughout hold"
                }
            }
        }
    
    def analyze_exercise(self, pose: Pose, exercise_type: ExerciseType) -> FormFeedback:
        """Analyze exercise form and provide feedback"""
        
        # Add pose to buffer for temporal analysis
        self.pose_buffer.append(pose)
        
        # Calculate metrics
        metrics = self._calculate_metrics(pose, exercise_type)
        
        # Check form against rules
        form_errors = self._check_form_rules(metrics, exercise_type)
        
        # Determine status
        if len(form_errors) == 0:
            status = FormStatus.CORRECT
        elif len(form_errors) <= 2:
            status = FormStatus.WARNING
        else:
            status = FormStatus.INCORRECT
        
        # Generate feedback
        feedback = self._generate_feedback(form_errors, exercise_type)
        corrections = self._generate_corrections(form_errors, exercise_type)
        
        # Calculate form score
        score = self._calculate_form_score(metrics, exercise_type)
        
        # Count repetitions
        rep_count = self.rep_counter.count_reps(metrics, exercise_type)
        
        return FormFeedback(
            exercise=exercise_type.value,
            status=status.value,
            feedback=feedback,
            corrections=corrections,
            score=score,
            rep_count=rep_count,
            metrics=metrics
        )
    
    def _calculate_metrics(self, pose: Pose, exercise_type: ExerciseType) -> ExerciseMetrics:
        """Calculate all relevant metrics for the exercise"""
        
        angle_metrics = {}
        position_metrics = {}
        velocity_metrics = {}
        
        if exercise_type == ExerciseType.SQUAT:
            # Calculate knee angle
            if all(k in pose.keypoints for k in ['left_hip', 'left_knee', 'left_ankle']):
                angle_metrics['left_knee'] = self.angle_calculator.calculate_angle(
                    pose.keypoints['left_hip'],
                    pose.keypoints['left_knee'],
                    pose.keypoints['left_ankle']
                )
            
            if all(k in pose.keypoints for k in ['right_hip', 'right_knee', 'right_ankle']):
                angle_metrics['right_knee'] = self.angle_calculator.calculate_angle(
                    pose.keypoints['right_hip'],
                    pose.keypoints['right_knee'],
                    pose.keypoints['right_ankle']
                )
            
            # Calculate hip angle
            if all(k in pose.keypoints for k in ['left_shoulder', 'left_hip', 'left_knee']):
                angle_metrics['left_hip'] = self.angle_calculator.calculate_angle(
                    pose.keypoints['left_shoulder'],
                    pose.keypoints['left_hip'],
                    pose.keypoints['left_knee']
                )
            
            # Check knee alignment
            if 'left_knee' in pose.keypoints and 'left_ankle' in pose.keypoints:
                position_metrics['knee_over_toe'] = (
                    pose.keypoints['left_knee'].x - pose.keypoints['left_ankle'].x
                )
            
        elif exercise_type == ExerciseType.PUSH_UP:
            # Calculate elbow angle
            if all(k in pose.keypoints for k in ['left_shoulder', 'left_elbow', 'left_wrist']):
                angle_metrics['left_elbow'] = self.angle_calculator.calculate_angle(
                    pose.keypoints['left_shoulder'],
                    pose.keypoints['left_elbow'],
                    pose.keypoints['left_wrist']
                )
            
            # Check body alignment
            if all(k in pose.keypoints for k in ['left_shoulder', 'left_hip', 'left_ankle']):
                angle_metrics['body_line'] = self.angle_calculator.calculate_angle(
                    pose.keypoints['left_shoulder'],
                    pose.keypoints['left_hip'],
                    pose.keypoints['left_ankle']
                )
        
        elif exercise_type == ExerciseType.PLANK:
            # Check body alignment for plank
            if all(k in pose.keypoints for k in ['left_shoulder', 'left_hip', 'left_ankle']):
                angle_metrics['body_line'] = self.angle_calculator.calculate_angle(
                    pose.keypoints['left_shoulder'],
                    pose.keypoints['left_hip'],
                    pose.keypoints['left_ankle']
                )
            
            # Check hip height relative to shoulders
            if 'left_shoulder' in pose.keypoints and 'left_hip' in pose.keypoints:
                position_metrics['hip_height'] = (
                    pose.keypoints['left_hip'].y - pose.keypoints['left_shoulder'].y
                )
        
        # Calculate velocity if we have enough poses in buffer
        if len(self.pose_buffer) >= 3:
            velocity_metrics = self._calculate_velocity_metrics()
        
        # Calculate symmetry score
        symmetry_score = self._calculate_symmetry(angle_metrics)
        
        # Calculate stability score
        stability_score = self._calculate_stability()
        
        # Calculate range of motion
        range_of_motion = self._calculate_range_of_motion(angle_metrics, exercise_type)
        
        # Calculate tempo
        tempo = self._calculate_tempo(velocity_metrics)
        
        return ExerciseMetrics(
            angle_metrics=angle_metrics,
            position_metrics=position_metrics,
            velocity_metrics=velocity_metrics,
            symmetry_score=symmetry_score,
            stability_score=stability_score,
            range_of_motion=range_of_motion,
            tempo=tempo
        )
    
    def _check_form_rules(self, metrics: ExerciseMetrics, 
                         exercise_type: ExerciseType) -> List[str]:
        """Check form against exercise-specific rules"""
        
        errors = []
        rules = self.exercise_rules[exercise_type]
        
        # Check angle constraints
        for angle_name, constraints in rules['key_angles'].items():
            if angle_name in metrics.angle_metrics:
                angle = metrics.angle_metrics[angle_name]
                if angle < constraints['min']:
                    errors.append(f"{angle_name}_too_small")
                elif angle > constraints['max']:
                    errors.append(f"{angle_name}_too_large")
        
        # Check position metrics
        if exercise_type == ExerciseType.SQUAT:
            if 'knee_over_toe' in metrics.position_metrics:
                if metrics.position_metrics['knee_over_toe'] > 50:  # pixels
                    errors.append('knee_past_toe')
        
        elif exercise_type == ExerciseType.PLANK:
            if 'hip_height' in metrics.position_metrics:
                if abs(metrics.position_metrics['hip_height']) > 30:  # pixels
                    if metrics.position_metrics['hip_height'] > 0:
                        errors.append('sagging_hips')
                    else:
                        errors.append('raised_hips')
        
        # Check symmetry
        if metrics.symmetry_score < 0.8:
            errors.append('asymmetric_form')
        
        # Check stability
        if metrics.stability_score < 0.7:
            errors.append('unstable_form')
        
        # Check range of motion
        if metrics.range_of_motion < 0.7:
            errors.append('limited_range')
        
        return errors
    
    def _generate_feedback(self, errors: List[str], 
                          exercise_type: ExerciseType) -> List[str]:
        """Generate human-readable feedback"""
        
        feedback = []
        
        if not errors:
            feedback.append("Great form! Keep it up!")
            return feedback
        
        error_messages = self.exercise_rules[exercise_type]['common_errors']
        
        for error in errors:
            if error in error_messages:
                feedback.append(error_messages[error])
            elif 'too_small' in error:
                angle = error.replace('_too_small', '')
                feedback.append(f"Increase your {angle.replace('_', ' ')} angle")
            elif 'too_large' in error:
                angle = error.replace('_too_large', '')
                feedback.append(f"Decrease your {angle.replace('_', ' ')} angle")
            elif error == 'asymmetric_form':
                feedback.append("Try to maintain symmetry between both sides")
            elif error == 'unstable_form':
                feedback.append("Focus on stability - engage your core")
            elif error == 'limited_range':
                feedback.append("Use full range of motion for best results")
        
        return feedback
    
    def _generate_corrections(self, errors: List[str], 
                            exercise_type: ExerciseType) -> List[str]:
        """Generate specific corrections"""
        
        corrections = []
        
        for error in errors[:3]:  # Limit to top 3 corrections
            if 'knee' in error:
                corrections.append("Adjust knee position")
            elif 'hip' in error:
                corrections.append("Check hip alignment")
            elif 'elbow' in error:
                corrections.append("Fix elbow angle")
            elif 'body_line' in error:
                corrections.append("Straighten body line")
        
        return corrections
    
    def _calculate_form_score(self, metrics: ExerciseMetrics, 
                             exercise_type: ExerciseType) -> float:
        """Calculate overall form score (0-100)"""
        
        score = 100.0
        rules = self.exercise_rules[exercise_type]
        
        # Deduct points for angle deviations
        for angle_name, constraints in rules['key_angles'].items():
            if angle_name in metrics.angle_metrics:
                angle = metrics.angle_metrics[angle_name]
                optimal = constraints['optimal']
                deviation = abs(angle - optimal) / optimal
                score -= min(20, deviation * 20)
        
        # Factor in symmetry
        score *= metrics.symmetry_score
        
        # Factor in stability
        score *= metrics.stability_score
        
        # Factor in range of motion
        score *= metrics.range_of_motion
        
        return max(0, min(100, score))
    
    def _calculate_symmetry(self, angle_metrics: Dict[str, float]) -> float:
        """Calculate symmetry between left and right sides"""
        
        symmetry_scores = []
        
        # Compare left and right angles
        for left_key in angle_metrics:
            if 'left' in left_key:
                right_key = left_key.replace('left', 'right')
                if right_key in angle_metrics:
                    diff = abs(angle_metrics[left_key] - angle_metrics[right_key])
                    symmetry = 1.0 - (diff / 180.0)  # Normalize to 0-1
                    symmetry_scores.append(symmetry)
        
        if symmetry_scores:
            return np.mean(symmetry_scores)
        return 1.0
    
    def _calculate_stability(self) -> float:
        """Calculate stability based on pose buffer"""
        
        if len(self.pose_buffer) < 5:
            return 1.0
        
        # Calculate variance in key points over time
        variances = []
        
        for keypoint_name in ['left_hip', 'right_hip', 'left_shoulder', 'right_shoulder']:
            positions = []
            for pose in self.pose_buffer:
                if keypoint_name in pose.keypoints:
                    kp = pose.keypoints[keypoint_name]
                    positions.append([kp.x, kp.y])
            
            if len(positions) > 1:
                positions = np.array(positions)
                variance = np.var(positions, axis=0).sum()
                variances.append(variance)
        
        if variances:
            avg_variance = np.mean(variances)
            # Convert variance to stability score (lower variance = higher stability)
            stability = 1.0 / (1.0 + avg_variance / 100.0)
            return stability
        
        return 1.0
    
    def _calculate_range_of_motion(self, angle_metrics: Dict[str, float], 
                                  exercise_type: ExerciseType) -> float:
        """Calculate range of motion score"""
        
        if not angle_metrics:
            return 1.0
        
        rules = self.exercise_rules[exercise_type]
        rom_scores = []
        
        for angle_name in angle_metrics:
            if angle_name in rules['key_angles']:
                constraints = rules['key_angles'][angle_name]
                angle = angle_metrics[angle_name]
                expected_range = constraints['max'] - constraints['min']
                
                # Check if angle is within expected range
                if constraints['min'] <= angle <= constraints['max']:
                    rom_scores.append(1.0)
                else:
                    # Calculate how far outside the range
                    if angle < constraints['min']:
                        deviation = (constraints['min'] - angle) / expected_range
                    else:
                        deviation = (angle - constraints['max']) / expected_range
                    rom_scores.append(max(0, 1.0 - deviation))
        
        if rom_scores:
            return np.mean(rom_scores)
        return 1.0
    
    def _calculate_tempo(self, velocity_metrics: Dict[str, float]) -> float:
        """Calculate exercise tempo"""
        
        if not velocity_metrics:
            return 2.0  # Default 2 seconds per rep
        
        # Implement tempo calculation based on velocity
        return 2.0
    
    def _calculate_velocity_metrics(self) -> Dict[str, float]:
        """Calculate velocity metrics from pose buffer"""
        
        velocities = {}
        
        if len(self.pose_buffer) < 2:
            return velocities
        
        # Calculate velocity for key points
        for keypoint_name in ['left_hip', 'left_knee', 'left_wrist']:
            if all(keypoint_name in p.keypoints for p in self.pose_buffer[-2:]):
                p1 = self.pose_buffer[-2].keypoints[keypoint_name]
                p2 = self.pose_buffer[-1].keypoints[keypoint_name]
                dt = self.pose_buffer[-1].timestamp - self.pose_buffer[-2].timestamp
                
                if dt > 0:
                    velocity = np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2) / dt
                    velocities[keypoint_name] = velocity
        
        return velocities

class RepetitionCounter:
    """Counts exercise repetitions"""
    
    def __init__(self):
        self.rep_count = 0
        self.position_history = []
        self.last_peak_time = 0
        self.min_rep_duration = 0.5  # seconds
        
    def count_reps(self, metrics: ExerciseMetrics, 
                   exercise_type: ExerciseType) -> int:
        """Count repetitions based on metrics"""
        
        # Get primary angle for the exercise
        primary_angle = self._get_primary_angle(metrics, exercise_type)
        
        if primary_angle is not None:
            self.position_history.append(primary_angle)
            
            # Keep only recent history (last 3 seconds)
            if len(self.position_history) > 90:  # 30 fps * 3 seconds
                self.position_history = self.position_history[-90:]
            
            # Detect peaks in position history
            if len(self.position_history) > 10:
                peaks = self._detect_peaks()
                
                # Count new peaks
                current_time = time.time()
                for peak_idx in peaks:
                    peak_time = current_time - (len(self.position_history) - peak_idx) / 30.0
                    if peak_time - self.last_peak_time > self.min_rep_duration:
                        self.rep_count += 1
                        self.last_peak_time = peak_time
        
        return self.rep_count
    
    def _get_primary_angle(self, metrics: ExerciseMetrics, 
                          exercise_type: ExerciseType) -> Optional[float]:
        """Get the primary angle to track for rep counting"""
        
        if exercise_type == ExerciseType.SQUAT:
            return metrics.angle_metrics.get('left_knee')
        elif exercise_type == ExerciseType.PUSH_UP:
            return metrics.angle_metrics.get('left_elbow')
        elif exercise_type == ExerciseType.BICEP_CURL:
            return metrics.angle_metrics.get('left_elbow')
        elif exercise_type == ExerciseType.LUNGE:
            return metrics.angle_metrics.get('front_knee')
        
        return None
    
    def _detect_peaks(self) -> List[int]:
        """Detect peaks in position history"""
        
        if len(self.position_history) < 3:
            return []
        
        # Smooth the signal
        smoothed = signal.savgol_filter(self.position_history, 
                                       window_length=min(11, len(self.position_history) // 2 * 2 + 1), 
                                       polyorder=2)
        
        # Find peaks
        peaks, _ = signal.find_peaks(smoothed, 
                                    distance=15,  # Minimum frames between peaks
                                    prominence=10)  # Minimum prominence
        
        return peaks.tolist()
    
    def reset(self):
        """Reset rep counter"""
        self.rep_count = 0
        self.position_history = []
        self.last_peak_time = 0

class RealTimeFormChecker:
    """Main system for real-time exercise form checking"""
    
    def __init__(self, model_type: str = "mediapipe"):
        self.pose_estimator = PoseEstimator(model_type)
        self.exercise_analyzer = ExerciseAnalyzer()
        self.current_exercise = None
        self.is_running = False
        self.frame_queue = Queue(maxsize=10)
        self.result_queue = Queue(maxsize=10)
        self.processing_thread = None
        
        # Performance metrics
        self.fps = 30
        self.processing_time = 0
        self.frame_count = 0
        
    def start(self, exercise_type: ExerciseType):
        """Start the form checker"""
        self.current_exercise = exercise_type
        self.is_running = True
        self.exercise_analyzer.rep_counter.reset()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_frames)
        self.processing_thread.start()
    
    def stop(self):
        """Stop the form checker"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()
    
    def process_frame(self, frame: np.ndarray) -> Optional[FormFeedback]:
        """Process a single frame and return feedback"""
        
        if not self.is_running or self.current_exercise is None:
            return None
        
        # Add frame to queue
        if not self.frame_queue.full():
            self.frame_queue.put((frame, self.frame_count))
            self.frame_count += 1
        
        # Get latest result
        result = None
        while not self.result_queue.empty():
            result = self.result_queue.get()
        
        return result
    
    def _process_frames(self):
        """Background thread for processing frames"""
        
        while self.is_running:
            if not self.frame_queue.empty():
                frame, frame_id = self.frame_queue.get()
                
                start_time = time.time()
                
                # Detect pose
                pose = self.pose_estimator.detect_pose(frame, frame_id)
                
                if pose:
                    # Analyze exercise
                    feedback = self.exercise_analyzer.analyze_exercise(
                        pose, self.current_exercise
                    )
                    
                    # Add to result queue
                    if not self.result_queue.full():
                        self.result_queue.put(feedback)
                
                # Update processing time
                self.processing_time = time.time() - start_time
                
                # Maintain target FPS
                sleep_time = max(0, (1.0 / self.fps) - self.processing_time)
                time.sleep(sleep_time)
            else:
                time.sleep(0.01)
    
    def process_webcam(self, exercise_type: ExerciseType, 
                      duration: Optional[int] = None):
        """Process webcam feed in real-time"""
        
        cap = cv2.VideoCapture(0)
        self.start(exercise_type)
        
        start_time = time.time()
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Process frame
                feedback = self.process_frame(frame)
                
                # Detect pose for visualization
                pose = self.pose_estimator.detect_pose(frame, self.frame_count)
                
                # Draw pose on frame
                if pose:
                    frame = self.pose_estimator.draw_pose(frame, pose)
                
                # Display feedback on frame
                if feedback:
                    self._draw_feedback(frame, feedback)
                
                # Show frame
                cv2.imshow('Exercise Form Checker', frame)
                
                # Check for exit or duration
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                if duration and (time.time() - start_time) > duration:
                    break
        
        finally:
            self.stop()
            cap.release()
            cv2.destroyAllWindows()
    
    def _draw_feedback(self, frame: np.ndarray, feedback: FormFeedback):
        """Draw feedback on frame"""
        
        h, w = frame.shape[:2]
        
        # Draw status
        status_color = (0, 255, 0) if feedback.status == "correct" else (0, 0, 255)
        cv2.putText(frame, f"Status: {feedback.status.upper()}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, status_color, 2)
        
        # Draw score
        cv2.putText(frame, f"Score: {feedback.score:.0f}/100", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        # Draw rep count
        cv2.putText(frame, f"Reps: {feedback.rep_count}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        # Draw feedback messages
        y_offset = 130
        for msg in feedback.feedback[:3]:  # Show top 3 feedback items
            cv2.putText(frame, msg[:50], 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 0), 1)
            y_offset += 25
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {1/self.processing_time:.0f}", 
                   (w - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (0, 255, 0), 1)

class ExerciseFormAPI:
    """API interface for exercise form checking"""
    
    def __init__(self):
        self.form_checker = RealTimeFormChecker()
        
    def check_form(self, image_data: Union[np.ndarray, str], 
                  exercise_type: str) -> Dict[str, Any]:
        """Check form from image data"""
        
        # Parse exercise type
        try:
            exercise = ExerciseType(exercise_type.lower())
        except ValueError:
            return {
                "error": f"Unknown exercise type: {exercise_type}",
                "supported_exercises": [e.value for e in ExerciseType]
            }
        
        # Load image if path provided
        if isinstance(image_data, str):
            frame = cv2.imread(image_data)
        else:
            frame = image_data
        
        # Process frame
        self.form_checker.current_exercise = exercise
        pose = self.form_checker.pose_estimator.detect_pose(frame, 0)
        
        if not pose:
            return {
                "exercise": exercise_type,
                "status": "not_detected",
                "feedback": ["No pose detected in image"]
            }
        
        # Analyze exercise
        feedback = self.form_checker.exercise_analyzer.analyze_exercise(pose, exercise)
        
        # Convert to JSON-serializable format
        result = {
            "exercise": feedback.exercise,
            "status": feedback.status,
            "feedback": feedback.feedback,
            "corrections": feedback.corrections,
            "score": round(feedback.score, 1),
            "rep_count": feedback.rep_count
        }
        
        if feedback.metrics:
            result["metrics"] = {
                "angles": feedback.metrics.angle_metrics,
                "symmetry": round(feedback.metrics.symmetry_score, 2),
                "stability": round(feedback.metrics.stability_score, 2),
                "range_of_motion": round(feedback.metrics.range_of_motion, 2)
            }
        
        return result
    
    def start_session(self, exercise_type: str) -> str:
        """Start a new exercise session"""
        try:
            exercise = ExerciseType(exercise_type.lower())
            self.form_checker.start(exercise)
            return "Session started successfully"
        except ValueError:
            return f"Invalid exercise type: {exercise_type}"
    
    def stop_session(self) -> Dict[str, Any]:
        """Stop current session and return summary"""
        self.form_checker.stop()
        
        # Get final rep count
        rep_count = self.form_checker.exercise_analyzer.rep_counter.rep_count
        
        return {
            "status": "session_completed",
            "total_reps": rep_count,
            "duration": self.form_checker.frame_count / 30.0  # Approximate duration
        }

# Example usage
if __name__ == "__main__":
    # Initialize API
    api = ExerciseFormAPI()
    
    # Example 1: Check form from image
    result = api.check_form("path/to/image.jpg", "squat")
    print(json.dumps(result, indent=2))
    
    # Example 2: Real-time webcam processing
    form_checker = RealTimeFormChecker()
    
    print("Starting real-time form checking...")
    print("Performing: SQUAT")
    print("Press 'q' to quit")
    
    # Process webcam for 30 seconds
    form_checker.process_webcam(ExerciseType.SQUAT, duration=30)
    
    print("Session completed!")
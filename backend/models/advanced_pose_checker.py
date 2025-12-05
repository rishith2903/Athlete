"""
Advanced 3D Pose Estimation and Biomechanical Analysis Model
Uses state-of-the-art computer vision with temporal analysis and physics simulation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import cv2
from collections import deque
import mediapipe as mp

# Advanced architectures
from transformers import TimesformerModel, ViTModel
import pytorch3d.transforms as transforms_3d

@dataclass
class Pose3D:
    """3D pose representation with confidence scores"""
    keypoints_3d: np.ndarray  # (33, 4) - x, y, z, confidence
    joint_angles: Dict[str, float]
    velocity: np.ndarray
    acceleration: np.ndarray
    timestamp: float
    confidence: float

@dataclass
class BiomechanicalMetrics:
    """Biomechanical analysis metrics"""
    center_of_mass: np.ndarray
    joint_torques: Dict[str, float]
    muscle_activation: Dict[str, float]
    stability_score: float
    power_output: float
    energy_expenditure: float
    risk_score: float

class AdvancedPoseNet(nn.Module):
    """Advanced neural network for 3D pose estimation with attention mechanisms"""
    
    def __init__(self, input_dim=2048, hidden_dim=512, num_keypoints=33):
        super().__init__()
        
        # Vision Transformer backbone for feature extraction
        self.vit_backbone = ViTModel.from_pretrained('google/vit-large-patch16-224')
        
        # Temporal encoder for video analysis
        self.temporal_encoder = nn.LSTM(
            input_size=self.vit_backbone.config.hidden_size,
            hidden_size=hidden_dim,
            num_layers=3,
            batch_first=True,
            bidirectional=True
        )
        
        # 3D keypoint decoder with attention
        self.keypoint_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=0.1
        )
        
        # 3D pose estimation head
        self.pose_3d_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_keypoints * 4)  # x, y, z, confidence
        )
        
        # Biomechanical analysis network
        self.biomech_network = nn.Sequential(
            nn.Linear(num_keypoints * 4, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)  # Biomechanical features
        )
        
        # Exercise classification head
        self.exercise_classifier = nn.Linear(hidden_dim * 2, 50)  # 50 exercise types
        
        # Quality assessment head
        self.quality_scorer = nn.Linear(64, 1)
        
    def forward(self, video_frames: torch.Tensor, previous_poses: Optional[torch.Tensor] = None):
        batch_size, num_frames, C, H, W = video_frames.shape
        
        # Extract features from each frame
        frame_features = []
        for i in range(num_frames):
            frame = video_frames[:, i]
            features = self.vit_backbone(frame).last_hidden_state
            frame_features.append(features.mean(dim=1))  # Global average pooling
        
        frame_features = torch.stack(frame_features, dim=1)
        
        # Temporal encoding
        temporal_features, (hidden, cell) = self.temporal_encoder(frame_features)
        
        # Self-attention over temporal features
        attended_features, attention_weights = self.keypoint_attention(
            temporal_features, temporal_features, temporal_features
        )
        
        # 3D pose estimation
        pose_3d = self.pose_3d_head(attended_features)
        pose_3d = pose_3d.view(batch_size, num_frames, 33, 4)
        
        # Biomechanical analysis
        biomech_features = self.biomech_network(pose_3d.view(batch_size * num_frames, -1))
        quality_scores = torch.sigmoid(self.quality_scorer(biomech_features))
        
        # Exercise classification
        exercise_logits = self.exercise_classifier(attended_features.mean(dim=1))
        
        return {
            'pose_3d': pose_3d,
            'quality_scores': quality_scores.view(batch_size, num_frames),
            'exercise_class': F.softmax(exercise_logits, dim=-1),
            'attention_weights': attention_weights,
            'biomech_features': biomech_features.view(batch_size, num_frames, -1)
        }

class BiomechanicalAnalyzer:
    """Advanced biomechanical analysis using physics simulation"""
    
    def __init__(self):
        self.body_segments = self._initialize_body_model()
        self.muscle_model = self._initialize_muscle_model()
        self.joint_limits = self._load_joint_limits()
        
    def _initialize_body_model(self):
        """Initialize anthropometric body model"""
        return {
            'head': {'mass': 0.08, 'length': 0.2},
            'torso': {'mass': 0.43, 'length': 0.5},
            'upper_arm': {'mass': 0.03, 'length': 0.3},
            'forearm': {'mass': 0.02, 'length': 0.25},
            'thigh': {'mass': 0.10, 'length': 0.4},
            'shank': {'mass': 0.05, 'length': 0.4},
            'foot': {'mass': 0.01, 'length': 0.1}
        }
    
    def _initialize_muscle_model(self):
        """Initialize Hill-type muscle model parameters"""
        return {
            'quadriceps': {'max_force': 5000, 'optimal_length': 0.08, 'pennation': 8},
            'hamstrings': {'max_force': 3000, 'optimal_length': 0.09, 'pennation': 10},
            'glutes': {'max_force': 4000, 'optimal_length': 0.11, 'pennation': 12},
            'calves': {'max_force': 2500, 'optimal_length': 0.05, 'pennation': 17},
            'deltoids': {'max_force': 1500, 'optimal_length': 0.07, 'pennation': 15},
            'pectorals': {'max_force': 2000, 'optimal_length': 0.10, 'pennation': 10},
            'latissimus': {'max_force': 2500, 'optimal_length': 0.12, 'pennation': 11}
        }
    
    def _load_joint_limits(self):
        """Load anatomical joint range of motion limits"""
        return {
            'hip_flexion': (0, 120),
            'hip_extension': (-30, 0),
            'knee_flexion': (0, 140),
            'ankle_dorsiflexion': (-20, 30),
            'shoulder_flexion': (0, 180),
            'shoulder_abduction': (0, 180),
            'elbow_flexion': (0, 145),
            'spine_flexion': (-30, 90),
            'spine_rotation': (-45, 45)
        }
    
    def analyze_biomechanics(self, pose_3d: Pose3D, exercise_type: str) -> BiomechanicalMetrics:
        """Perform comprehensive biomechanical analysis"""
        
        # Calculate center of mass
        com = self._calculate_center_of_mass(pose_3d.keypoints_3d)
        
        # Calculate joint torques using inverse dynamics
        joint_torques = self._calculate_joint_torques(pose_3d)
        
        # Estimate muscle activation using EMG model
        muscle_activation = self._estimate_muscle_activation(pose_3d, exercise_type)
        
        # Calculate stability score
        stability = self._calculate_stability(com, pose_3d)
        
        # Calculate power output
        power = self._calculate_power_output(pose_3d)
        
        # Estimate energy expenditure
        energy = self._estimate_energy_expenditure(muscle_activation, pose_3d.velocity)
        
        # Calculate injury risk score
        risk = self._calculate_injury_risk(joint_torques, pose_3d.joint_angles)
        
        return BiomechanicalMetrics(
            center_of_mass=com,
            joint_torques=joint_torques,
            muscle_activation=muscle_activation,
            stability_score=stability,
            power_output=power,
            energy_expenditure=energy,
            risk_score=risk
        )
    
    def _calculate_center_of_mass(self, keypoints: np.ndarray) -> np.ndarray:
        """Calculate whole-body center of mass"""
        segment_masses = []
        segment_positions = []
        
        # Map keypoints to body segments
        segments = {
            'head': keypoints[[0, 1, 2, 3, 4]].mean(axis=0),
            'torso': keypoints[[11, 12, 23, 24]].mean(axis=0),
            'left_arm': keypoints[[11, 13, 15]].mean(axis=0),
            'right_arm': keypoints[[12, 14, 16]].mean(axis=0),
            'left_leg': keypoints[[23, 25, 27]].mean(axis=0),
            'right_leg': keypoints[[24, 26, 28]].mean(axis=0)
        }
        
        total_mass = 0
        weighted_position = np.zeros(3)
        
        for segment_name, position in segments.items():
            if 'arm' in segment_name:
                mass = self.body_segments['upper_arm']['mass'] + self.body_segments['forearm']['mass']
            elif 'leg' in segment_name:
                mass = self.body_segments['thigh']['mass'] + self.body_segments['shank']['mass']
            elif segment_name == 'head':
                mass = self.body_segments['head']['mass']
            else:  # torso
                mass = self.body_segments['torso']['mass']
            
            total_mass += mass
            weighted_position += mass * position[:3]
        
        return weighted_position / total_mass
    
    def _calculate_joint_torques(self, pose: Pose3D) -> Dict[str, float]:
        """Calculate joint torques using inverse dynamics"""
        torques = {}
        
        # Simplified inverse dynamics calculation
        for joint_name, angle in pose.joint_angles.items():
            # Calculate moment arm
            moment_arm = 0.1  # Simplified, should be joint-specific
            
            # Calculate gravitational torque
            segment_mass = 5.0  # kg, simplified
            gravity = 9.81
            torque = moment_arm * segment_mass * gravity * np.sin(np.radians(angle))
            
            # Add velocity-dependent torque
            if hasattr(pose, 'angular_velocity'):
                torque += 0.1 * pose.angular_velocity.get(joint_name, 0)
            
            torques[joint_name] = torque
        
        return torques
    
    def _estimate_muscle_activation(self, pose: Pose3D, exercise: str) -> Dict[str, float]:
        """Estimate muscle activation using EMG model"""
        activation = {}
        
        # Exercise-specific muscle activation patterns
        activation_patterns = {
            'squat': {'quadriceps': 0.8, 'glutes': 0.7, 'hamstrings': 0.5, 'calves': 0.3},
            'pushup': {'pectorals': 0.8, 'deltoids': 0.6, 'triceps': 0.7, 'core': 0.5},
            'deadlift': {'glutes': 0.9, 'hamstrings': 0.8, 'latissimus': 0.7, 'core': 0.6},
            'plank': {'core': 0.9, 'deltoids': 0.4, 'glutes': 0.3}
        }
        
        base_pattern = activation_patterns.get(exercise, {})
        
        for muscle, base_activation in base_pattern.items():
            # Modulate based on joint angles
            if muscle == 'quadriceps':
                knee_angle = pose.joint_angles.get('knee', 90)
                activation[muscle] = base_activation * (1 + 0.2 * np.sin(np.radians(knee_angle)))
            elif muscle == 'glutes':
                hip_angle = pose.joint_angles.get('hip', 90)
                activation[muscle] = base_activation * (1 + 0.15 * np.cos(np.radians(hip_angle)))
            else:
                activation[muscle] = base_activation
            
            # Ensure activation is in [0, 1]
            activation[muscle] = np.clip(activation[muscle], 0, 1)
        
        return activation
    
    def _calculate_stability(self, com: np.ndarray, pose: Pose3D) -> float:
        """Calculate postural stability score"""
        # Get base of support (feet positions)
        left_foot = pose.keypoints_3d[27][:3]
        right_foot = pose.keypoints_3d[28][:3]
        
        # Calculate base of support area (simplified as distance between feet)
        base_width = np.linalg.norm(left_foot - right_foot)
        
        # Calculate COM projection onto ground plane
        com_ground = com.copy()
        com_ground[2] = 0  # z = 0 for ground plane
        
        # Calculate distance from COM to center of base
        base_center = (left_foot + right_foot) / 2
        base_center[2] = 0
        
        com_offset = np.linalg.norm(com_ground - base_center)
        
        # Stability score (1.0 = perfectly stable, 0.0 = unstable)
        stability = np.exp(-com_offset / (base_width + 1e-6))
        
        # Account for velocity (moving = less stable)
        velocity_magnitude = np.linalg.norm(pose.velocity)
        stability *= np.exp(-velocity_magnitude * 0.1)
        
        return float(np.clip(stability, 0, 1))
    
    def _calculate_power_output(self, pose: Pose3D) -> float:
        """Calculate mechanical power output"""
        # Power = Force × Velocity
        # Simplified calculation using joint velocities and estimated forces
        
        total_power = 0
        for joint_name, torque in pose.joint_angles.items():
            # Get angular velocity (rad/s)
            angular_velocity = np.linalg.norm(pose.velocity) * 0.1  # Simplified
            
            # Power = Torque × Angular Velocity
            joint_power = abs(torque * angular_velocity)
            total_power += joint_power
        
        return total_power
    
    def _estimate_energy_expenditure(self, muscle_activation: Dict, velocity: np.ndarray) -> float:
        """Estimate metabolic energy expenditure"""
        # Based on muscle activation and movement velocity
        
        # Basal metabolic rate component
        bmr = 1.5  # kcal/min
        
        # Activity component
        activity_energy = 0
        for muscle, activation in muscle_activation.items():
            muscle_params = self.muscle_model.get(muscle, {'max_force': 2000})
            # Energy proportional to activation and muscle size
            activity_energy += activation * muscle_params['max_force'] * 0.001
        
        # Velocity component (moving faster = more energy)
        velocity_factor = 1 + np.linalg.norm(velocity) * 0.5
        
        total_energy = (bmr + activity_energy) * velocity_factor
        
        return total_energy
    
    def _calculate_injury_risk(self, joint_torques: Dict, joint_angles: Dict) -> float:
        """Calculate injury risk score based on joint stress and angles"""
        risk_score = 0
        risk_factors = []
        
        for joint_name, angle in joint_angles.items():
            if joint_name in self.joint_limits:
                min_angle, max_angle = self.joint_limits[joint_name]
                
                # Check if joint is near limits
                if angle < min_angle + 10 or angle > max_angle - 10:
                    risk_factors.append(0.3)  # Near limit risk
                
                # Check if joint is beyond limits
                if angle < min_angle or angle > max_angle:
                    risk_factors.append(0.7)  # Over limit risk
            
            # Check torque levels
            torque = joint_torques.get(joint_name, 0)
            if abs(torque) > 100:  # High torque threshold (Nm)
                risk_factors.append(0.4)
        
        # Combine risk factors
        if risk_factors:
            risk_score = 1 - np.prod([1 - r for r in risk_factors])
        
        return float(np.clip(risk_score, 0, 1))

class TemporalMovementAnalyzer:
    """Analyzes movement patterns over time"""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.pose_buffer = deque(maxlen=window_size)
        self.pattern_recognizer = self._initialize_pattern_recognizer()
        
    def _initialize_pattern_recognizer(self):
        """Initialize LSTM for temporal pattern recognition"""
        return nn.LSTM(
            input_size=33 * 4,  # 33 keypoints × 4 values
            hidden_size=256,
            num_layers=2,
            batch_first=True
        )
    
    def analyze_movement_quality(self, pose_sequence: List[Pose3D]) -> Dict:
        """Analyze movement quality over time"""
        
        # Calculate movement smoothness
        smoothness = self._calculate_smoothness(pose_sequence)
        
        # Detect repetitions
        reps = self._detect_repetitions(pose_sequence)
        
        # Analyze tempo
        tempo = self._analyze_tempo(pose_sequence)
        
        # Detect compensatory movements
        compensations = self._detect_compensations(pose_sequence)
        
        # Calculate consistency score
        consistency = self._calculate_consistency(reps)
        
        return {
            'smoothness': smoothness,
            'repetitions': reps,
            'tempo': tempo,
            'compensations': compensations,
            'consistency': consistency,
            'movement_quality_score': self._calculate_overall_quality(
                smoothness, consistency, len(compensations)
            )
        }
    
    def _calculate_smoothness(self, poses: List[Pose3D]) -> float:
        """Calculate movement smoothness using jerk metric"""
        if len(poses) < 3:
            return 1.0
        
        jerks = []
        for i in range(2, len(poses)):
            # Calculate jerk (derivative of acceleration)
            acc_diff = poses[i].acceleration - poses[i-1].acceleration
            dt = poses[i].timestamp - poses[i-1].timestamp
            if dt > 0:
                jerk = np.linalg.norm(acc_diff) / dt
                jerks.append(jerk)
        
        if jerks:
            # Normalized smoothness (lower jerk = smoother)
            avg_jerk = np.mean(jerks)
            smoothness = np.exp(-avg_jerk / 10)  # Normalize to [0, 1]
            return float(np.clip(smoothness, 0, 1))
        return 1.0
    
    def _detect_repetitions(self, poses: List[Pose3D]) -> List[Dict]:
        """Detect and analyze exercise repetitions"""
        reps = []
        
        # Simple peak detection for repetitions
        # Track vertical movement of key joints
        if not poses:
            return reps
        
        # Use hip keypoint for squat/lunge detection
        hip_heights = [p.keypoints_3d[23][2] for p in poses]  # Left hip z-coordinate
        
        # Find peaks and valleys
        peaks = []
        valleys = []
        
        for i in range(1, len(hip_heights) - 1):
            if hip_heights[i] > hip_heights[i-1] and hip_heights[i] > hip_heights[i+1]:
                peaks.append(i)
            elif hip_heights[i] < hip_heights[i-1] and hip_heights[i] < hip_heights[i+1]:
                valleys.append(i)
        
        # Pair peaks and valleys to form repetitions
        for i in range(len(valleys) - 1):
            if i < len(peaks) - 1:
                rep = {
                    'start_frame': valleys[i],
                    'end_frame': valleys[i+1],
                    'duration': poses[valleys[i+1]].timestamp - poses[valleys[i]].timestamp,
                    'range_of_motion': abs(hip_heights[peaks[i]] - hip_heights[valleys[i]]),
                    'quality_score': poses[valleys[i]].confidence
                }
                reps.append(rep)
        
        return reps
    
    def _analyze_tempo(self, poses: List[Pose3D]) -> Dict:
        """Analyze movement tempo and rhythm"""
        if len(poses) < 2:
            return {'eccentric': 0, 'concentric': 0, 'total': 0}
        
        velocities = [np.linalg.norm(p.velocity) for p in poses]
        
        # Identify eccentric (lowering) and concentric (lifting) phases
        eccentric_frames = [i for i, v in enumerate(velocities) if v < 0]
        concentric_frames = [i for i, v in enumerate(velocities) if v > 0]
        
        eccentric_time = len(eccentric_frames) / 30.0  # Assuming 30 FPS
        concentric_time = len(concentric_frames) / 30.0
        
        return {
            'eccentric': eccentric_time,
            'concentric': concentric_time,
            'total': eccentric_time + concentric_time,
            'ratio': eccentric_time / (concentric_time + 1e-6)
        }
    
    def _detect_compensations(self, poses: List[Pose3D]) -> List[str]:
        """Detect compensatory movement patterns"""
        compensations = []
        
        for pose in poses:
            # Check for common compensations
            
            # Excessive forward lean
            torso_angle = pose.joint_angles.get('torso_forward_lean', 0)
            if abs(torso_angle) > 30:
                compensations.append("Excessive forward lean detected")
            
            # Knee valgus (knees caving in)
            knee_alignment = pose.joint_angles.get('knee_valgus', 0)
            if abs(knee_alignment) > 10:
                compensations.append("Knee valgus detected")
            
            # Hip shift
            hip_shift = pose.joint_angles.get('hip_lateral_shift', 0)
            if abs(hip_shift) > 5:
                compensations.append("Lateral hip shift detected")
        
        # Return unique compensations
        return list(set(compensations))
    
    def _calculate_consistency(self, reps: List[Dict]) -> float:
        """Calculate movement consistency across repetitions"""
        if len(reps) < 2:
            return 1.0
        
        # Calculate variance in rep duration and ROM
        durations = [r['duration'] for r in reps]
        roms = [r['range_of_motion'] for r in reps]
        
        duration_cv = np.std(durations) / (np.mean(durations) + 1e-6)  # Coefficient of variation
        rom_cv = np.std(roms) / (np.mean(roms) + 1e-6)
        
        # Lower CV = higher consistency
        consistency = np.exp(-(duration_cv + rom_cv))
        
        return float(np.clip(consistency, 0, 1))
    
    def _calculate_overall_quality(self, smoothness: float, consistency: float, 
                                  num_compensations: int) -> float:
        """Calculate overall movement quality score"""
        # Weighted combination of factors
        compensation_penalty = np.exp(-num_compensations * 0.2)
        
        quality = (
            0.3 * smoothness + 
            0.3 * consistency + 
            0.4 * compensation_penalty
        )
        
        return float(np.clip(quality, 0, 1))

class AdvancedPoseChecker:
    """Main class for advanced pose checking with all features integrated"""
    
    def __init__(self, model_path: Optional[str] = None):
        # Initialize models
        self.pose_net = AdvancedPoseNet()
        self.biomech_analyzer = BiomechanicalAnalyzer()
        self.temporal_analyzer = TemporalMovementAnalyzer()
        
        # MediaPipe for initial pose detection
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )
        
        # Load pre-trained weights if available
        if model_path:
            self.load_model(model_path)
        
        # Exercise-specific form rules
        self.form_rules = self._load_form_rules()
        
    def _load_form_rules(self) -> Dict:
        """Load exercise-specific form checking rules"""
        return {
            'squat': {
                'knee_angle_min': 70,
                'knee_angle_max': 100,
                'hip_angle_min': 70,
                'hip_angle_max': 100,
                'back_angle_min': 160,
                'knee_over_toe_max': 5,
                'hip_width_ratio': 1.2
            },
            'pushup': {
                'elbow_angle_min': 70,
                'elbow_angle_max': 100,
                'body_alignment_min': 170,
                'hand_width_ratio': 1.5,
                'core_engagement_min': 0.7
            },
            'deadlift': {
                'back_angle_min': 160,
                'hip_hinge_angle': 45,
                'knee_angle_max': 30,
                'bar_path_deviation_max': 5,
                'shoulder_position': 'over_bar'
            }
        }
    
    def analyze_video(self, video_path: str) -> Dict:
        """Perform comprehensive video analysis"""
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames = []
        poses_3d = []
        timestamps = []
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frames.append(frame)
            timestamps.append(frame_count / fps)
            frame_count += 1
            
            # Limit to reasonable number of frames
            if frame_count > 300:  # 10 seconds at 30fps
                break
        
        cap.release()
        
        # Process frames in batches
        batch_size = 16
        all_results = []
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            batch_timestamps = timestamps[i:i+batch_size]
            
            # Extract 3D poses
            batch_poses = self._extract_3d_poses(batch_frames, batch_timestamps)
            poses_3d.extend(batch_poses)
            
            # Perform frame-level analysis
            for frame, pose, timestamp in zip(batch_frames, batch_poses, batch_timestamps):
                frame_result = self._analyze_frame(frame, pose, timestamp)
                all_results.append(frame_result)
        
        # Perform temporal analysis
        temporal_results = self.temporal_analyzer.analyze_movement_quality(poses_3d)
        
        # Aggregate results
        final_results = self._aggregate_results(all_results, temporal_results)
        
        return final_results
    
    def _extract_3d_poses(self, frames: List[np.ndarray], 
                         timestamps: List[float]) -> List[Pose3D]:
        """Extract 3D poses from frames"""
        poses_3d = []
        
        for frame, timestamp in zip(frames, timestamps):
            # Use MediaPipe for initial detection
            results = self.pose_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                # Convert to 3D keypoints
                keypoints_3d = np.zeros((33, 4))
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    keypoints_3d[i] = [
                        landmark.x,
                        landmark.y,
                        landmark.z,
                        landmark.visibility
                    ]
                
                # Calculate joint angles
                joint_angles = self._calculate_joint_angles(keypoints_3d)
                
                # Calculate velocity and acceleration
                if len(poses_3d) > 0:
                    prev_pose = poses_3d[-1]
                    dt = timestamp - prev_pose.timestamp
                    velocity = (keypoints_3d[:, :3] - prev_pose.keypoints_3d[:, :3]) / (dt + 1e-6)
                    
                    if len(poses_3d) > 1:
                        prev_prev_pose = poses_3d[-2]
                        prev_velocity = prev_pose.velocity
                        acceleration = (velocity - prev_velocity) / (dt + 1e-6)
                    else:
                        acceleration = np.zeros_like(velocity)
                else:
                    velocity = np.zeros((33, 3))
                    acceleration = np.zeros((33, 3))
                
                pose_3d = Pose3D(
                    keypoints_3d=keypoints_3d,
                    joint_angles=joint_angles,
                    velocity=velocity.mean(axis=0),  # Average velocity
                    acceleration=acceleration.mean(axis=0),
                    timestamp=timestamp,
                    confidence=float(keypoints_3d[:, 3].mean())
                )
                
                poses_3d.append(pose_3d)
        
        return poses_3d
    
    def _calculate_joint_angles(self, keypoints: np.ndarray) -> Dict[str, float]:
        """Calculate joint angles from keypoints"""
        angles = {}
        
        # Define joint triplets for angle calculation
        joint_triplets = {
            'left_elbow': (11, 13, 15),  # shoulder-elbow-wrist
            'right_elbow': (12, 14, 16),
            'left_knee': (23, 25, 27),  # hip-knee-ankle
            'right_knee': (24, 26, 28),
            'left_hip': (11, 23, 25),  # shoulder-hip-knee
            'right_hip': (12, 24, 26),
            'left_shoulder': (13, 11, 23),  # elbow-shoulder-hip
            'right_shoulder': (14, 12, 24)
        }
        
        for joint_name, (p1_idx, p2_idx, p3_idx) in joint_triplets.items():
            p1 = keypoints[p1_idx, :3]
            p2 = keypoints[p2_idx, :3]
            p3 = keypoints[p3_idx, :3]
            
            # Calculate angle using dot product
            v1 = p1 - p2
            v2 = p3 - p2
            
            cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
            
            angles[joint_name] = float(angle)
        
        return angles
    
    def _analyze_frame(self, frame: np.ndarray, pose: Pose3D, timestamp: float) -> Dict:
        """Analyze single frame"""
        
        # Detect exercise type
        exercise_type = self._detect_exercise_type(pose)
        
        # Biomechanical analysis
        biomech_metrics = self.biomech_analyzer.analyze_biomechanics(pose, exercise_type)
        
        # Form checking
        form_feedback = self._check_form(pose, exercise_type)
        
        return {
            'timestamp': timestamp,
            'exercise_type': exercise_type,
            'pose_confidence': pose.confidence,
            'biomech_metrics': biomech_metrics,
            'form_feedback': form_feedback,
            'joint_angles': pose.joint_angles
        }
    
    def _detect_exercise_type(self, pose: Pose3D) -> str:
        """Detect exercise type from pose"""
        # Simple rule-based detection (can be replaced with ML model)
        
        hip_angle = np.mean([
            pose.joint_angles.get('left_hip', 90),
            pose.joint_angles.get('right_hip', 90)
        ])
        knee_angle = np.mean([
            pose.joint_angles.get('left_knee', 90),
            pose.joint_angles.get('right_knee', 90)
        ])
        
        # Check for squat
        if hip_angle < 100 and knee_angle < 100:
            return 'squat'
        
        # Check for plank
        if abs(pose.keypoints_3d[11][2] - pose.keypoints_3d[27][2]) < 0.3:  # Hip and ankle similar height
            return 'plank'
        
        # Check for pushup
        if pose.keypoints_3d[11][2] < 0.5:  # Low body position
            return 'pushup'
        
        return 'unknown'
    
    def _check_form(self, pose: Pose3D, exercise_type: str) -> Dict:
        """Check exercise form and provide feedback"""
        
        feedback = {
            'score': 100,
            'issues': [],
            'corrections': [],
            'is_safe': True
        }
        
        if exercise_type not in self.form_rules:
            return feedback
        
        rules = self.form_rules[exercise_type]
        
        # Check each rule
        for rule_name, rule_value in rules.items():
            if 'angle' in rule_name:
                # Check angle constraints
                joint_name = rule_name.replace('_min', '').replace('_max', '').replace('_angle', '')
                if joint_name in pose.joint_angles:
                    angle = pose.joint_angles[joint_name]
                    
                    if '_min' in rule_name and angle < rule_value:
                        feedback['issues'].append(f"{joint_name} angle too small: {angle:.1f}°")
                        feedback['corrections'].append(f"Increase {joint_name} angle to at least {rule_value}°")
                        feedback['score'] -= 10
                    elif '_max' in rule_name and angle > rule_value:
                        feedback['issues'].append(f"{joint_name} angle too large: {angle:.1f}°")
                        feedback['corrections'].append(f"Decrease {joint_name} angle to maximum {rule_value}°")
                        feedback['score'] -= 10
        
        # Check safety
        if feedback['score'] < 70:
            feedback['is_safe'] = False
        
        return feedback
    
    def _aggregate_results(self, frame_results: List[Dict], 
                          temporal_results: Dict) -> Dict:
        """Aggregate all analysis results"""
        
        # Calculate average metrics
        avg_confidence = np.mean([r['pose_confidence'] for r in frame_results])
        avg_form_score = np.mean([r['form_feedback']['score'] for r in frame_results])
        
        # Identify most common exercise type
        exercise_types = [r['exercise_type'] for r in frame_results]
        most_common_exercise = max(set(exercise_types), key=exercise_types.count)
        
        # Collect all issues and corrections
        all_issues = []
        all_corrections = []
        for result in frame_results:
            all_issues.extend(result['form_feedback']['issues'])
            all_corrections.extend(result['form_feedback']['corrections'])
        
        # Get unique issues and corrections
        unique_issues = list(set(all_issues))
        unique_corrections = list(set(all_corrections))
        
        # Calculate biomechanical summary
        biomech_summary = self._summarize_biomechanics(frame_results)
        
        return {
            'exercise_type': most_common_exercise,
            'overall_score': (avg_form_score + temporal_results['movement_quality_score'] * 100) / 2,
            'pose_confidence': avg_confidence,
            'form_score': avg_form_score,
            'movement_quality': temporal_results,
            'biomechanics': biomech_summary,
            'issues_detected': unique_issues,
            'corrections_suggested': unique_corrections,
            'repetitions': temporal_results['repetitions'],
            'is_safe': all(r['form_feedback']['is_safe'] for r in frame_results),
            'detailed_results': frame_results
        }
    
    def _summarize_biomechanics(self, frame_results: List[Dict]) -> Dict:
        """Summarize biomechanical metrics"""
        
        # Extract all biomech metrics
        all_metrics = [r['biomech_metrics'] for r in frame_results]
        
        return {
            'avg_stability': np.mean([m.stability_score for m in all_metrics]),
            'avg_power': np.mean([m.power_output for m in all_metrics]),
            'total_energy': sum([m.energy_expenditure for m in all_metrics]),
            'max_risk_score': max([m.risk_score for m in all_metrics]),
            'muscle_activation_summary': self._summarize_muscle_activation(all_metrics)
        }
    
    def _summarize_muscle_activation(self, metrics: List[BiomechanicalMetrics]) -> Dict:
        """Summarize muscle activation across all frames"""
        muscle_totals = {}
        
        for metric in metrics:
            for muscle, activation in metric.muscle_activation.items():
                if muscle not in muscle_totals:
                    muscle_totals[muscle] = []
                muscle_totals[muscle].append(activation)
        
        summary = {}
        for muscle, activations in muscle_totals.items():
            summary[muscle] = {
                'average': float(np.mean(activations)),
                'peak': float(np.max(activations)),
                'time_active': float(np.mean([a > 0.3 for a in activations]))  # % time > 30% activation
            }
        
        return summary
    
    def provide_real_time_feedback(self, frame: np.ndarray) -> Dict:
        """Provide real-time feedback for live video"""
        
        # Quick pose extraction
        results = self.pose_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if not results.pose_landmarks:
            return {'status': 'No pose detected', 'feedback': []}
        
        # Convert to simplified pose
        keypoints = np.array([[l.x, l.y, l.z, l.visibility] 
                             for l in results.pose_landmarks.landmark])
        
        # Quick form check
        joint_angles = self._calculate_joint_angles(keypoints)
        
        # Generate immediate feedback
        feedback = []
        
        # Check key angles
        if 'left_knee' in joint_angles:
            knee_angle = joint_angles['left_knee']
            if knee_angle < 70:
                feedback.append("Go deeper in your squat")
            elif knee_angle > 160:
                feedback.append("Bend your knees more")
        
        return {
            'status': 'Pose detected',
            'confidence': float(keypoints[:, 3].mean()),
            'feedback': feedback,
            'joint_angles': joint_angles
        }
    
    def save_model(self, path: str):
        """Save trained model weights"""
        torch.save({
            'pose_net': self.pose_net.state_dict(),
            'form_rules': self.form_rules
        }, path)
    
    def load_model(self, path: str):
        """Load trained model weights"""
        checkpoint = torch.load(path)
        self.pose_net.load_state_dict(checkpoint['pose_net'])
        self.form_rules = checkpoint['form_rules']
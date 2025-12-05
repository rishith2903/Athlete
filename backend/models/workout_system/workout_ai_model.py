"""
Advanced Workout System Model
Combines deep learning workout recommendation with progression tracking and performance prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import networkx as nx
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Enums for workout categories
class WorkoutType(Enum):
    STRENGTH = "strength"
    CARDIO = "cardio"
    FLEXIBILITY = "flexibility"
    HIIT = "hiit"
    ENDURANCE = "endurance"
    POWERLIFTING = "powerlifting"
    BODYBUILDING = "bodybuilding"
    CROSSFIT = "crossfit"

class MuscleGroup(Enum):
    CHEST = "chest"
    BACK = "back"
    SHOULDERS = "shoulders"
    ARMS = "arms"
    LEGS = "legs"
    CORE = "core"
    GLUTES = "glutes"
    FULL_BODY = "full_body"

class ExperienceLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    ELITE = "elite"

@dataclass
class UserProfile:
    """Comprehensive user profile for personalization"""
    user_id: str
    age: int
    weight: float  # kg
    height: float  # cm
    body_fat: Optional[float] = None
    experience_level: ExperienceLevel = ExperienceLevel.BEGINNER
    goals: List[str] = field(default_factory=list)
    injuries: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    available_equipment: List[str] = field(default_factory=list)
    workout_frequency: int = 3  # days per week
    max_workout_duration: int = 60  # minutes

@dataclass
class Exercise:
    """Exercise data structure"""
    name: str
    muscle_groups: List[MuscleGroup]
    equipment: List[str]
    difficulty: int  # 1-10
    calories_per_minute: float
    movement_pattern: str  # push, pull, squat, hinge, carry, etc.
    
@dataclass
class WorkoutSession:
    """Represents a single workout session"""
    date: datetime
    exercises: List[Dict[str, Any]]
    duration: int  # minutes
    calories_burned: float
    avg_heart_rate: Optional[float] = None
    perceived_effort: Optional[int] = None  # 1-10
    notes: Optional[str] = None

@dataclass
class ProgressMetrics:
    """Tracks user progress over time"""
    strength_gains: Dict[str, float]
    endurance_improvement: float
    weight_change: float
    body_composition_change: Optional[float]
    consistency_score: float
    recovery_rate: float
    performance_index: float

class TransformerBlock(nn.Module):
    """Transformer block for sequence modeling"""
    def __init__(self, embed_dim: int, num_heads: int = 8, ff_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attn_output, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_output)
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

class CollaborativeFilteringNetwork(nn.Module):
    """Deep collaborative filtering for workout recommendations"""
    def __init__(self, num_users: int, num_exercises: int, embed_dim: int = 128):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.exercise_embedding = nn.Embedding(num_exercises, embed_dim)
        
        # Neural collaborative filtering layers
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Matrix factorization component
        self.user_bias = nn.Embedding(num_users, 1)
        self.exercise_bias = nn.Embedding(num_exercises, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, user_ids: torch.Tensor, exercise_ids: torch.Tensor) -> torch.Tensor:
        user_embeds = self.user_embedding(user_ids)
        exercise_embeds = self.exercise_embedding(exercise_ids)
        
        # Concatenate embeddings
        concat = torch.cat([user_embeds, exercise_embeds], dim=-1)
        
        # MLP prediction
        mlp_output = self.mlp(concat)
        
        # Matrix factorization prediction
        mf_output = (user_embeds * exercise_embeds).sum(dim=-1, keepdim=True)
        mf_output += self.user_bias(user_ids) + self.exercise_bias(exercise_ids) + self.global_bias
        
        # Combine predictions
        output = 0.7 * mlp_output + 0.3 * mf_output
        
        return torch.sigmoid(output)

class WorkoutRecommender(nn.Module):
    """Advanced workout recommendation system using deep learning"""
    def __init__(self, num_exercises: int = 500, num_features: int = 50):
        super().__init__()
        
        # User profile encoder
        self.user_encoder = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        # Exercise encoder
        self.exercise_encoder = nn.Sequential(
            nn.Linear(20, 128),  # Exercise features
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Attention mechanism for exercise selection
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=8)
        
        # Workout sequence generator
        self.sequence_generator = nn.GRU(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Final recommendation layers
        self.recommender = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_exercises)
        )
        
        # Transformer blocks for sequence modeling
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(128, 8, 512, 0.1) for _ in range(3)
        ])
        
    def forward(self, user_features: torch.Tensor, 
                exercise_history: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate workout recommendations"""
        
        # Encode user profile
        user_repr = self.user_encoder(user_features)
        
        # If no history, generate from scratch
        if exercise_history is None:
            batch_size = user_features.size(0)
            hidden = user_repr.unsqueeze(0).repeat(2, 1, 1)
            
            # Generate initial sequence
            output, _ = self.sequence_generator(
                user_repr.unsqueeze(1).repeat(1, 10, 1),
                hidden
            )
        else:
            # Process exercise history through transformers
            for transformer in self.transformer_blocks:
                exercise_history = transformer(exercise_history)
            
            # Combine with user representation
            attended, _ = self.attention(
                user_repr.unsqueeze(1),
                exercise_history,
                exercise_history
            )
            
            # Generate sequence
            output, _ = self.sequence_generator(attended)
        
        # Get recommendations
        recommendations = self.recommender(output[:, -1, :])
        
        return F.softmax(recommendations, dim=-1)

class ProgressionTracker(nn.Module):
    """LSTM-based progression tracking and performance prediction"""
    def __init__(self, input_dim: int = 20, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        
        # Bidirectional LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Attention layer for important time points
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Performance prediction head
        self.performance_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # Multiple performance metrics
        )
        
        # Fatigue and recovery model
        self.fatigue_model = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 5, 64),  # +5 for additional features
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Fatigue level, recovery time
        )
        
    def forward(self, workout_history: torch.Tensor, 
                additional_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Analyze workout history and predict future performance"""
        
        # Process through LSTM
        lstm_out, (hidden, cell) = self.lstm(workout_history)
        
        # Apply temporal attention
        attention_weights = F.softmax(self.temporal_attention(lstm_out), dim=1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Predict performance metrics
        performance = self.performance_predictor(attended)
        
        # Predict fatigue and recovery
        if additional_features is not None:
            fatigue_input = torch.cat([attended, additional_features], dim=-1)
        else:
            fatigue_input = torch.cat([attended, torch.zeros(attended.size(0), 5)], dim=-1)
        
        fatigue_recovery = self.fatigue_model(fatigue_input)
        
        return {
            'performance_metrics': performance,
            'fatigue_level': torch.sigmoid(fatigue_recovery[:, 0]),
            'recovery_time': torch.abs(fatigue_recovery[:, 1]) * 48,  # Hours
            'attention_weights': attention_weights
        }

class AdaptiveWorkoutOptimizer:
    """Optimizes workout plans using reinforcement learning principles"""
    
    def __init__(self, state_dim: int = 50, action_dim: int = 100):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Q-Network for workout selection
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        # Target network for stable learning
        self.target_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        # Copy weights to target
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        self.memory = []
        self.epsilon = 0.1
        
    def select_action(self, state: torch.Tensor) -> int:
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        
        with torch.no_grad():
            q_values = self.q_network(state)
            return torch.argmax(q_values).item()
    
    def optimize(self, batch_size: int = 32) -> float:
        """Optimize the Q-network"""
        if len(self.memory) < batch_size:
            return 0.0
        
        # Sample batch from memory
        batch = np.random.choice(self.memory, batch_size, replace=False)
        states = torch.stack([e[0] for e in batch])
        actions = torch.tensor([e[1] for e in batch])
        rewards = torch.tensor([e[2] for e in batch], dtype=torch.float32)
        next_states = torch.stack([e[3] for e in batch])
        dones = torch.tensor([e[4] for e in batch], dtype=torch.float32)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + 0.99 * next_q * (1 - dones)
        
        # Compute loss
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

class WorkoutSystemAI:
    """Main system integrating all workout AI components"""
    
    def __init__(self):
        # Initialize models
        self.recommender = WorkoutRecommender()
        self.progression_tracker = ProgressionTracker()
        self.collaborative_filter = CollaborativeFilteringNetwork(
            num_users=10000, 
            num_exercises=500
        )
        self.optimizer = AdaptiveWorkoutOptimizer()
        
        # Exercise database
        self.exercise_db = self._initialize_exercise_database()
        
        # User profiles storage
        self.user_profiles: Dict[str, UserProfile] = {}
        
        # Workout history storage
        self.workout_history: Dict[str, List[WorkoutSession]] = {}
        
        # Feature scalers
        self.user_scaler = StandardScaler()
        self.workout_scaler = MinMaxScaler()
        
        # Performance metrics
        self.performance_metrics: Dict[str, ProgressMetrics] = {}
        
    def _initialize_exercise_database(self) -> Dict[str, Exercise]:
        """Initialize exercise database"""
        exercises = {
            "barbell_squat": Exercise(
                "Barbell Squat", [MuscleGroup.LEGS, MuscleGroup.GLUTES], 
                ["barbell", "squat_rack"], 9, 8.5, "squat"
            ),
            "bench_press": Exercise(
                "Bench Press", [MuscleGroup.CHEST, MuscleGroup.ARMS], 
                ["barbell", "bench"], 8, 7.0, "push"
            ),
            "deadlift": Exercise(
                "Deadlift", [MuscleGroup.BACK, MuscleGroup.LEGS], 
                ["barbell"], 9, 9.0, "hinge"
            ),
            "pull_up": Exercise(
                "Pull Up", [MuscleGroup.BACK, MuscleGroup.ARMS], 
                ["pull_up_bar"], 7, 6.5, "pull"
            ),
            "push_up": Exercise(
                "Push Up", [MuscleGroup.CHEST, MuscleGroup.ARMS], 
                [], 4, 5.0, "push"
            ),
            "plank": Exercise(
                "Plank", [MuscleGroup.CORE], 
                [], 3, 4.0, "isometric"
            ),
            "running": Exercise(
                "Running", [MuscleGroup.LEGS, MuscleGroup.FULL_BODY], 
                [], 5, 10.0, "cardio"
            ),
            "burpee": Exercise(
                "Burpee", [MuscleGroup.FULL_BODY], 
                [], 6, 8.0, "compound"
            )
        }
        return exercises
    
    def create_user_profile(self, user_data: Dict[str, Any]) -> UserProfile:
        """Create and store user profile"""
        profile = UserProfile(
            user_id=user_data['user_id'],
            age=user_data['age'],
            weight=user_data['weight'],
            height=user_data['height'],
            body_fat=user_data.get('body_fat'),
            experience_level=ExperienceLevel(user_data.get('experience', 'beginner')),
            goals=user_data.get('goals', []),
            injuries=user_data.get('injuries', []),
            preferences=user_data.get('preferences', {}),
            available_equipment=user_data.get('equipment', []),
            workout_frequency=user_data.get('frequency', 3),
            max_workout_duration=user_data.get('max_duration', 60)
        )
        
        self.user_profiles[profile.user_id] = profile
        return profile
    
    def generate_workout_plan(self, user_id: str, 
                            num_weeks: int = 4) -> Dict[str, Any]:
        """Generate personalized workout plan"""
        
        profile = self.user_profiles.get(user_id)
        if not profile:
            raise ValueError(f"User profile not found for {user_id}")
        
        # Prepare user features
        user_features = self._encode_user_profile(profile)
        user_tensor = torch.FloatTensor(user_features).unsqueeze(0)
        
        # Get workout history if exists
        history = self.workout_history.get(user_id, [])
        history_tensor = None
        if history:
            history_tensor = self._encode_workout_history(history[-10:])  # Last 10 workouts
        
        # Generate recommendations
        with torch.no_grad():
            recommendations = self.recommender(user_tensor, history_tensor)
        
        # Create weekly plan
        weekly_plan = self._create_weekly_plan(
            profile, recommendations, num_weeks
        )
        
        # Optimize plan using RL
        optimized_plan = self._optimize_workout_plan(weekly_plan, profile)
        
        # Add progression scheme
        progressive_plan = self._add_progression(optimized_plan, profile)
        
        return {
            'user_id': user_id,
            'plan_duration': f"{num_weeks} weeks",
            'weekly_schedule': progressive_plan,
            'estimated_results': self._estimate_results(profile, progressive_plan),
            'nutrition_guidelines': self._generate_nutrition_guidelines(profile),
            'recovery_protocol': self._create_recovery_protocol(profile)
        }
    
    def _encode_user_profile(self, profile: UserProfile) -> np.ndarray:
        """Encode user profile into feature vector"""
        features = []
        
        # Basic features
        features.extend([
            profile.age / 100.0,
            profile.weight / 150.0,
            profile.height / 200.0,
            profile.body_fat / 100.0 if profile.body_fat else 0.2,
            profile.workout_frequency / 7.0,
            profile.max_workout_duration / 120.0
        ])
        
        # Experience level (one-hot)
        exp_encoding = [0, 0, 0, 0]
        exp_encoding[profile.experience_level.value] = 1
        features.extend(exp_encoding)
        
        # Goals encoding
        goal_types = ['weight_loss', 'muscle_gain', 'endurance', 'strength', 'flexibility']
        goal_encoding = [1 if g in profile.goals else 0 for g in goal_types]
        features.extend(goal_encoding)
        
        # Equipment availability
        equipment_types = ['barbell', 'dumbbell', 'kettlebell', 'resistance_bands', 
                          'pull_up_bar', 'bench', 'squat_rack', 'cardio_machine']
        equipment_encoding = [1 if e in profile.available_equipment else 0 for e in equipment_types]
        features.extend(equipment_encoding)
        
        # Injury flags
        injury_areas = ['knee', 'back', 'shoulder', 'ankle', 'wrist']
        injury_encoding = [1 if i in profile.injuries else 0 for i in injury_areas]
        features.extend(injury_encoding)
        
        # Pad to fixed size
        while len(features) < 50:
            features.append(0)
        
        return np.array(features[:50])
    
    def _encode_workout_history(self, history: List[WorkoutSession]) -> torch.Tensor:
        """Encode workout history for model input"""
        encoded = []
        
        for session in history:
            session_features = [
                session.duration / 120.0,
                session.calories_burned / 1000.0,
                session.avg_heart_rate / 200.0 if session.avg_heart_rate else 0.5,
                session.perceived_effort / 10.0 if session.perceived_effort else 0.5,
                len(session.exercises) / 20.0
            ]
            
            # Add exercise type distribution
            exercise_types = {'strength': 0, 'cardio': 0, 'flexibility': 0}
            for ex in session.exercises:
                if 'type' in ex:
                    exercise_types[ex['type']] = exercise_types.get(ex['type'], 0) + 1
            
            total_exercises = sum(exercise_types.values())
            if total_exercises > 0:
                for ex_type in ['strength', 'cardio', 'flexibility']:
                    session_features.append(exercise_types[ex_type] / total_exercises)
            else:
                session_features.extend([0, 0, 0])
            
            # Pad to fixed size
            while len(session_features) < 20:
                session_features.append(0)
            
            encoded.append(session_features[:20])
        
        return torch.FloatTensor(encoded).unsqueeze(0)
    
    def _create_weekly_plan(self, profile: UserProfile, 
                           recommendations: torch.Tensor,
                           num_weeks: int) -> List[Dict[str, Any]]:
        """Create weekly workout plan based on recommendations"""
        
        # Get top recommended exercises
        top_k = min(20, recommendations.size(-1))
        _, top_indices = torch.topk(recommendations[0], top_k)
        
        # Map indices to exercises
        exercise_list = list(self.exercise_db.keys())
        recommended_exercises = [exercise_list[min(idx, len(exercise_list)-1)] 
                                for idx in top_indices.tolist()]
        
        weekly_plan = []
        
        for week in range(num_weeks):
            week_plan = {
                'week': week + 1,
                'workouts': []
            }
            
            # Create workouts for the week
            for day in range(profile.workout_frequency):
                workout = self._create_single_workout(
                    profile, recommended_exercises, week, day
                )
                week_plan['workouts'].append(workout)
            
            weekly_plan.append(week_plan)
        
        return weekly_plan
    
    def _create_single_workout(self, profile: UserProfile, 
                              exercises: List[str],
                              week: int, day: int) -> Dict[str, Any]:
        """Create a single workout session"""
        
        # Determine workout type based on day
        workout_types = ['upper_body', 'lower_body', 'full_body', 'cardio', 'active_recovery']
        workout_type = workout_types[day % len(workout_types)]
        
        selected_exercises = []
        total_duration = 0
        
        # Warm-up
        selected_exercises.append({
            'name': 'Dynamic Warm-up',
            'duration': 10,
            'type': 'warm_up'
        })
        total_duration += 10
        
        # Main workout
        for exercise_name in exercises:
            if total_duration >= profile.max_workout_duration - 10:
                break
            
            exercise = self.exercise_db.get(exercise_name)
            if not exercise:
                continue
            
            # Check if exercise fits workout type
            if self._exercise_fits_workout_type(exercise, workout_type):
                # Check equipment availability
                if all(eq in profile.available_equipment or not eq 
                      for eq in exercise.equipment):
                    # Add exercise with appropriate volume
                    volume = self._calculate_volume(profile, exercise, week)
                    selected_exercises.append({
                        'name': exercise.name,
                        'sets': volume['sets'],
                        'reps': volume['reps'],
                        'rest': volume['rest'],
                        'intensity': volume['intensity'],
                        'type': 'main'
                    })
                    total_duration += volume['duration']
        
        # Cool-down
        selected_exercises.append({
            'name': 'Stretching & Cool-down',
            'duration': 10,
            'type': 'cool_down'
        })
        total_duration += 10
        
        return {
            'day': day + 1,
            'type': workout_type,
            'exercises': selected_exercises,
            'total_duration': total_duration,
            'estimated_calories': self._estimate_calories(selected_exercises, profile)
        }
    
    def _exercise_fits_workout_type(self, exercise: Exercise, 
                                   workout_type: str) -> bool:
        """Check if exercise fits the workout type"""
        if workout_type == 'upper_body':
            return any(mg in [MuscleGroup.CHEST, MuscleGroup.BACK, 
                            MuscleGroup.SHOULDERS, MuscleGroup.ARMS] 
                      for mg in exercise.muscle_groups)
        elif workout_type == 'lower_body':
            return any(mg in [MuscleGroup.LEGS, MuscleGroup.GLUTES] 
                      for mg in exercise.muscle_groups)
        elif workout_type == 'full_body':
            return MuscleGroup.FULL_BODY in exercise.muscle_groups or len(exercise.muscle_groups) > 2
        elif workout_type == 'cardio':
            return exercise.movement_pattern == 'cardio'
        else:
            return exercise.difficulty <= 5
    
    def _calculate_volume(self, profile: UserProfile, 
                         exercise: Exercise, week: int) -> Dict[str, Any]:
        """Calculate appropriate volume for exercise"""
        
        # Base volume based on experience
        base_sets = {
            ExperienceLevel.BEGINNER: 3,
            ExperienceLevel.INTERMEDIATE: 4,
            ExperienceLevel.ADVANCED: 5,
            ExperienceLevel.ELITE: 6
        }
        
        sets = base_sets[profile.experience_level]
        
        # Adjust for week (progressive overload)
        sets = min(sets + week // 2, 8)
        
        # Calculate reps based on goals
        if 'strength' in profile.goals:
            reps = np.random.randint(3, 6)
            rest = 180  # seconds
            intensity = 0.85 + week * 0.02
        elif 'muscle_gain' in profile.goals:
            reps = np.random.randint(8, 12)
            rest = 90
            intensity = 0.75 + week * 0.02
        elif 'endurance' in profile.goals:
            reps = np.random.randint(15, 20)
            rest = 45
            intensity = 0.60 + week * 0.01
        else:
            reps = np.random.randint(10, 15)
            rest = 60
            intensity = 0.70 + week * 0.015
        
        # Calculate duration
        duration = sets * (reps * 3 + rest) / 60  # minutes
        
        return {
            'sets': sets,
            'reps': reps,
            'rest': rest,
            'intensity': min(intensity, 0.95),
            'duration': duration
        }
    
    def _optimize_workout_plan(self, plan: List[Dict[str, Any]], 
                              profile: UserProfile) -> List[Dict[str, Any]]:
        """Optimize workout plan using RL"""
        
        optimized_plan = []
        
        for week_plan in plan:
            state = self._encode_user_profile(profile)
            state_tensor = torch.FloatTensor(state)
            
            optimized_week = {
                'week': week_plan['week'],
                'workouts': []
            }
            
            for workout in week_plan['workouts']:
                # Get action from optimizer
                action = self.optimizer.select_action(state_tensor)
                
                # Apply optimization action
                optimized_workout = self._apply_optimization(workout, action)
                optimized_week['workouts'].append(optimized_workout)
                
                # Update state based on workout
                state = self._update_state(state, optimized_workout)
                state_tensor = torch.FloatTensor(state)
            
            optimized_plan.append(optimized_week)
        
        return optimized_plan
    
    def _apply_optimization(self, workout: Dict[str, Any], 
                           action: int) -> Dict[str, Any]:
        """Apply optimization action to workout"""
        
        # Define optimization actions
        if action < 20:  # Adjust volume
            factor = 0.8 + (action / 20) * 0.4
            for exercise in workout['exercises']:
                if 'sets' in exercise:
                    exercise['sets'] = max(1, int(exercise['sets'] * factor))
        elif action < 40:  # Adjust intensity
            factor = 0.7 + ((action - 20) / 20) * 0.3
            for exercise in workout['exercises']:
                if 'intensity' in exercise:
                    exercise['intensity'] = min(1.0, exercise['intensity'] * factor)
        elif action < 60:  # Adjust rest periods
            factor = 0.5 + ((action - 40) / 20) * 1.0
            for exercise in workout['exercises']:
                if 'rest' in exercise:
                    exercise['rest'] = int(exercise['rest'] * factor)
        elif action < 80:  # Add/remove exercises
            if action < 70 and len(workout['exercises']) > 5:
                # Remove an exercise
                workout['exercises'].pop(np.random.randint(1, len(workout['exercises'])-1))
            elif len(workout['exercises']) < 12:
                # Add an exercise
                new_exercise = {
                    'name': 'Additional Exercise',
                    'sets': 3,
                    'reps': 12,
                    'rest': 60,
                    'intensity': 0.7,
                    'type': 'main'
                }
                workout['exercises'].insert(-1, new_exercise)
        
        # Recalculate duration and calories
        workout['total_duration'] = sum(
            ex.get('duration', ex.get('sets', 3) * ex.get('reps', 10) * 3 / 60 + ex.get('rest', 60) / 60)
            for ex in workout['exercises']
        )
        
        return workout
    
    def _update_state(self, state: np.ndarray, 
                     workout: Dict[str, Any]) -> np.ndarray:
        """Update state based on completed workout"""
        new_state = state.copy()
        
        # Update fatigue indicator
        new_state[0] = min(1.0, new_state[0] + workout['total_duration'] / 200)
        
        # Update workout count
        new_state[1] = min(1.0, new_state[1] + 1/30)
        
        return new_state
    
    def _add_progression(self, plan: List[Dict[str, Any]], 
                        profile: UserProfile) -> List[Dict[str, Any]]:
        """Add progressive overload to the plan"""
        
        for week_idx, week_plan in enumerate(plan):
            progression_factor = 1.0 + (week_idx * 0.05)
            
            for workout in week_plan['workouts']:
                for exercise in workout['exercises']:
                    if 'sets' in exercise:
                        # Increase volume progressively
                        if week_idx % 2 == 0:
                            exercise['sets'] = min(8, int(exercise['sets'] * progression_factor))
                        else:
                            exercise['reps'] = min(20, int(exercise.get('reps', 10) * progression_factor))
                    
                    if 'intensity' in exercise:
                        # Increase intensity
                        exercise['intensity'] = min(0.95, exercise['intensity'] * (1 + week_idx * 0.02))
        
        return plan
    
    def _estimate_calories(self, exercises: List[Dict[str, Any]], 
                          profile: UserProfile) -> float:
        """Estimate calories burned in workout"""
        total_calories = 0
        
        for exercise in exercises:
            if exercise['type'] == 'warm_up':
                calories = 3 * exercise.get('duration', 10)
            elif exercise['type'] == 'cool_down':
                calories = 2 * exercise.get('duration', 10)
            else:
                # Get exercise from database
                ex_name = exercise['name'].lower().replace(' ', '_')
                ex_data = self.exercise_db.get(ex_name)
                
                if ex_data:
                    duration = exercise.get('sets', 3) * exercise.get('reps', 10) * 3 / 60
                    calories = ex_data.calories_per_minute * duration
                else:
                    calories = 5 * exercise.get('sets', 3) * exercise.get('reps', 10) / 10
            
            # Adjust for body weight
            weight_factor = profile.weight / 70  # Normalize to 70kg
            total_calories += calories * weight_factor
        
        return total_calories
    
    def _estimate_results(self, profile: UserProfile, 
                         plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate expected results from the plan"""
        
        total_workouts = sum(len(week['workouts']) for week in plan)
        total_calories = sum(
            workout['estimated_calories'] 
            for week in plan 
            for workout in week['workouts']
        )
        
        # Calculate expected changes
        weeks = len(plan)
        
        # Weight change (assuming proper nutrition)
        if 'weight_loss' in profile.goals:
            weight_change = -0.5 * weeks  # kg per week
        elif 'muscle_gain' in profile.goals:
            weight_change = 0.25 * weeks
        else:
            weight_change = 0
        
        # Strength gains (percentage)
        strength_gain = min(50, weeks * 2.5 * (1 if profile.experience_level == ExperienceLevel.BEGINNER else 0.5))
        
        # Endurance improvement (VO2 max percentage)
        cardio_workouts = sum(
            1 for week in plan 
            for workout in week['workouts'] 
            if workout['type'] == 'cardio'
        )
        endurance_gain = min(20, cardio_workouts * 0.5)
        
        # Body composition change
        body_fat_change = -1.5 * weeks if 'weight_loss' in profile.goals else -0.5 * weeks
        
        return {
            'estimated_weight_change': f"{weight_change:+.1f} kg",
            'estimated_strength_gain': f"{strength_gain:.0f}%",
            'estimated_endurance_improvement': f"{endurance_gain:.0f}%",
            'estimated_body_fat_change': f"{body_fat_change:+.1f}%",
            'total_workouts': total_workouts,
            'total_calories_burned': f"{total_calories:.0f}",
            'confidence_level': 'High' if profile.experience_level != ExperienceLevel.BEGINNER else 'Moderate'
        }
    
    def _generate_nutrition_guidelines(self, profile: UserProfile) -> Dict[str, Any]:
        """Generate nutrition guidelines based on goals"""
        
        # Calculate BMR
        if profile.body_fat:
            # Katch-McArdle formula
            lean_mass = profile.weight * (1 - profile.body_fat / 100)
            bmr = 370 + (21.6 * lean_mass)
        else:
            # Mifflin-St Jeor formula
            bmr = 10 * profile.weight + 6.25 * profile.height - 5 * profile.age
            bmr += 5  # Assuming male, -161 for female
        
        # Activity factor
        tdee = bmr * (1.2 + profile.workout_frequency * 0.1)
        
        # Adjust for goals
        if 'weight_loss' in profile.goals:
            target_calories = tdee - 500
            protein_ratio = 0.35
            carb_ratio = 0.35
            fat_ratio = 0.30
        elif 'muscle_gain' in profile.goals:
            target_calories = tdee + 300
            protein_ratio = 0.30
            carb_ratio = 0.45
            fat_ratio = 0.25
        else:
            target_calories = tdee
            protein_ratio = 0.30
            carb_ratio = 0.40
            fat_ratio = 0.30
        
        return {
            'daily_calories': f"{target_calories:.0f}",
            'protein_grams': f"{(target_calories * protein_ratio) / 4:.0f}g",
            'carb_grams': f"{(target_calories * carb_ratio) / 4:.0f}g",
            'fat_grams': f"{(target_calories * fat_ratio) / 9:.0f}g",
            'water_intake': f"{profile.weight * 35:.0f}ml",
            'meal_timing': {
                'pre_workout': 'Carbs + moderate protein 1-2 hours before',
                'post_workout': 'Protein + carbs within 30 minutes',
                'general': '4-5 balanced meals throughout the day'
            }
        }
    
    def _create_recovery_protocol(self, profile: UserProfile) -> Dict[str, Any]:
        """Create recovery protocol"""
        
        recovery_days = max(1, 7 - profile.workout_frequency)
        
        return {
            'rest_days_per_week': recovery_days,
            'sleep_recommendation': '7-9 hours per night',
            'active_recovery': [
                'Light walking or swimming',
                'Yoga or stretching',
                'Foam rolling',
                'Mobility work'
            ],
            'recovery_techniques': [
                'Ice baths or cold showers',
                'Massage therapy',
                'Proper hydration',
                'Stress management'
            ],
            'supplements': [
                'Protein powder for recovery',
                'Creatine for strength (optional)',
                'Omega-3 for inflammation',
                'Vitamin D for overall health'
            ] if profile.experience_level != ExperienceLevel.BEGINNER else ['Basic multivitamin', 'Protein powder']
        }
    
    def track_workout(self, user_id: str, workout_data: Dict[str, Any]) -> None:
        """Track completed workout"""
        
        session = WorkoutSession(
            date=datetime.now(),
            exercises=workout_data['exercises'],
            duration=workout_data['duration'],
            calories_burned=workout_data.get('calories', 0),
            avg_heart_rate=workout_data.get('heart_rate'),
            perceived_effort=workout_data.get('effort'),
            notes=workout_data.get('notes')
        )
        
        if user_id not in self.workout_history:
            self.workout_history[user_id] = []
        
        self.workout_history[user_id].append(session)
        
        # Update progress metrics
        self._update_progress_metrics(user_id, session)
    
    def _update_progress_metrics(self, user_id: str, 
                                 session: WorkoutSession) -> None:
        """Update user's progress metrics"""
        
        if user_id not in self.performance_metrics:
            self.performance_metrics[user_id] = ProgressMetrics(
                strength_gains={},
                endurance_improvement=0,
                weight_change=0,
                body_composition_change=None,
                consistency_score=0,
                recovery_rate=0,
                performance_index=0
            )
        
        metrics = self.performance_metrics[user_id]
        
        # Update consistency score
        history = self.workout_history[user_id]
        if len(history) > 1:
            days_between = (session.date - history[-2].date).days
            if days_between <= 3:
                metrics.consistency_score = min(1.0, metrics.consistency_score + 0.05)
            else:
                metrics.consistency_score = max(0, metrics.consistency_score - 0.02)
        
        # Calculate performance index
        metrics.performance_index = self._calculate_performance_index(history)
    
    def _calculate_performance_index(self, history: List[WorkoutSession]) -> float:
        """Calculate overall performance index"""
        if len(history) < 2:
            return 0.5
        
        # Analyze recent vs past performance
        recent = history[-5:] if len(history) >= 5 else history
        past = history[-10:-5] if len(history) >= 10 else history[:len(history)//2]
        
        if not past:
            return 0.5
        
        recent_avg_duration = np.mean([s.duration for s in recent])
        past_avg_duration = np.mean([s.duration for s in past])
        
        recent_avg_effort = np.mean([s.perceived_effort for s in recent if s.perceived_effort])
        past_avg_effort = np.mean([s.perceived_effort for s in past if s.perceived_effort])
        
        # Performance index (0-1)
        duration_improvement = min(1.0, recent_avg_duration / past_avg_duration) if past_avg_duration > 0 else 0.5
        effort_consistency = 1.0 - abs(recent_avg_effort - past_avg_effort) / 10 if recent_avg_effort and past_avg_effort else 0.5
        
        return (duration_improvement + effort_consistency) / 2
    
    def predict_performance(self, user_id: str, 
                           weeks_ahead: int = 4) -> Dict[str, Any]:
        """Predict future performance"""
        
        history = self.workout_history.get(user_id, [])
        if len(history) < 5:
            return {
                'prediction': 'Insufficient data',
                'confidence': 'Low',
                'recommendation': 'Continue training consistently for better predictions'
            }
        
        # Prepare data for prediction
        history_tensor = self._encode_workout_history(history[-20:])
        
        # Get predictions from progression tracker
        with torch.no_grad():
            predictions = self.progression_tracker(history_tensor)
        
        # Extract predictions
        performance_metrics = predictions['performance_metrics'][0].numpy()
        fatigue_level = predictions['fatigue_level'][0].item()
        recovery_time = predictions['recovery_time'][0].item()
        
        return {
            'predicted_strength_gain': f"{performance_metrics[0]*10:.1f}%",
            'predicted_endurance_gain': f"{performance_metrics[1]*10:.1f}%",
            'predicted_weight_change': f"{performance_metrics[2]*weeks_ahead:.1f}kg",
            'current_fatigue_level': f"{fatigue_level*100:.0f}%",
            'recommended_recovery': f"{recovery_time:.0f} hours",
            'confidence': 'High' if len(history) > 20 else 'Moderate',
            'recommendations': self._generate_recommendations(performance_metrics, fatigue_level)
        }
    
    def _generate_recommendations(self, metrics: np.ndarray, 
                                 fatigue: float) -> List[str]:
        """Generate training recommendations"""
        recommendations = []
        
        if fatigue > 0.7:
            recommendations.append("Consider taking an extra rest day")
            recommendations.append("Focus on recovery activities")
        elif fatigue < 0.3:
            recommendations.append("You can increase training intensity")
            recommendations.append("Consider adding an extra workout day")
        
        if metrics[0] < 0.2:  # Low strength gain
            recommendations.append("Increase weight or resistance gradually")
            recommendations.append("Focus on progressive overload")
        
        if metrics[1] < 0.2:  # Low endurance gain
            recommendations.append("Add more cardio sessions")
            recommendations.append("Increase workout duration")
        
        return recommendations

# Example usage
if __name__ == "__main__":
    # Initialize system
    workout_system = WorkoutSystemAI()
    
    # Create user profile
    user_data = {
        'user_id': 'user_001',
        'age': 28,
        'weight': 75,
        'height': 175,
        'body_fat': 18,
        'experience': 'intermediate',
        'goals': ['muscle_gain', 'strength'],
        'injuries': [],
        'equipment': ['barbell', 'dumbbell', 'bench', 'pull_up_bar'],
        'frequency': 4,
        'max_duration': 75
    }
    
    profile = workout_system.create_user_profile(user_data)
    
    # Generate workout plan
    plan = workout_system.generate_workout_plan('user_001', num_weeks=4)
    
    print("=== PERSONALIZED WORKOUT PLAN ===")
    print(f"User: {plan['user_id']}")
    print(f"Duration: {plan['plan_duration']}")
    print("\n=== WEEK 1 ===")
    for workout in plan['weekly_schedule'][0]['workouts']:
        print(f"\nDay {workout['day']} - {workout['type'].upper()}")
        print(f"Duration: {workout['total_duration']:.0f} minutes")
        print(f"Calories: {workout['estimated_calories']:.0f}")
        print("Exercises:")
        for ex in workout['exercises'][:5]:  # Show first 5 exercises
            if 'sets' in ex:
                print(f"  - {ex['name']}: {ex['sets']} sets x {ex['reps']} reps @ {ex['intensity']*100:.0f}%")
            else:
                print(f"  - {ex['name']}: {ex.get('duration', 'N/A')} minutes")
    
    print("\n=== EXPECTED RESULTS ===")
    for key, value in plan['estimated_results'].items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    print("\n=== NUTRITION GUIDELINES ===")
    for key, value in plan['nutrition_guidelines'].items():
        if isinstance(value, dict):
            print(f"{key.replace('_', ' ').title()}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key.replace('_', ' ').title()}: {value}")
    
    print("\n=== RECOVERY PROTOCOL ===")
    for key, value in plan['recovery_protocol'].items():
        if isinstance(value, list):
            print(f"{key.replace('_', ' ').title()}:")
            for item in value[:3]:
                print(f"  - {item}")
        else:
            print(f"{key.replace('_', ' ').title()}: {value}")
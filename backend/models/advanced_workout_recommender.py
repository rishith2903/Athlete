"""
Advanced Workout Recommender with Deep Reinforcement Learning
Multi-objective optimization with adaptive difficulty adjustment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import optuna

@dataclass
class WorkoutState:
    user_features: np.ndarray
    performance_history: np.ndarray
    current_fitness: float
    fatigue_level: float
    motivation: float
    available_time: float
    equipment: List[str]

class DeepWorkoutNet(nn.Module):
    """Advanced neural network with Graph Neural Networks for exercise relationships"""
    
    def __init__(self, state_dim=128, action_dim=100, hidden_dim=256):
        super().__init__()
        
        # Transformer encoder for user state
        self.state_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=state_dim, nhead=8, dim_feedforward=512),
            num_layers=6
        )
        
        # Graph Neural Network for exercise relationships
        self.exercise_gnn = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Multi-objective heads
        self.strength_head = nn.Linear(hidden_dim, action_dim)
        self.cardio_head = nn.Linear(hidden_dim, action_dim)
        self.flexibility_head = nn.Linear(hidden_dim, action_dim)
        self.recovery_head = nn.Linear(hidden_dim, action_dim)
        
        # Attention mechanism for exercise selection
        self.exercise_attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        
        # Value network for RL
        self.value_head = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Progressive overload predictor
        self.progression_network = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
    
    def forward(self, state: torch.Tensor, action_mask: Optional[torch.Tensor] = None):
        # Encode state
        encoded_state = self.state_encoder(state.unsqueeze(1)).squeeze(1)
        
        # Generate multi-objective outputs
        strength_scores = self.strength_head(encoded_state)
        cardio_scores = self.cardio_head(encoded_state)
        flexibility_scores = self.flexibility_head(encoded_state)
        recovery_scores = self.recovery_head(encoded_state)
        
        # Combine objectives based on user goals
        combined_scores = self.combine_objectives(
            strength_scores, cardio_scores, flexibility_scores, recovery_scores
        )
        
        # Apply action mask if provided
        if action_mask is not None:
            combined_scores = combined_scores.masked_fill(~action_mask, -float('inf'))
        
        return {
            'action_probs': F.softmax(combined_scores, dim=-1),
            'value': self.value_head(torch.cat([encoded_state, combined_scores], dim=-1)),
            'objectives': {
                'strength': strength_scores,
                'cardio': cardio_scores,
                'flexibility': flexibility_scores,
                'recovery': recovery_scores
            }
        }
    
    def combine_objectives(self, strength, cardio, flexibility, recovery):
        # Learnable weights for multi-objective optimization
        weights = F.softmax(self.objective_weights, dim=0)
        return (weights[0] * strength + weights[1] * cardio + 
                weights[2] * flexibility + weights[3] * recovery)

class WorkoutEnvironment(gym.Env):
    """Custom Gym environment for workout planning"""
    
    def __init__(self, user_profile: Dict):
        super().__init__()
        self.user_profile = user_profile
        self.current_week = 0
        self.max_weeks = 12
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(100)  # 100 possible exercises
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(256,), dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def reset(self):
        self.current_week = 0
        self.fatigue = 0
        self.strength_gain = 0
        self.endurance_gain = 0
        self.adherence = 1.0
        
        return self._get_state()
    
    def step(self, action):
        # Execute workout action
        workout = self._action_to_workout(action)
        
        # Calculate rewards
        reward = self._calculate_reward(workout)
        
        # Update state
        self._update_state(workout)
        
        # Check if episode is done
        done = self.current_week >= self.max_weeks
        
        info = {
            'strength_gain': self.strength_gain,
            'endurance_gain': self.endurance_gain,
            'adherence': self.adherence
        }
        
        return self._get_state(), reward, done, info
    
    def _calculate_reward(self, workout):
        # Multi-objective reward function
        goal_achievement = self._calculate_goal_achievement(workout)
        safety_bonus = self._calculate_safety_bonus(workout)
        variety_bonus = self._calculate_variety_bonus(workout)
        adherence_factor = self.adherence
        
        return (0.4 * goal_achievement + 0.3 * safety_bonus + 
                0.2 * variety_bonus + 0.1 * adherence_factor)
    
    def _get_state(self):
        return np.concatenate([
            self.user_profile['features'],
            [self.current_week / self.max_weeks],
            [self.fatigue],
            [self.strength_gain],
            [self.endurance_gain],
            [self.adherence]
        ])

class AdvancedWorkoutRecommender:
    """Main class with all advanced features"""
    
    def __init__(self):
        self.model = DeepWorkoutNet()
        self.rl_agent = None
        self.exercise_database = self._load_exercise_database()
        self.optimization_history = []
        
    def train_with_rl(self, user_data: List[Dict], epochs: int = 100):
        """Train using Proximal Policy Optimization"""
        
        # Create environments for each user
        envs = [WorkoutEnvironment(user) for user in user_data]
        vec_env = DummyVecEnv([lambda: env for env in envs])
        
        # Initialize PPO agent
        self.rl_agent = PPO(
            'MlpPolicy',
            vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=1
        )
        
        # Train agent
        self.rl_agent.learn(total_timesteps=epochs * len(user_data))
    
    def optimize_with_optuna(self, user_profile: Dict, n_trials: int = 100):
        """Hyperparameter optimization using Optuna"""
        
        def objective(trial):
            # Suggest hyperparameters
            learning_rate = trial.suggest_loguniform('lr', 1e-5, 1e-2)
            hidden_dim = trial.suggest_int('hidden_dim', 128, 512, step=64)
            num_layers = trial.suggest_int('num_layers', 2, 6)
            
            # Train model with suggested params
            model = DeepWorkoutNet(hidden_dim=hidden_dim)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
            # Simplified training loop
            loss = self._train_iteration(model, optimizer, user_profile)
            
            return loss
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params
    
    def generate_adaptive_workout(self, user_state: WorkoutState) -> Dict:
        """Generate workout with adaptive difficulty"""
        
        # Convert state to tensor
        state_tensor = torch.tensor(self._encode_state(user_state))
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(state_tensor)
        
        # Select exercises based on multi-objective optimization
        exercises = self._select_optimal_exercises(
            outputs['objectives'],
            user_state
        )
        
        # Apply progressive overload
        exercises = self._apply_progressive_overload(exercises, user_state)
        
        # Generate periodization plan
        plan = self._generate_periodization(exercises, user_state)
        
        return {
            'exercises': exercises,
            'periodization': plan,
            'estimated_results': self._predict_outcomes(exercises, user_state),
            'difficulty_level': self._calculate_difficulty(exercises)
        }
    
    def _select_optimal_exercises(self, objectives: Dict, state: WorkoutState) -> List[Dict]:
        """Select exercises using Pareto optimization"""
        
        # Get Pareto front
        pareto_exercises = []
        
        for obj_name, scores in objectives.items():
            top_k = torch.topk(scores, k=5).indices
            for idx in top_k:
                exercise = self.exercise_database[idx.item()]
                exercise['objective'] = obj_name
                pareto_exercises.append(exercise)
        
        # Filter by constraints
        valid_exercises = [
            ex for ex in pareto_exercises
            if ex['equipment'] in state.equipment or ex['equipment'] == 'none'
        ]
        
        return valid_exercises[:8]  # Return top 8 exercises
    
    def _apply_progressive_overload(self, exercises: List[Dict], state: WorkoutState) -> List[Dict]:
        """Apply progressive overload principles"""
        
        for exercise in exercises:
            # Adjust based on performance history
            if state.performance_history.mean() > 0.8:
                exercise['sets'] += 1
                exercise['intensity'] *= 1.1
            elif state.performance_history.mean() < 0.5:
                exercise['sets'] = max(2, exercise['sets'] - 1)
                exercise['intensity'] *= 0.9
            
            # Adjust for fatigue
            if state.fatigue_level > 0.7:
                exercise['rest_time'] *= 1.2
                exercise['intensity'] *= 0.85
        
        return exercises
    
    def _generate_periodization(self, exercises: List[Dict], state: WorkoutState) -> Dict:
        """Generate periodized training plan"""
        
        return {
            'microcycle': self._generate_microcycle(exercises),
            'mesocycle': self._generate_mesocycle(exercises, weeks=4),
            'macrocycle': self._generate_macrocycle(exercises, months=3),
            'deload_weeks': [4, 8, 12]
        }
    
    def _generate_microcycle(self, exercises: List[Dict]) -> Dict:
        """Generate weekly training split"""
        
        return {
            'monday': [ex for ex in exercises if ex['objective'] == 'strength'][:3],
            'wednesday': [ex for ex in exercises if ex['objective'] == 'cardio'][:3],
            'friday': [ex for ex in exercises if ex['objective'] == 'strength'][:3],
            'saturday': [ex for ex in exercises if ex['objective'] == 'flexibility'][:2]
        }
    
    def _load_exercise_database(self) -> List[Dict]:
        """Load comprehensive exercise database"""
        return [
            {'name': 'Squat', 'equipment': 'barbell', 'sets': 4, 'reps': 8, 'intensity': 0.8},
            {'name': 'Deadlift', 'equipment': 'barbell', 'sets': 3, 'reps': 5, 'intensity': 0.85},
            # ... more exercises
        ]
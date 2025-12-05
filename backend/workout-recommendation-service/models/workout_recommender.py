"""
Hybrid Workout Recommendation Model
Combines collaborative filtering, content-based filtering, and reinforcement learning
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    """User profile for workout recommendations"""
    user_id: str
    age: int
    gender: str
    fitness_level: str  # beginner, intermediate, advanced
    goals: List[str]  # weight_loss, muscle_gain, endurance, strength, flexibility
    available_equipment: List[str]
    workout_days_per_week: int
    session_duration_minutes: int
    injuries: List[str] = None
    preferences: Dict = None
    
class ExerciseEmbedding(nn.Module):
    """Neural network for exercise embeddings"""
    
    def __init__(self, num_exercises: int, embedding_dim: int = 128):
        super().__init__()
        self.exercise_embedding = nn.Embedding(num_exercises, embedding_dim)
        self.muscle_group_embedding = nn.Embedding(20, 32)  # Major muscle groups
        self.equipment_embedding = nn.Embedding(30, 16)  # Equipment types
        self.difficulty_embedding = nn.Embedding(5, 8)  # Difficulty levels
        
        # Combine embeddings
        self.fc1 = nn.Linear(embedding_dim + 32 + 16 + 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, exercise_id, muscle_group_id, equipment_id, difficulty_id):
        ex_emb = self.exercise_embedding(exercise_id)
        mg_emb = self.muscle_group_embedding(muscle_group_id)
        eq_emb = self.equipment_embedding(equipment_id)
        diff_emb = self.difficulty_embedding(difficulty_id)
        
        combined = torch.cat([ex_emb, mg_emb, eq_emb, diff_emb], dim=-1)
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class UserEmbedding(nn.Module):
    """Neural network for user embeddings"""
    
    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.age_embedding = nn.Linear(1, 16)
        self.gender_embedding = nn.Embedding(3, 8)  # male, female, other
        self.fitness_level_embedding = nn.Embedding(3, 16)  # beginner, intermediate, advanced
        self.goal_embedding = nn.Embedding(10, 32)  # Various fitness goals
        
        self.fc1 = nn.Linear(16 + 8 + 16 + 32, 128)
        self.fc2 = nn.Linear(128, embedding_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, age, gender_id, fitness_level_id, goal_ids):
        age_emb = self.age_embedding(age.unsqueeze(-1))
        gender_emb = self.gender_embedding(gender_id)
        fitness_emb = self.fitness_level_embedding(fitness_level_id)
        
        # Handle multiple goals
        goal_emb = self.goal_embedding(goal_ids).mean(dim=1)
        
        combined = torch.cat([age_emb, gender_emb, fitness_emb, goal_emb], dim=-1)
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class HybridWorkoutRecommender(nn.Module):
    """Hybrid recommendation model for workout plans"""
    
    def __init__(self, num_exercises: int, num_users: int):
        super().__init__()
        self.num_exercises = num_exercises
        self.num_users = num_users
        
        # Embeddings
        self.exercise_encoder = ExerciseEmbedding(num_exercises)
        self.user_encoder = UserEmbedding()
        
        # Collaborative filtering component
        self.user_factors = nn.Embedding(num_users, 64)
        self.exercise_factors = nn.Embedding(num_exercises, 64)
        
        # Attention mechanism for exercise selection
        self.attention = nn.MultiheadAttention(64, num_heads=4)
        
        # Workout plan generator
        self.plan_generator = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_exercises)
        )
        
        # Progressive overload predictor
        self.intensity_predictor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # sets, reps, weight_modifier
        )
        
        self.exercise_database = None
        self.load_exercise_database()
        
    def load_exercise_database(self):
        """Load exercise database with metadata"""
        self.exercise_database = {
            # Strength exercises
            "squat": {
                "id": 0,
                "name": "Barbell Squat",
                "muscle_groups": ["quadriceps", "glutes", "hamstrings"],
                "equipment": "barbell",
                "difficulty": "intermediate",
                "category": "strength",
                "calories_per_minute": 8
            },
            "bench_press": {
                "id": 1,
                "name": "Bench Press",
                "muscle_groups": ["chest", "triceps", "shoulders"],
                "equipment": "barbell",
                "difficulty": "intermediate",
                "category": "strength",
                "calories_per_minute": 6
            },
            "deadlift": {
                "id": 2,
                "name": "Deadlift",
                "muscle_groups": ["back", "glutes", "hamstrings"],
                "equipment": "barbell",
                "difficulty": "advanced",
                "category": "strength",
                "calories_per_minute": 9
            },
            "pushup": {
                "id": 3,
                "name": "Push-up",
                "muscle_groups": ["chest", "triceps", "shoulders"],
                "equipment": "none",
                "difficulty": "beginner",
                "category": "strength",
                "calories_per_minute": 7
            },
            "pullup": {
                "id": 4,
                "name": "Pull-up",
                "muscle_groups": ["back", "biceps"],
                "equipment": "pull_up_bar",
                "difficulty": "intermediate",
                "category": "strength",
                "calories_per_minute": 8
            },
            "plank": {
                "id": 5,
                "name": "Plank",
                "muscle_groups": ["core"],
                "equipment": "none",
                "difficulty": "beginner",
                "category": "strength",
                "calories_per_minute": 4
            },
            "lunges": {
                "id": 6,
                "name": "Lunges",
                "muscle_groups": ["quadriceps", "glutes"],
                "equipment": "none",
                "difficulty": "beginner",
                "category": "strength",
                "calories_per_minute": 6
            },
            # Cardio exercises
            "running": {
                "id": 7,
                "name": "Running",
                "muscle_groups": ["legs", "cardiovascular"],
                "equipment": "none",
                "difficulty": "beginner",
                "category": "cardio",
                "calories_per_minute": 10
            },
            "cycling": {
                "id": 8,
                "name": "Cycling",
                "muscle_groups": ["legs", "cardiovascular"],
                "equipment": "bike",
                "difficulty": "beginner",
                "category": "cardio",
                "calories_per_minute": 8
            },
            "burpees": {
                "id": 9,
                "name": "Burpees",
                "muscle_groups": ["full_body"],
                "equipment": "none",
                "difficulty": "intermediate",
                "category": "cardio",
                "calories_per_minute": 10
            },
            "jumping_jacks": {
                "id": 10,
                "name": "Jumping Jacks",
                "muscle_groups": ["full_body"],
                "equipment": "none",
                "difficulty": "beginner",
                "category": "cardio",
                "calories_per_minute": 8
            },
            # Add more exercises as needed
        }
        
    def forward(self, user_profile: UserProfile, context: Optional[Dict] = None):
        """Generate workout recommendations for a user"""
        
        # Encode user profile
        user_embedding = self.encode_user(user_profile)
        
        # Get collaborative filtering scores
        if hasattr(self, 'user_id_mapping') and user_profile.user_id in self.user_id_mapping:
            user_idx = self.user_id_mapping[user_profile.user_id]
            user_factor = self.user_factors(torch.tensor([user_idx]))
            exercise_scores = torch.matmul(user_factor, self.exercise_factors.weight.T)
        else:
            exercise_scores = torch.zeros(self.num_exercises)
        
        # Get content-based scores
        content_scores = self.get_content_scores(user_embedding, user_profile)
        
        # Combine scores (hybrid approach)
        alpha = 0.6  # Weight for content-based
        final_scores = alpha * content_scores + (1 - alpha) * exercise_scores.squeeze()
        
        # Apply constraints (equipment, injuries, etc.)
        final_scores = self.apply_constraints(final_scores, user_profile)
        
        # Generate workout plan
        workout_plan = self.generate_workout_plan(
            final_scores, 
            user_profile,
            context
        )
        
        return workout_plan
    
    def encode_user(self, user_profile: UserProfile) -> torch.Tensor:
        """Encode user profile into embedding"""
        
        # Map categorical values to IDs
        gender_map = {"male": 0, "female": 1, "other": 2}
        fitness_map = {"beginner": 0, "intermediate": 1, "advanced": 2}
        goal_map = {
            "weight_loss": 0, "muscle_gain": 1, "endurance": 2,
            "strength": 3, "flexibility": 4, "general_fitness": 5
        }
        
        age = torch.tensor([user_profile.age / 100.0], dtype=torch.float32)  # Normalize age
        gender_id = torch.tensor([gender_map.get(user_profile.gender, 2)])
        fitness_id = torch.tensor([fitness_map.get(user_profile.fitness_level, 0)])
        
        goal_ids = torch.tensor([
            goal_map.get(goal, 5) for goal in user_profile.goals
        ])
        
        return self.user_encoder(age, gender_id, fitness_id, goal_ids.unsqueeze(0))
    
    def get_content_scores(self, user_embedding: torch.Tensor, user_profile: UserProfile) -> torch.Tensor:
        """Calculate content-based scores for exercises"""
        scores = []
        
        for exercise_key, exercise_info in self.exercise_database.items():
            # Calculate relevance score based on user goals
            relevance = 0.0
            
            # Goal-based scoring
            if "weight_loss" in user_profile.goals:
                if exercise_info["category"] == "cardio":
                    relevance += 0.4
                if exercise_info["calories_per_minute"] >= 8:
                    relevance += 0.3
            
            if "muscle_gain" in user_profile.goals:
                if exercise_info["category"] == "strength":
                    relevance += 0.5
                if exercise_info["difficulty"] in ["intermediate", "advanced"]:
                    relevance += 0.2
            
            if "endurance" in user_profile.goals:
                if exercise_info["category"] == "cardio":
                    relevance += 0.5
            
            # Fitness level matching
            fitness_match = {
                "beginner": {"beginner": 1.0, "intermediate": 0.3, "advanced": 0.1},
                "intermediate": {"beginner": 0.5, "intermediate": 1.0, "advanced": 0.5},
                "advanced": {"beginner": 0.3, "intermediate": 0.7, "advanced": 1.0}
            }
            relevance += 0.3 * fitness_match[user_profile.fitness_level][exercise_info["difficulty"]]
            
            scores.append(relevance)
        
        return torch.tensor(scores, dtype=torch.float32)
    
    def apply_constraints(self, scores: torch.Tensor, user_profile: UserProfile) -> torch.Tensor:
        """Apply equipment and injury constraints"""
        constrained_scores = scores.clone()
        
        for i, (exercise_key, exercise_info) in enumerate(self.exercise_database.items()):
            # Equipment constraint
            if exercise_info["equipment"] != "none":
                if exercise_info["equipment"] not in user_profile.available_equipment:
                    constrained_scores[i] = -float('inf')
            
            # Injury constraints
            if user_profile.injuries:
                for injury in user_profile.injuries:
                    if injury.lower() in ["knee", "knees"]:
                        if exercise_key in ["squat", "lunges", "running"]:
                            constrained_scores[i] = -float('inf')
                    elif injury.lower() in ["back", "lower back"]:
                        if exercise_key in ["deadlift", "squat"]:
                            constrained_scores[i] = -float('inf')
                    elif injury.lower() in ["shoulder", "shoulders"]:
                        if exercise_key in ["bench_press", "pullup"]:
                            constrained_scores[i] = -float('inf')
        
        return constrained_scores
    
    def generate_workout_plan(
        self, 
        scores: torch.Tensor, 
        user_profile: UserProfile,
        context: Optional[Dict] = None
    ) -> Dict:
        """Generate complete workout plan"""
        
        # Determine workout structure based on goals and fitness level
        workout_templates = self.get_workout_templates(user_profile)
        
        # Generate weekly plan
        weekly_plan = {}
        workout_days = min(user_profile.workout_days_per_week, 7)
        
        for day_idx in range(workout_days):
            day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][day_idx]
            template = workout_templates[day_idx % len(workout_templates)]
            
            day_workout = self.generate_day_workout(
                scores,
                template,
                user_profile,
                context
            )
            
            weekly_plan[day_name] = day_workout
        
        # Add rest days
        rest_days = 7 - workout_days
        rest_day_names = ["Sunday", "Saturday", "Friday", "Thursday", "Wednesday", "Tuesday", "Monday"]
        for i in range(rest_days):
            weekly_plan[rest_day_names[i]] = {"type": "rest", "notes": "Active recovery recommended"}
        
        return {
            "user_id": user_profile.user_id,
            "generated_at": datetime.utcnow().isoformat(),
            "program_duration_weeks": 4,
            "weekly_plan": weekly_plan,
            "progression_strategy": self.get_progression_strategy(user_profile),
            "notes": self.generate_notes(user_profile)
        }
    
    def get_workout_templates(self, user_profile: UserProfile) -> List[Dict]:
        """Get workout templates based on user profile"""
        
        if "muscle_gain" in user_profile.goals:
            if user_profile.fitness_level == "beginner":
                return [
                    {"type": "full_body", "focus": "compound"},
                    {"type": "full_body", "focus": "compound"},
                    {"type": "full_body", "focus": "compound"}
                ]
            elif user_profile.fitness_level == "intermediate":
                return [
                    {"type": "upper", "focus": "push"},
                    {"type": "lower", "focus": "legs"},
                    {"type": "upper", "focus": "pull"},
                    {"type": "lower", "focus": "glutes"},
                ]
            else:  # advanced
                return [
                    {"type": "chest_triceps", "focus": "push"},
                    {"type": "back_biceps", "focus": "pull"},
                    {"type": "legs", "focus": "quad"},
                    {"type": "shoulders", "focus": "delts"},
                    {"type": "legs", "focus": "hamstring"}
                ]
        
        elif "weight_loss" in user_profile.goals:
            return [
                {"type": "circuit", "focus": "full_body"},
                {"type": "cardio", "focus": "hiit"},
                {"type": "circuit", "focus": "full_body"},
                {"type": "cardio", "focus": "steady"},
            ]
        
        elif "endurance" in user_profile.goals:
            return [
                {"type": "cardio", "focus": "long"},
                {"type": "circuit", "focus": "endurance"},
                {"type": "cardio", "focus": "intervals"},
                {"type": "strength", "focus": "light"}
            ]
        
        else:  # General fitness
            return [
                {"type": "full_body", "focus": "balanced"},
                {"type": "cardio", "focus": "moderate"},
                {"type": "full_body", "focus": "balanced"}
            ]
    
    def generate_day_workout(
        self,
        scores: torch.Tensor,
        template: Dict,
        user_profile: UserProfile,
        context: Optional[Dict] = None
    ) -> Dict:
        """Generate workout for a specific day"""
        
        workout = {
            "type": template["type"],
            "focus": template["focus"],
            "warmup": self.get_warmup(template["type"]),
            "main_workout": [],
            "cooldown": self.get_cooldown(template["type"]),
            "estimated_duration": user_profile.session_duration_minutes,
            "estimated_calories": 0
        }
        
        # Select exercises based on template and scores
        selected_exercises = self.select_exercises(scores, template, user_profile)
        
        # Generate sets, reps, and rest for each exercise
        total_calories = 0
        for exercise_key in selected_exercises:
            exercise_info = self.exercise_database[exercise_key]
            
            # Determine sets and reps based on goals and fitness level
            sets, reps, rest_seconds = self.get_exercise_parameters(
                exercise_info,
                user_profile,
                template
            )
            
            exercise_data = {
                "exercise": exercise_info["name"],
                "sets": sets,
                "reps": reps,
                "rest_seconds": rest_seconds,
                "muscle_groups": exercise_info["muscle_groups"],
                "notes": self.get_exercise_notes(exercise_info, user_profile)
            }
            
            # Add duration for time-based exercises
            if exercise_key in ["plank", "running", "cycling"]:
                exercise_data["duration"] = f"{reps}s" if exercise_key == "plank" else f"{reps} minutes"
                exercise_data.pop("reps", None)
            
            workout["main_workout"].append(exercise_data)
            
            # Calculate calories
            exercise_minutes = (sets * (reps/60 if exercise_key not in ["running", "cycling"] else reps))
            total_calories += exercise_info["calories_per_minute"] * exercise_minutes
        
        workout["estimated_calories"] = int(total_calories)
        
        return workout
    
    def select_exercises(
        self,
        scores: torch.Tensor,
        template: Dict,
        user_profile: UserProfile
    ) -> List[str]:
        """Select exercises based on scores and template"""
        
        exercise_keys = list(self.exercise_database.keys())
        
        # Filter exercises by template type
        filtered_indices = []
        for i, exercise_key in enumerate(exercise_keys):
            exercise = self.exercise_database[exercise_key]
            
            if template["type"] == "cardio":
                if exercise["category"] == "cardio":
                    filtered_indices.append(i)
            elif template["type"] == "full_body":
                if exercise["category"] == "strength":
                    filtered_indices.append(i)
            elif template["type"] == "upper":
                if any(mg in exercise["muscle_groups"] for mg in ["chest", "back", "shoulders", "biceps", "triceps"]):
                    filtered_indices.append(i)
            elif template["type"] == "lower":
                if any(mg in exercise["muscle_groups"] for mg in ["quadriceps", "hamstrings", "glutes", "calves"]):
                    filtered_indices.append(i)
            elif template["type"] == "circuit":
                filtered_indices.append(i)  # All exercises valid for circuit
        
        # Get top scoring exercises from filtered set
        if filtered_indices:
            filtered_scores = scores[filtered_indices]
            valid_indices = filtered_scores != -float('inf')
            
            if valid_indices.any():
                valid_filtered_indices = [filtered_indices[i] for i in range(len(filtered_indices)) if valid_indices[i]]
                valid_filtered_scores = filtered_scores[valid_indices]
                
                # Select 4-6 exercises based on workout duration
                num_exercises = min(
                    max(4, user_profile.session_duration_minutes // 10),
                    len(valid_filtered_indices)
                )
                
                top_indices = valid_filtered_scores.argsort(descending=True)[:num_exercises]
                selected_indices = [valid_filtered_indices[i] for i in top_indices]
                
                return [exercise_keys[i] for i in selected_indices]
        
        # Fallback to basic exercises
        return ["pushup", "plank", "lunges", "jumping_jacks"][:4]
    
    def get_exercise_parameters(
        self,
        exercise_info: Dict,
        user_profile: UserProfile,
        template: Dict
    ) -> Tuple[int, int, int]:
        """Determine sets, reps, and rest for an exercise"""
        
        # Base parameters by fitness level
        params = {
            "beginner": {
                "strength": (3, 10, 60),
                "cardio": (1, 20, 45),
                "circuit": (2, 12, 30)
            },
            "intermediate": {
                "strength": (4, 12, 45),
                "cardio": (1, 30, 30),
                "circuit": (3, 15, 30)
            },
            "advanced": {
                "strength": (5, 15, 30),
                "cardio": (1, 45, 20),
                "circuit": (4, 20, 20)
            }
        }
        
        category = exercise_info["category"]
        if template["type"] == "circuit":
            category = "circuit"
        
        sets, reps, rest = params[user_profile.fitness_level][category]
        
        # Adjust for goals
        if "muscle_gain" in user_profile.goals and category == "strength":
            sets += 1
            reps = max(8, reps - 2)  # Lower reps, higher weight
            rest += 15
        elif "weight_loss" in user_profile.goals:
            rest = max(20, rest - 10)  # Shorter rest for higher intensity
        elif "endurance" in user_profile.goals:
            reps = int(reps * 1.5)  # More reps
            sets = max(2, sets - 1)
        
        return sets, reps, rest
    
    def get_warmup(self, workout_type: str) -> List[Dict]:
        """Generate warmup routine"""
        if workout_type in ["cardio", "circuit"]:
            return [
                {"exercise": "Light Jogging", "duration": "3 minutes"},
                {"exercise": "Dynamic Stretching", "duration": "5 minutes"},
                {"exercise": "Jumping Jacks", "duration": "2 minutes"}
            ]
        else:
            return [
                {"exercise": "Arm Circles", "duration": "1 minute"},
                {"exercise": "Leg Swings", "duration": "1 minute"},
                {"exercise": "Dynamic Stretching", "duration": "5 minutes"},
                {"exercise": "Light Cardio", "duration": "3 minutes"}
            ]
    
    def get_cooldown(self, workout_type: str) -> List[Dict]:
        """Generate cooldown routine"""
        return [
            {"exercise": "Walking", "duration": "3 minutes"},
            {"exercise": "Static Stretching", "duration": "5 minutes"},
            {"exercise": "Deep Breathing", "duration": "2 minutes"}
        ]
    
    def get_exercise_notes(self, exercise_info: Dict, user_profile: UserProfile) -> str:
        """Generate exercise-specific notes"""
        notes = []
        
        if user_profile.fitness_level == "beginner":
            notes.append("Focus on proper form over weight/speed")
        
        if exercise_info["difficulty"] == "advanced" and user_profile.fitness_level != "advanced":
            notes.append("Consider using assisted variation or lighter weight")
        
        if exercise_info["equipment"] != "none":
            notes.append(f"Equipment needed: {exercise_info['equipment']}")
        
        return ". ".join(notes) if notes else "Maintain good form throughout"
    
    def get_progression_strategy(self, user_profile: UserProfile) -> Dict:
        """Generate progression strategy for the program"""
        return {
            "week_1": "Establish baseline, focus on form",
            "week_2": "Increase intensity by 5-10%",
            "week_3": "Add 1 set or increase reps by 2-3",
            "week_4": "Deload week - reduce volume by 20%",
            "next_cycle": "Reassess and adjust based on progress"
        }
    
    def generate_notes(self, user_profile: UserProfile) -> List[str]:
        """Generate general workout notes"""
        notes = [
            "Always warm up before and cool down after workouts",
            "Stay hydrated throughout your session",
            "Listen to your body and rest if needed"
        ]
        
        if "weight_loss" in user_profile.goals:
            notes.append("Combine with a caloric deficit for optimal results")
        
        if "muscle_gain" in user_profile.goals:
            notes.append("Ensure adequate protein intake (0.8-1g per lb body weight)")
        
        if user_profile.fitness_level == "beginner":
            notes.append("Start with lighter weights and progress gradually")
        
        return notes

class ReinforcementLearningAdapter:
    """Reinforcement learning component for adaptive recommendations"""
    
    def __init__(self, state_dim: int = 64, action_dim: int = 100):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = []
        self.epsilon = 0.1  # Exploration rate
        
        # Q-network for exercise selection
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        
    def update(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor):
        """Update Q-network based on user feedback"""
        
        # Store experience
        self.memory.append((state, action, reward, next_state))
        
        # Sample batch from memory if enough experiences
        if len(self.memory) >= 32:
            batch = self.sample_batch(32)
            self.train_on_batch(batch)
    
    def sample_batch(self, batch_size: int):
        """Sample batch from memory"""
        import random
        return random.sample(self.memory, min(batch_size, len(self.memory)))
    
    def train_on_batch(self, batch):
        """Train Q-network on batch"""
        states = torch.stack([exp[0] for exp in batch])
        actions = torch.tensor([exp[1] for exp in batch])
        rewards = torch.tensor([exp[2] for exp in batch])
        next_states = torch.stack([exp[3] for exp in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.q_network(next_states).max(1)[0].detach()
        target_q_values = rewards + 0.99 * next_q_values  # Discount factor = 0.99
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def get_adaptive_scores(self, state: torch.Tensor) -> torch.Tensor:
        """Get adaptive scores based on learned preferences"""
        with torch.no_grad():
            return self.q_network(state)
    
    def calculate_reward(self, user_feedback: Dict) -> float:
        """Calculate reward from user feedback"""
        reward = 0.0
        
        # Positive signals
        if user_feedback.get("completed", False):
            reward += 1.0
        if user_feedback.get("rating", 0) > 3:
            reward += (user_feedback["rating"] - 3) * 0.5
        if user_feedback.get("felt_good", False):
            reward += 0.5
        
        # Negative signals
        if user_feedback.get("too_hard", False):
            reward -= 0.5
        if user_feedback.get("too_easy", False):
            reward -= 0.3
        if user_feedback.get("injury", False):
            reward -= 2.0
        if not user_feedback.get("completed", True):
            reward -= 0.5
        
        return reward
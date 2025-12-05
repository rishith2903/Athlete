"""
Training Pipeline for Workout Recommendation Model
Processes fitness datasets and trains the hybrid recommendation system
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from tqdm import tqdm
import logging
import json
from datetime import datetime
from typing import Dict, List, Tuple
import mlflow
import mlflow.pytorch

from models.workout_recommender import (
    HybridWorkoutRecommender,
    UserProfile,
    ReinforcementLearningAdapter
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkoutDataset(Dataset):
    """Dataset for workout recommendations"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.users = []
        self.workouts = []
        self.interactions = []
        self.load_data()
        
    def load_data(self):
        """Load and preprocess fitness dataset"""
        try:
            # Load synthetic or real fitness data
            if os.path.exists(f"{self.data_path}/users.csv"):
                self.users = pd.read_csv(f"{self.data_path}/users.csv")
            else:
                self.users = self.generate_synthetic_users(1000)
                
            if os.path.exists(f"{self.data_path}/exercises.csv"):
                self.exercises = pd.read_csv(f"{self.data_path}/exercises.csv")
            else:
                self.exercises = self.generate_synthetic_exercises()
                
            if os.path.exists(f"{self.data_path}/interactions.csv"):
                self.interactions = pd.read_csv(f"{self.data_path}/interactions.csv")
            else:
                self.interactions = self.generate_synthetic_interactions()
                
            logger.info(f"Loaded {len(self.users)} users, {len(self.exercises)} exercises")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            # Generate synthetic data as fallback
            self.users = self.generate_synthetic_users(1000)
            self.exercises = self.generate_synthetic_exercises()
            self.interactions = self.generate_synthetic_interactions()
    
    def generate_synthetic_users(self, num_users: int) -> pd.DataFrame:
        """Generate synthetic user data"""
        np.random.seed(42)
        
        users = []
        fitness_levels = ["beginner", "intermediate", "advanced"]
        goals = ["weight_loss", "muscle_gain", "endurance", "strength", "general_fitness"]
        genders = ["male", "female", "other"]
        equipment = ["none", "dumbbells", "barbell", "resistance_bands", "pull_up_bar", "kettlebell"]
        
        for i in range(num_users):
            user = {
                "user_id": f"user_{i}",
                "age": np.random.randint(18, 65),
                "gender": np.random.choice(genders),
                "fitness_level": np.random.choice(fitness_levels),
                "primary_goal": np.random.choice(goals),
                "secondary_goal": np.random.choice(goals),
                "workout_days_per_week": np.random.randint(2, 7),
                "session_duration": np.random.choice([30, 45, 60, 90]),
                "available_equipment": ",".join(np.random.choice(equipment, 
                                                                size=np.random.randint(1, 4), 
                                                                replace=False)),
                "has_injuries": np.random.choice([0, 1], p=[0.8, 0.2])
            }
            users.append(user)
            
        return pd.DataFrame(users)
    
    def generate_synthetic_exercises(self) -> pd.DataFrame:
        """Generate synthetic exercise database"""
        
        exercises = [
            # Strength exercises
            {"exercise_id": 0, "name": "Barbell Squat", "category": "strength", 
             "muscle_groups": "quadriceps,glutes,hamstrings", "equipment": "barbell", 
             "difficulty": "intermediate", "calories_per_minute": 8},
            {"exercise_id": 1, "name": "Bench Press", "category": "strength",
             "muscle_groups": "chest,triceps,shoulders", "equipment": "barbell",
             "difficulty": "intermediate", "calories_per_minute": 6},
            {"exercise_id": 2, "name": "Deadlift", "category": "strength",
             "muscle_groups": "back,glutes,hamstrings", "equipment": "barbell",
             "difficulty": "advanced", "calories_per_minute": 9},
            {"exercise_id": 3, "name": "Push-up", "category": "strength",
             "muscle_groups": "chest,triceps,shoulders", "equipment": "none",
             "difficulty": "beginner", "calories_per_minute": 7},
            {"exercise_id": 4, "name": "Pull-up", "category": "strength",
             "muscle_groups": "back,biceps", "equipment": "pull_up_bar",
             "difficulty": "intermediate", "calories_per_minute": 8},
            {"exercise_id": 5, "name": "Plank", "category": "strength",
             "muscle_groups": "core", "equipment": "none",
             "difficulty": "beginner", "calories_per_minute": 4},
            {"exercise_id": 6, "name": "Lunges", "category": "strength",
             "muscle_groups": "quadriceps,glutes", "equipment": "none",
             "difficulty": "beginner", "calories_per_minute": 6},
            {"exercise_id": 7, "name": "Dumbbell Curl", "category": "strength",
             "muscle_groups": "biceps", "equipment": "dumbbells",
             "difficulty": "beginner", "calories_per_minute": 4},
            {"exercise_id": 8, "name": "Shoulder Press", "category": "strength",
             "muscle_groups": "shoulders,triceps", "equipment": "dumbbells",
             "difficulty": "intermediate", "calories_per_minute": 5},
            {"exercise_id": 9, "name": "Lat Pulldown", "category": "strength",
             "muscle_groups": "back,biceps", "equipment": "machine",
             "difficulty": "beginner", "calories_per_minute": 5},
            # Cardio exercises
            {"exercise_id": 10, "name": "Running", "category": "cardio",
             "muscle_groups": "legs,cardiovascular", "equipment": "none",
             "difficulty": "beginner", "calories_per_minute": 10},
            {"exercise_id": 11, "name": "Cycling", "category": "cardio",
             "muscle_groups": "legs,cardiovascular", "equipment": "bike",
             "difficulty": "beginner", "calories_per_minute": 8},
            {"exercise_id": 12, "name": "Burpees", "category": "cardio",
             "muscle_groups": "full_body", "equipment": "none",
             "difficulty": "intermediate", "calories_per_minute": 10},
            {"exercise_id": 13, "name": "Jumping Jacks", "category": "cardio",
             "muscle_groups": "full_body", "equipment": "none",
             "difficulty": "beginner", "calories_per_minute": 8},
            {"exercise_id": 14, "name": "Mountain Climbers", "category": "cardio",
             "muscle_groups": "core,legs", "equipment": "none",
             "difficulty": "intermediate", "calories_per_minute": 9},
            {"exercise_id": 15, "name": "Rowing", "category": "cardio",
             "muscle_groups": "back,legs,cardiovascular", "equipment": "rowing_machine",
             "difficulty": "intermediate", "calories_per_minute": 9},
            # Flexibility exercises
            {"exercise_id": 16, "name": "Yoga Flow", "category": "flexibility",
             "muscle_groups": "full_body", "equipment": "yoga_mat",
             "difficulty": "beginner", "calories_per_minute": 3},
            {"exercise_id": 17, "name": "Static Stretching", "category": "flexibility",
             "muscle_groups": "full_body", "equipment": "none",
             "difficulty": "beginner", "calories_per_minute": 2},
        ]
        
        return pd.DataFrame(exercises)
    
    def generate_synthetic_interactions(self) -> pd.DataFrame:
        """Generate synthetic user-exercise interaction data"""
        np.random.seed(42)
        
        interactions = []
        
        for _, user in self.users.iterrows():
            # Generate workout history for each user
            num_workouts = np.random.randint(10, 100)
            
            for _ in range(num_workouts):
                # Select exercises based on user profile
                if user['fitness_level'] == 'beginner':
                    exercise_pool = self.exercises[
                        self.exercises['difficulty'].isin(['beginner', 'intermediate'])
                    ]
                else:
                    exercise_pool = self.exercises
                
                # Filter by equipment
                user_equipment = user['available_equipment'].split(',')
                exercise_pool = exercise_pool[
                    exercise_pool['equipment'].isin(user_equipment + ['none'])
                ]
                
                if len(exercise_pool) > 0:
                    exercise = exercise_pool.sample(1).iloc[0]
                    
                    # Generate interaction data
                    interaction = {
                        'user_id': user['user_id'],
                        'exercise_id': exercise['exercise_id'],
                        'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 365)),
                        'sets': np.random.randint(2, 5),
                        'reps': np.random.randint(8, 15),
                        'weight': np.random.randint(0, 100) if exercise['equipment'] != 'none' else 0,
                        'duration_minutes': np.random.randint(20, 60),
                        'completed': np.random.choice([0, 1], p=[0.1, 0.9]),
                        'rating': np.random.randint(1, 6),
                        'difficulty_feedback': np.random.choice(['too_easy', 'just_right', 'too_hard'])
                    }
                    interactions.append(interaction)
        
        return pd.DataFrame(interactions)
    
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        interaction = self.interactions.iloc[idx]
        user = self.users[self.users['user_id'] == interaction['user_id']].iloc[0]
        exercise = self.exercises[self.exercises['exercise_id'] == interaction['exercise_id']].iloc[0]
        
        # Create feature vectors
        user_features = self.encode_user_features(user)
        exercise_features = self.encode_exercise_features(exercise)
        
        # Target is the rating
        target = interaction['rating'] / 5.0  # Normalize to [0, 1]
        
        return {
            'user_features': torch.tensor(user_features, dtype=torch.float32),
            'exercise_features': torch.tensor(exercise_features, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32)
        }
    
    def encode_user_features(self, user) -> np.ndarray:
        """Encode user features"""
        features = []
        
        # Numerical features
        features.append(user['age'] / 100.0)
        features.append(user['workout_days_per_week'] / 7.0)
        features.append(user['session_duration'] / 120.0)
        
        # Categorical features (one-hot encoding)
        # Gender
        gender_map = {'male': [1, 0, 0], 'female': [0, 1, 0], 'other': [0, 0, 1]}
        features.extend(gender_map.get(user['gender'], [0, 0, 1]))
        
        # Fitness level
        fitness_map = {
            'beginner': [1, 0, 0],
            'intermediate': [0, 1, 0],
            'advanced': [0, 0, 1]
        }
        features.extend(fitness_map.get(user['fitness_level'], [1, 0, 0]))
        
        # Goals (multi-hot encoding)
        goals = ['weight_loss', 'muscle_gain', 'endurance', 'strength', 'general_fitness']
        goal_vector = [0] * len(goals)
        if user['primary_goal'] in goals:
            goal_vector[goals.index(user['primary_goal'])] = 1
        if user['secondary_goal'] in goals:
            goal_vector[goals.index(user['secondary_goal'])] = 0.5
        features.extend(goal_vector)
        
        return np.array(features)
    
    def encode_exercise_features(self, exercise) -> np.ndarray:
        """Encode exercise features"""
        features = []
        
        # Numerical features
        features.append(exercise['calories_per_minute'] / 10.0)
        
        # Category
        category_map = {
            'strength': [1, 0, 0],
            'cardio': [0, 1, 0],
            'flexibility': [0, 0, 1]
        }
        features.extend(category_map.get(exercise['category'], [0, 0, 0]))
        
        # Difficulty
        difficulty_map = {
            'beginner': [1, 0, 0],
            'intermediate': [0, 1, 0],
            'advanced': [0, 0, 1]
        }
        features.extend(difficulty_map.get(exercise['difficulty'], [1, 0, 0]))
        
        # Equipment (simplified)
        has_equipment = 0 if exercise['equipment'] == 'none' else 1
        features.append(has_equipment)
        
        # Muscle groups (simplified multi-hot)
        muscle_groups = ['chest', 'back', 'legs', 'shoulders', 'arms', 'core', 'full_body']
        muscle_vector = [0] * len(muscle_groups)
        exercise_muscles = exercise['muscle_groups'].lower()
        for i, muscle in enumerate(muscle_groups):
            if muscle in exercise_muscles:
                muscle_vector[i] = 1
        features.extend(muscle_vector)
        
        return np.array(features)

class ModelTrainer:
    """Training orchestrator for the workout recommendation model"""
    
    def __init__(self, data_path: str, model_save_path: str):
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
    def train(self, epochs: int = 50, batch_size: int = 32, learning_rate: float = 0.001):
        """Train the hybrid recommendation model"""
        
        # Initialize MLflow
        mlflow.set_experiment("workout_recommendation_training")
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "device": str(self.device)
            })
            
            # Load dataset
            logger.info("Loading dataset...")
            dataset = WorkoutDataset(self.data_path)
            
            # Split data
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Initialize model
            num_exercises = len(dataset.exercises)
            num_users = len(dataset.users)
            model = HybridWorkoutRecommender(num_exercises, num_users).to(self.device)
            
            # Initialize RL adapter
            rl_adapter = ReinforcementLearningAdapter()
            
            # Optimizer and loss
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            
            # Training loop
            logger.info("Starting training...")
            best_val_loss = float('inf')
            
            for epoch in range(epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                train_batches = 0
                
                progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
                for batch in progress_bar:
                    user_features = batch['user_features'].to(self.device)
                    exercise_features = batch['exercise_features'].to(self.device)
                    targets = batch['target'].to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass (simplified for training)
                    # In practice, you'd use the full model forward method
                    user_embeddings = model.user_encoder(
                        user_features[:, 0],  # age
                        user_features[:, 3:6].argmax(dim=1),  # gender
                        user_features[:, 6:9].argmax(dim=1),  # fitness level
                        user_features[:, 9:14].nonzero()[:, 1].view(-1, 1)  # goals
                    )
                    
                    # Get predictions (simplified)
                    predictions = torch.sigmoid(user_embeddings.mean(dim=1))
                    
                    loss = criterion(predictions, targets)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    train_batches += 1
                    
                    progress_bar.set_postfix({'loss': loss.item()})
                
                avg_train_loss = train_loss / train_batches
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                val_batches = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        user_features = batch['user_features'].to(self.device)
                        exercise_features = batch['exercise_features'].to(self.device)
                        targets = batch['target'].to(self.device)
                        
                        # Forward pass (simplified)
                        user_embeddings = model.user_encoder(
                            user_features[:, 0],
                            user_features[:, 3:6].argmax(dim=1),
                            user_features[:, 6:9].argmax(dim=1),
                            user_features[:, 9:14].nonzero()[:, 1].view(-1, 1)
                        )
                        
                        predictions = torch.sigmoid(user_embeddings.mean(dim=1))
                        loss = criterion(predictions, targets)
                        
                        val_loss += loss.item()
                        val_batches += 1
                
                avg_val_loss = val_loss / val_batches
                
                # Log metrics
                mlflow.log_metrics({
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss
                }, step=epoch)
                
                logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self.save_model(model, rl_adapter, dataset)
                    logger.info(f"Saved best model with val loss: {best_val_loss:.4f}")
            
            # Log final model
            mlflow.pytorch.log_model(model, "model")
            
            logger.info("Training completed!")
            return model, rl_adapter
    
    def save_model(self, model, rl_adapter, dataset):
        """Save trained model and associated data"""
        
        # Create save directory
        os.makedirs(self.model_save_path, exist_ok=True)
        
        # Save model weights
        torch.save({
            'model_state_dict': model.state_dict(),
            'rl_adapter_state_dict': rl_adapter.q_network.state_dict(),
            'num_exercises': len(dataset.exercises),
            'num_users': len(dataset.users),
        }, f"{self.model_save_path}/workout_recommender.pth")
        
        # Save exercise database
        dataset.exercises.to_csv(f"{self.model_save_path}/exercises.csv", index=False)
        
        # Save user encoder mappings
        joblib.dump({
            'user_columns': dataset.users.columns.tolist(),
            'exercise_columns': dataset.exercises.columns.tolist()
        }, f"{self.model_save_path}/feature_mappings.pkl")
        
        # Save metadata
        metadata = {
            'trained_at': datetime.now().isoformat(),
            'num_users': len(dataset.users),
            'num_exercises': len(dataset.exercises),
            'num_interactions': len(dataset.interactions),
            'model_version': '1.0.0'
        }
        
        with open(f"{self.model_save_path}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {self.model_save_path}")

def main():
    """Main training script"""
    
    # Configuration
    data_path = "data/"
    model_save_path = "models/trained/"
    
    # Create directories
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)
    
    # Initialize trainer
    trainer = ModelTrainer(data_path, model_save_path)
    
    # Train model
    model, rl_adapter = trainer.train(
        epochs=50,
        batch_size=32,
        learning_rate=0.001
    )
    
    logger.info("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()
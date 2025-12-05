"""
Training script for Advanced Workout Recommender
Includes data generation, model training, and evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import wandb
import json
from typing import Dict, List, Tuple
import pickle
from advanced_workout_recommender import (
    AdvancedWorkoutRecommender, 
    WorkoutState,
    DeepWorkoutNet
)

class WorkoutDataset(Dataset):
    """Custom dataset for workout recommendations"""
    
    def __init__(self, user_data: List[Dict], transform=None):
        self.user_data = user_data
        self.transform = transform
        self.scaler = StandardScaler()
        
        # Prepare features
        self.features = self._prepare_features()
        self.labels = self._prepare_labels()
        
    def _prepare_features(self):
        features = []
        for user in self.user_data:
            feature_vec = np.concatenate([
                user['demographics'],  # age, gender, height, weight
                user['fitness_metrics'],  # vo2max, strength_level, flexibility
                user['goals'],  # weight_loss, muscle_gain, endurance
                user['preferences'],  # morning/evening, intensity preference
                user['medical_history']  # injuries, conditions
            ])
            features.append(feature_vec)
        
        return self.scaler.fit_transform(np.array(features))
    
    def _prepare_labels(self):
        # Generate optimal workout labels for supervised pre-training
        labels = []
        for user in self.user_data:
            workout_plan = self._generate_optimal_plan(user)
            labels.append(workout_plan)
        return np.array(labels)
    
    def _generate_optimal_plan(self, user: Dict):
        # Rule-based optimal plan generation for pre-training
        plan = np.zeros(100)  # 100 possible exercises
        
        if user['goals']['muscle_gain'] > 0.7:
            plan[0:20] = np.random.uniform(0.5, 1.0, 20)  # Strength exercises
        if user['goals']['weight_loss'] > 0.7:
            plan[20:40] = np.random.uniform(0.5, 1.0, 20)  # Cardio exercises
        if user['goals']['flexibility'] > 0.5:
            plan[40:50] = np.random.uniform(0.3, 0.7, 10)  # Flexibility exercises
            
        return plan
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), \
               torch.tensor(self.labels[idx], dtype=torch.float32)

def generate_synthetic_users(n_users: int = 1000) -> List[Dict]:
    """Generate synthetic user data for training"""
    users = []
    
    for i in range(n_users):
        user = {
            'id': f'user_{i}',
            'demographics': np.array([
                np.random.randint(18, 70),  # age
                np.random.choice([0, 1]),  # gender
                np.random.uniform(150, 200),  # height (cm)
                np.random.uniform(50, 120)  # weight (kg)
            ]),
            'fitness_metrics': np.array([
                np.random.uniform(30, 60),  # vo2max
                np.random.uniform(0, 5),  # strength level (0-5)
                np.random.uniform(0, 10)  # flexibility score
            ]),
            'goals': {
                'weight_loss': np.random.random(),
                'muscle_gain': np.random.random(),
                'endurance': np.random.random(),
                'flexibility': np.random.random()
            },
            'preferences': np.array([
                np.random.choice([0, 1]),  # morning person
                np.random.uniform(0, 1),  # intensity preference
                np.random.uniform(30, 120)  # available time (minutes)
            ]),
            'medical_history': np.random.uniform(0, 1, 5),  # 5 medical flags
            'performance_history': np.random.uniform(0.3, 1.0, 30),  # 30 days
            'features': np.random.randn(256)  # Pre-computed feature vector
        }
        users.append(user)
    
    return users

class AdvancedTrainer:
    """Training coordinator for the advanced recommender"""
    
    def __init__(self, model: DeepWorkoutNet, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        self.best_loss = float('inf')
        
        # Initialize wandb for tracking
        wandb.init(project="advanced-workout-recommender")
        
    def train_supervised(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 50):
        """Supervised pre-training phase"""
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            
            for batch_idx, (features, labels) in enumerate(train_loader):
                features, labels = features.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(features)
                
                # Calculate multi-task loss
                loss = self._calculate_multitask_loss(outputs, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    wandb.log({
                        'batch_loss': loss.item(),
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    })
            
            # Validation phase
            val_loss = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Logging
            avg_train_loss = train_loss / len(train_loader)
            print(f'Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}')
            
            wandb.log({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': val_loss
            })
            
            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(f'best_model_epoch_{epoch}.pt')
    
    def _calculate_multitask_loss(self, outputs: Dict, labels: torch.Tensor) -> torch.Tensor:
        """Calculate combined loss for all objectives"""
        
        # Reconstruction loss for action probabilities
        action_loss = nn.functional.mse_loss(outputs['action_probs'], labels)
        
        # Objective-specific losses
        objective_losses = []
        for obj_name, obj_scores in outputs['objectives'].items():
            obj_loss = nn.functional.mse_loss(obj_scores, labels)
            objective_losses.append(obj_loss)
        
        # Value loss (if available)
        value_loss = torch.tensor(0.0).to(self.device)
        if 'value' in outputs:
            # Estimate value targets from labels
            value_targets = labels.sum(dim=1, keepdim=True)
            value_loss = nn.functional.mse_loss(outputs['value'], value_targets)
        
        # Combine losses with weights
        total_loss = (
            0.4 * action_loss +
            0.3 * sum(objective_losses) / len(objective_losses) +
            0.3 * value_loss
        )
        
        return total_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validation phase"""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                loss = self._calculate_multitask_loss(outputs, labels)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def train_with_reinforcement_learning(self, recommender: AdvancedWorkoutRecommender, 
                                         user_data: List[Dict], epochs: int = 100):
        """Reinforcement learning fine-tuning phase"""
        
        print("Starting RL training phase...")
        recommender.train_with_rl(user_data, epochs)
        
        # Evaluate RL performance
        rewards = self.evaluate_rl_performance(recommender, user_data[:10])
        print(f"Average RL reward: {np.mean(rewards):.4f}")
        
        wandb.log({'rl_average_reward': np.mean(rewards)})
    
    def evaluate_rl_performance(self, recommender: AdvancedWorkoutRecommender, 
                               test_users: List[Dict]) -> List[float]:
        """Evaluate RL agent performance"""
        rewards = []
        
        for user in test_users:
            state = WorkoutState(
                user_features=user['features'],
                performance_history=user['performance_history'],
                current_fitness=np.random.uniform(0.3, 0.8),
                fatigue_level=np.random.uniform(0.1, 0.6),
                motivation=np.random.uniform(0.4, 1.0),
                available_time=user['preferences'][2],
                equipment=['barbell', 'dumbbell', 'none']
            )
            
            workout = recommender.generate_adaptive_workout(state)
            
            # Simulate reward calculation
            reward = self._simulate_workout_reward(workout, user)
            rewards.append(reward)
        
        return rewards
    
    def _simulate_workout_reward(self, workout: Dict, user: Dict) -> float:
        """Simulate reward for a generated workout"""
        
        # Check goal alignment
        goal_alignment = 0
        for goal, weight in user['goals'].items():
            if weight > 0.5:
                # Check if workout addresses this goal
                relevant_exercises = [e for e in workout['exercises'] 
                                    if goal in e.get('targets', [])]
                goal_alignment += len(relevant_exercises) * weight
        
        # Check difficulty appropriateness
        difficulty_score = 1.0 - abs(workout['difficulty_level'] - 0.5)
        
        # Variety score
        unique_exercises = len(set(e['name'] for e in workout['exercises']))
        variety_score = min(unique_exercises / 5, 1.0)
        
        return (0.5 * goal_alignment + 0.3 * difficulty_score + 0.2 * variety_score)
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss
        }, filename)
        print(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_loss = checkpoint['best_loss']
        print(f"Checkpoint loaded: {filename}")

def main():
    """Main training pipeline"""
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate synthetic data
    print("Generating synthetic user data...")
    users = generate_synthetic_users(n_users=5000)
    
    # Split data
    train_users, test_users = train_test_split(users, test_size=0.2, random_state=42)
    train_users, val_users = train_test_split(train_users, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = WorkoutDataset(train_users)
    val_dataset = WorkoutDataset(val_users)
    test_dataset = WorkoutDataset(test_users)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model and trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DeepWorkoutNet(state_dim=256, action_dim=100, hidden_dim=512)
    
    # Add missing objective_weights parameter
    model.objective_weights = nn.Parameter(torch.ones(4))
    
    trainer = AdvancedTrainer(model, device)
    
    # Phase 1: Supervised pre-training
    print("\nPhase 1: Supervised Pre-training")
    print("-" * 40)
    trainer.train_supervised(train_loader, val_loader, epochs=30)
    
    # Phase 2: Hyperparameter optimization
    print("\nPhase 2: Hyperparameter Optimization")
    print("-" * 40)
    recommender = AdvancedWorkoutRecommender()
    recommender.model = model
    best_params = recommender.optimize_with_optuna(train_users[0], n_trials=20)
    print(f"Best hyperparameters: {best_params}")
    
    # Phase 3: Reinforcement learning fine-tuning
    print("\nPhase 3: Reinforcement Learning Fine-tuning")
    print("-" * 40)
    trainer.train_with_reinforcement_learning(recommender, train_users[:100], epochs=50)
    
    # Final evaluation
    print("\nFinal Evaluation")
    print("-" * 40)
    test_loss = trainer.validate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save final model
    trainer.save_checkpoint('final_advanced_recommender.pt')
    
    # Save scaler and other artifacts
    with open('workout_scaler.pkl', 'wb') as f:
        pickle.dump(train_dataset.scaler, f)
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()
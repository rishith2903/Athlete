"""
Training Pipeline for Advanced Nutrition Model
================================================

Multi-task learning pipeline for training the nutrition model with:
- Food recognition from images
- Nutritional content estimation
- Meal generation with constraints
- Preference learning from feedback
- Knowledge graph construction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
import os
import sys
from pathlib import Path
import logging
import wandb
from tqdm import tqdm
import random
from datetime import datetime
import pickle
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image
import networkx as nx

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from models.advanced_nutrition_model import (
    AdvancedNutritionModel,
    create_advanced_nutrition_model
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FoodImageDataset(Dataset):
    """Dataset for food image recognition and nutritional estimation."""
    
    def __init__(self, image_paths: List[str], labels: Dict, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Get labels
        label_data = self.labels[image_path]
        
        return {
            'image': image,
            'food_class': torch.tensor(label_data['food_class'], dtype=torch.long),
            'nutrition': torch.tensor(label_data['nutrition'], dtype=torch.float32),
            'portion_size': torch.tensor(label_data['portion_size'], dtype=torch.float32)
        }

class MealGenerationDataset(Dataset):
    """Dataset for meal generation training."""
    
    def __init__(self, meal_data: List[Dict], tokenizer):
        self.meal_data = meal_data
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.meal_data)
        
    def __getitem__(self, idx):
        meal = self.meal_data[idx]
        
        # Prepare input prompt
        prompt = self._create_prompt(meal['user_profile'], meal['constraints'], meal['meal_type'])
        target = meal['meal_description']
        
        # Tokenize
        inputs = self.tokenizer(prompt, max_length=256, truncation=True, 
                               padding='max_length', return_tensors='pt')
        targets = self.tokenizer(target, max_length=256, truncation=True,
                                padding='max_length', return_tensors='pt')
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze(),
            'nutrition_constraints': torch.tensor(list(meal['constraints'].values()), 
                                                 dtype=torch.float32)
        }
        
    def _create_prompt(self, profile: Dict, constraints: Dict, meal_type: str) -> str:
        """Create input prompt for meal generation."""
        prompt = f"<BOS><{meal_type.upper()}>"
        
        for restriction in profile.get('dietary_restrictions', []):
            prompt += f"<{restriction.upper()}>"
            
        for nutrient, value in constraints.items():
            if nutrient in ['calories', 'protein', 'carbs', 'fat']:
                prompt += f"<{nutrient.upper()}>{value:.1f}"
                
        prompt += " Generate a healthy, delicious meal:"
        return prompt

class NutritionModelTrainer:
    """Advanced training orchestrator for the nutrition model."""
    
    def __init__(self, model: AdvancedNutritionModel, config: Dict):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize optimizers for different components
        self.vision_optimizer = optim.AdamW(
            self.model.vision_model.parameters(),
            lr=config.get('vision_lr', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        self.generator_optimizer = optim.AdamW(
            self.model.meal_generator.parameters(),
            lr=config.get('generator_lr', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        self.graph_optimizer = optim.AdamW(
            self.model.knowledge_graph.parameters(),
            lr=config.get('graph_lr', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
        self.generation_loss = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Initialize wandb
        if config.get('use_wandb', False):
            wandb.init(project='nutrition-model', config=config)
            
    def train_vision_model(self, train_loader: DataLoader, val_loader: DataLoader,
                          epochs: int = 10):
        """Train the food vision recognition model."""
        logger.info("Training vision model...")
        
        for epoch in range(epochs):
            self.model.vision_model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Vision Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                images = batch['image'].to(self.device)
                food_classes = batch['food_class'].to(self.device)
                nutrition = batch['nutrition'].to(self.device)
                portion_sizes = batch['portion_size'].to(self.device)
                
                self.vision_optimizer.zero_grad()
                
                with autocast():
                    outputs = self.model.vision_model(images)
                    
                    # Calculate losses
                    class_loss = self.classification_loss(
                        outputs['food_classification'], food_classes
                    )
                    nutrition_loss = self.regression_loss(
                        outputs['nutrition_estimation'], nutrition
                    )
                    portion_loss = self.regression_loss(
                        outputs['portion_size'].squeeze(), portion_sizes
                    )
                    
                    total_loss = class_loss + 0.5 * nutrition_loss + 0.3 * portion_loss
                    
                # Backward pass with mixed precision
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.vision_optimizer)
                self.scaler.update()
                
                # Track metrics
                train_loss += total_loss.item()
                _, predicted = torch.max(outputs['food_classification'], 1)
                train_correct += (predicted == food_classes).sum().item()
                train_total += food_classes.size(0)
                
                pbar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'acc': f"{100 * train_correct / train_total:.2f}%"
                })
                
            # Validation
            val_loss, val_acc = self.validate_vision_model(val_loader)
            
            logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, "
                       f"Train Acc: {100*train_correct/train_total:.2f}%, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            if self.config.get('use_wandb', False):
                wandb.log({
                    'vision/train_loss': train_loss/len(train_loader),
                    'vision/train_acc': 100*train_correct/train_total,
                    'vision/val_loss': val_loss,
                    'vision/val_acc': val_acc
                })
                
    def validate_vision_model(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the vision model."""
        self.model.vision_model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                food_classes = batch['food_class'].to(self.device)
                nutrition = batch['nutrition'].to(self.device)
                
                outputs = self.model.vision_model(images)
                
                # Calculate loss
                class_loss = self.classification_loss(
                    outputs['food_classification'], food_classes
                )
                val_loss += class_loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs['food_classification'], 1)
                val_correct += (predicted == food_classes).sum().item()
                val_total += food_classes.size(0)
                
        return val_loss / len(val_loader), 100 * val_correct / val_total
        
    def train_meal_generator(self, train_loader: DataLoader, val_loader: DataLoader,
                           epochs: int = 10):
        """Train the meal generation model."""
        logger.info("Training meal generator...")
        
        for epoch in range(epochs):
            self.model.meal_generator.train()
            train_loss = 0
            
            pbar = tqdm(train_loader, desc=f"Generator Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                constraints = batch['nutrition_constraints'].to(self.device)
                
                self.generator_optimizer.zero_grad()
                
                with autocast():
                    # Get constraint embeddings
                    constraint_emb = self.model.meal_generator.nutrition_projector(constraints)
                    
                    # Forward pass through GPT-2
                    outputs = self.model.meal_generator.gpt2(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.generator_optimizer)
                self.scaler.update()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
            # Validation
            val_loss = self.validate_meal_generator(val_loader)
            
            logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, "
                       f"Val Loss: {val_loss:.4f}")
            
            if self.config.get('use_wandb', False):
                wandb.log({
                    'generator/train_loss': train_loss/len(train_loader),
                    'generator/val_loss': val_loss
                })
                
    def validate_meal_generator(self, val_loader: DataLoader) -> float:
        """Validate the meal generator."""
        self.model.meal_generator.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model.meal_generator.gpt2(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
                
        return val_loss / len(val_loader)
        
    def train_knowledge_graph(self, graph_data: nx.Graph, epochs: int = 10):
        """Train the nutritional knowledge graph."""
        logger.info("Training knowledge graph...")
        
        # Convert NetworkX graph to PyTorch geometric format
        edge_index = torch.tensor(list(graph_data.edges())).t().contiguous()
        num_nodes = graph_data.number_of_nodes()
        
        for epoch in range(epochs):
            self.model.knowledge_graph.train()
            
            # Sample node IDs
            node_ids = torch.arange(num_nodes)
            
            self.graph_optimizer.zero_grad()
            
            with autocast():
                # Forward pass
                embeddings = self.model.knowledge_graph(node_ids, edge_index)
                
                # Contrastive loss for graph embeddings
                pos_pairs = self._sample_positive_pairs(graph_data, 100)
                neg_pairs = self._sample_negative_pairs(num_nodes, 100)
                
                loss = self._graph_contrastive_loss(embeddings, pos_pairs, neg_pairs)
                
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.graph_optimizer)
            self.scaler.update()
            
            logger.info(f"Graph Epoch {epoch+1}: Loss: {loss.item():.4f}")
            
            if self.config.get('use_wandb', False):
                wandb.log({'graph/loss': loss.item()})
                
    def _sample_positive_pairs(self, graph: nx.Graph, num_samples: int) -> List[Tuple[int, int]]:
        """Sample connected node pairs from the graph."""
        edges = list(graph.edges())
        return random.sample(edges, min(num_samples, len(edges)))
        
    def _sample_negative_pairs(self, num_nodes: int, num_samples: int) -> List[Tuple[int, int]]:
        """Sample random non-connected node pairs."""
        pairs = []
        for _ in range(num_samples):
            i, j = random.randint(0, num_nodes-1), random.randint(0, num_nodes-1)
            pairs.append((i, j))
        return pairs
        
    def _graph_contrastive_loss(self, embeddings: torch.Tensor,
                               pos_pairs: List[Tuple[int, int]],
                               neg_pairs: List[Tuple[int, int]]) -> torch.Tensor:
        """Calculate contrastive loss for graph embeddings."""
        loss = 0
        margin = 1.0
        
        # Positive pairs should be close
        for i, j in pos_pairs:
            dist = torch.norm(embeddings[i] - embeddings[j])
            loss += dist ** 2
            
        # Negative pairs should be far
        for i, j in neg_pairs:
            dist = torch.norm(embeddings[i] - embeddings[j])
            loss += torch.relu(margin - dist) ** 2
            
        return loss / (len(pos_pairs) + len(neg_pairs))
        
    def train_end_to_end(self, data_loaders: Dict, epochs: int = 20):
        """End-to-end training of the complete nutrition model."""
        logger.info("Starting end-to-end training...")
        
        for epoch in range(epochs):
            logger.info(f"=== Epoch {epoch+1}/{epochs} ===")
            
            # Train each component
            if 'vision_train' in data_loaders:
                self.train_vision_model(
                    data_loaders['vision_train'],
                    data_loaders['vision_val'],
                    epochs=1
                )
                
            if 'generator_train' in data_loaders:
                self.train_meal_generator(
                    data_loaders['generator_train'],
                    data_loaders['generator_val'],
                    epochs=1
                )
                
            if 'graph' in data_loaders:
                self.train_knowledge_graph(data_loaders['graph'], epochs=1)
                
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch + 1)
                
    def save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        checkpoint_path = f"checkpoints/nutrition_model_epoch_{epoch}.pt"
        os.makedirs("checkpoints", exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'vision_optimizer': self.vision_optimizer.state_dict(),
            'generator_optimizer': self.generator_optimizer.state_dict(),
            'graph_optimizer': self.graph_optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'config': self.config
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")

def generate_synthetic_data(num_samples: int = 1000) -> Dict:
    """Generate synthetic training data for the nutrition model."""
    logger.info(f"Generating {num_samples} synthetic samples...")
    
    # Food classes
    food_classes = ['salad', 'pasta', 'rice', 'soup', 'sandwich', 'pizza',
                   'burger', 'steak', 'chicken', 'fish', 'vegetables', 'fruit']
    
    # Generate meal descriptions
    meal_data = []
    for i in range(num_samples):
        meal = {
            'user_profile': {
                'dietary_restrictions': random.choice([[], ['vegetarian'], ['vegan'], ['gluten_free']]),
                'allergies': random.choice([[], ['nuts'], ['dairy'], ['shellfish']]),
                'age': random.randint(18, 70),
                'weight': random.randint(50, 100),
                'activity_level': random.uniform(1.2, 2.0)
            },
            'constraints': {
                'calories': random.randint(300, 800),
                'protein': random.randint(10, 40),
                'carbs': random.randint(20, 80),
                'fat': random.randint(5, 30),
                'fiber': random.randint(3, 15)
            },
            'meal_type': random.choice(['breakfast', 'lunch', 'dinner', 'snack']),
            'meal_description': f"A delicious {random.choice(food_classes)} with fresh ingredients..."
        }
        meal_data.append(meal)
    
    # Generate nutritional knowledge graph
    graph = nx.Graph()
    
    # Add food nodes
    for i, food in enumerate(food_classes):
        graph.add_node(i, name=food, type='food')
        
    # Add nutrient nodes
    nutrients = ['protein', 'carbs', 'fat', 'fiber', 'vitamins', 'minerals']
    for i, nutrient in enumerate(nutrients):
        graph.add_node(len(food_classes) + i, name=nutrient, type='nutrient')
        
    # Add edges between foods and nutrients
    for food_idx in range(len(food_classes)):
        for nutrient_idx in range(len(nutrients)):
            if random.random() > 0.5:
                graph.add_edge(food_idx, len(food_classes) + nutrient_idx)
    
    return {
        'meal_data': meal_data,
        'knowledge_graph': graph
    }

def main():
    """Main training pipeline."""
    # Configuration
    config = {
        'vision_lr': 1e-4,
        'generator_lr': 1e-4,
        'graph_lr': 1e-4,
        'weight_decay': 0.01,
        'batch_size': 32,
        'epochs': 20,
        'use_wandb': False  # Set to True to use wandb
    }
    
    # Create model
    model = create_advanced_nutrition_model()
    
    # Initialize trainer
    trainer = NutritionModelTrainer(model, config)
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_data(num_samples=1000)
    
    # Create data loaders
    # Note: In production, you would load real data here
    meal_dataset = MealGenerationDataset(
        synthetic_data['meal_data'],
        model.meal_generator.tokenizer
    )
    
    train_size = int(0.8 * len(meal_dataset))
    val_size = len(meal_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        meal_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    data_loaders = {
        'generator_train': train_loader,
        'generator_val': val_loader,
        'graph': synthetic_data['knowledge_graph']
    }
    
    # Train the model
    trainer.train_end_to_end(data_loaders, epochs=config['epochs'])
    
    # Save final model
    model.save_model('models/nutrition_model_final.pt')
    logger.info("Training complete! Model saved to models/nutrition_model_final.pt")

if __name__ == "__main__":
    main()
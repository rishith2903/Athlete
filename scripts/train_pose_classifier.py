"""
Pose Classifier Training Script
================================
Trains a classifier for exercise recognition using pose landmarks data

Uses:
- backend/models/pose_training_data.csv (MediaPipe landmarks)
- backend/models/pose_label_map.json (exercise labels)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import json
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "backend" / "models"
DATA_PATH = MODELS_DIR / "pose_training_data.csv"
LABEL_MAP_PATH = MODELS_DIR / "pose_label_map.json"


class PoseLandmarksDataset(Dataset):
    """Dataset for pose landmarks classification"""
    
    def __init__(self, landmarks, labels):
        self.landmarks = torch.tensor(landmarks, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.landmarks[idx], self.labels[idx]


class ExerciseClassifier(nn.Module):
    """Neural network for classifying exercises from pose landmarks"""
    
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


def load_pose_data():
    """Load pose training data from CSV"""
    
    if not DATA_PATH.exists():
        logger.error(f"Pose data not found at {DATA_PATH}")
        logger.info("Run scripts/pose_data_loader.py first to generate the data")
        return None, None, None
    
    logger.info(f"Loading pose data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Get feature columns (all columns except sample_id and label-related)
    exclude_cols = ['sample_id', 'exercise_class', 'class_idx', 'Unnamed: 0']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Extract features and labels
    X = df[feature_cols].values
    
    # Handle missing class info
    if 'class_idx' in df.columns:
        y = df['class_idx'].values
    elif 'exercise_class' in df.columns:
        le = LabelEncoder()
        y = le.fit_transform(df['exercise_class'])
    else:
        # Default to dummy labels if no class info
        logger.warning("No class labels found, using dummy labels")
        y = np.zeros(len(df), dtype=int)
    
    # Load label map
    label_map = {}
    if LABEL_MAP_PATH.exists():
        with open(LABEL_MAP_PATH) as f:
            label_map = json.load(f)
    
    num_classes = max(len(label_map), len(np.unique(y))) or 5
    
    logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features")
    logger.info(f"Number of classes: {num_classes}")
    
    return X, y, num_classes


def train_classifier(epochs=50, batch_size=32, learning_rate=0.001):
    """Train the exercise classifier"""
    
    # Load data
    X, y, num_classes = load_pose_data()
    if X is None:
        return
    
    # Handle NaN values
    X = np.nan_to_num(X, nan=0.0)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create datasets
    train_dataset = PoseLandmarksDataset(X_train, y_train)
    val_dataset = PoseLandmarksDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    input_dim = X.shape[1]
    model = ExerciseClassifier(input_dim, num_classes)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Using device: {device}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    best_val_acc = 0.0
    
    logger.info("Starting training...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for landmarks, labels in pbar:
            landmarks = landmarks.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(landmarks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100*train_correct/train_total:.1f}%"
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for landmarks, labels in val_loader:
                landmarks = landmarks.to(device)
                labels = labels.to(device)
                
                outputs = model(landmarks)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f"Epoch {epoch+1}: Train Acc: {100*train_correct/train_total:.1f}%, "
                   f"Val Acc: {val_acc:.1f}%, Val Loss: {avg_val_loss:.4f}")
        
        scheduler.step(avg_val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = MODELS_DIR / "pose_classifier.pt"
            torch.save({
                'model_state': model.state_dict(),
                'input_dim': input_dim,
                'num_classes': num_classes,
                'best_val_acc': best_val_acc
            }, save_path)
            logger.info(f"Saved best model with {val_acc:.1f}% accuracy")
    
    logger.info(f"Training complete! Best validation accuracy: {best_val_acc:.1f}%")
    return model


def main():
    print("=" * 60)
    print("Pose Classifier Training")
    print("=" * 60)
    
    train_classifier(epochs=30, batch_size=32, learning_rate=0.001)


if __name__ == "__main__":
    main()

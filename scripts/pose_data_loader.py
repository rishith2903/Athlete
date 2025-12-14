"""
Pose Data Loader
=================
Converts pose estimation datasets into format for training pose checker

Input datasets:
- backend/data/Pose Checker/landmarks.csv (33 MediaPipe landmarks)
- backend/data/Pose Checker/angles.csv (joint angles)
- backend/data/Pose Checker/labels.csv (exercise labels)

Output: Formatted pose data for training
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "backend" / "data" / "Pose Checker"
OUTPUT_DIR = BASE_DIR / "backend" / "models"

def load_landmarks():
    """Load pose landmarks data"""
    csv_path = DATA_DIR / "landmarks.csv"
    
    if not csv_path.exists():
        logger.error(f"Landmarks data not found at {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} pose samples from landmarks.csv")
    logger.info(f"Columns: {df.columns.tolist()[:10]}...")  # First 10 columns
    
    return df


def load_angles():
    """Load joint angles data"""
    csv_path = DATA_DIR / "angles.csv"
    
    if not csv_path.exists():
        logger.warning(f"Angles data not found at {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} angle samples from angles.csv")
    
    return df


def load_labels():
    """Load exercise labels"""
    csv_path = DATA_DIR / "labels.csv"
    
    if not csv_path.exists():
        logger.warning(f"Labels data not found at {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} labels from labels.csv")
    
    return df


def process_pose_data(landmarks_df, angles_df, labels_df):
    """Combine and process pose data for training"""
    
    # Create training features from landmarks
    # MediaPipe provides 33 landmarks with x, y, z, visibility for each
    
    processed = []
    
    # If labels exist, use them for classification
    if labels_df is not None and 'label' in labels_df.columns:
        unique_labels = labels_df['label'].unique()
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
        for idx, row in landmarks_df.iterrows():
            if idx < len(labels_df):
                label = labels_df.iloc[idx].get('label', 'unknown')
                processed.append({
                    'sample_id': idx,
                    'exercise_class': label,
                    'class_idx': label_to_idx.get(label, 0),
                    **{col: row[col] for col in landmarks_df.columns if col != 'Unnamed: 0'}
                })
    else:
        # No labels - just process landmarks
        for idx, row in landmarks_df.iterrows():
            processed.append({
                'sample_id': idx,
                **{col: row[col] for col in landmarks_df.columns if col != 'Unnamed: 0'}
            })
    
    df = pd.DataFrame(processed)
    output_path = OUTPUT_DIR / "pose_training_data.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} processed pose samples to {output_path}")
    
    return df


def create_exercise_form_rules():
    """Create exercise form rules from angle data"""
    
    # Define ideal joint angle ranges for common exercises
    form_rules = {
        'squat': {
            'knee_angle_min': 70,
            'knee_angle_max': 100,
            'hip_angle_min': 60,
            'hip_angle_max': 90,
            'back_angle_min': 45,
            'back_angle_max': 75,
            'key_points': ['Keep knees aligned with toes', 'Maintain neutral spine', 'Depth: thighs parallel to floor']
        },
        'pushup': {
            'elbow_angle_down_min': 80,
            'elbow_angle_down_max': 100,
            'elbow_angle_up_min': 160,
            'elbow_angle_up_max': 180,
            'body_alignment_tolerance': 10,
            'key_points': ['Body in straight line', 'Elbows at 45° from body', 'Full range of motion']
        },
        'pullup': {
            'elbow_angle_down_min': 0,
            'elbow_angle_down_max': 30,
            'elbow_angle_up_min': 160,
            'elbow_angle_up_max': 180,
            'key_points': ['Chin above bar at top', 'Full arm extension at bottom', 'Control the movement']
        },
        'situp': {
            'hip_angle_min': 30,
            'hip_angle_max': 80,
            'key_points': ['Engage core throughout', 'No neck strain', 'Controlled movement']
        },
        'jumping_jack': {
            'arm_angle_down': 0,
            'arm_angle_up': 180,
            'leg_spread_min': 45,
            'leg_spread_max': 60,
            'key_points': ['Full arm extension', 'Land softly', 'Maintain rhythm']
        }
    }
    
    output_path = OUTPUT_DIR / "exercise_form_rules.json"
    with open(output_path, 'w') as f:
        json.dump(form_rules, f, indent=2)
    logger.info(f"Saved exercise form rules to {output_path}")
    
    return form_rules


def generate_pose_classifier_labels():
    """Generate label mapping for pose classification"""
    labels_path = DATA_DIR / "labels.csv"
    
    if labels_path.exists():
        labels_df = pd.read_csv(labels_path)
        if 'label' in labels_df.columns:
            unique_labels = labels_df['label'].unique().tolist()
        else:
            unique_labels = ['pushup', 'pullup', 'squat', 'situp', 'jumping_jack']
    else:
        unique_labels = ['pushup', 'pullup', 'squat', 'situp', 'jumping_jack']
    
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    
    output_path = OUTPUT_DIR / "pose_label_map.json"
    with open(output_path, 'w') as f:
        json.dump(label_map, f, indent=2)
    logger.info(f"Saved {len(label_map)} exercise labels to {output_path}")
    
    return label_map


def main():
    """Main function to process all pose data"""
    print("=" * 60)
    print("Pose Data Loader")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    landmarks = load_landmarks()
    angles = load_angles()
    labels = load_labels()
    
    if landmarks is not None:
        # Process pose data
        processed = process_pose_data(landmarks, angles, labels)
        
        # Create form rules
        create_exercise_form_rules()
        
        # Generate label map
        generate_pose_classifier_labels()
        
        print("\n" + "=" * 60)
        print("✅ Pose data processing complete!")
        print("=" * 60)
        print(f"\nOutput files in: {OUTPUT_DIR}")
        print("  - pose_training_data.csv")
        print("  - exercise_form_rules.json")
        print("  - pose_label_map.json")
    else:
        print("❌ Failed to process pose data")


if __name__ == "__main__":
    main()

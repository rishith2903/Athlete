"""
Workout Data Loader
====================
Converts downloaded real datasets into format expected by train_model.py

Input datasets:
- backend/data/Workout Recommender/megaGymDataset.csv (exercises)
- backend/data/Workout Recommender/gym_members_exercise_tracking.csv (users)

Output datasets:
- users.csv, exercises.csv, interactions.csv
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
DATA_DIR = BASE_DIR / "backend" / "data" / "Workout Recommender"
OUTPUT_DIR = BASE_DIR / "backend" / "workout-recommendation-service" / "training"

def load_exercise_data():
    """Load and transform megaGymDataset.csv to exercises.csv format"""
    csv_path = DATA_DIR / "megaGymDataset.csv"
    
    if not csv_path.exists():
        logger.error(f"Exercise data not found at {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} exercises from megaGymDataset.csv")
    
    # Map category to training format
    category_map = {
        'Strength': 'strength',
        'Cardio': 'cardio',
        'Stretching': 'flexibility',
        'Plyometrics': 'cardio',
        'Olympic Weightlifting': 'strength',
        'Powerlifting': 'strength',
        'Strongman': 'strength'
    }
    
    # Map columns to expected format
    exercises = pd.DataFrame({
        'exercise_id': range(len(df)),
        'name': df['Title'].fillna('Unknown Exercise'),
        'category': df['Type'].fillna('Strength').map(lambda x: category_map.get(x, 'strength')),
        'muscle_groups': df['BodyPart'].fillna('Full Body'),
        'equipment': df['Equipment'].fillna('Body Only').map(lambda x: 'none' if x == 'Body Only' else x.lower().replace(' ', '_')),
        'difficulty': df['Level'].fillna('Intermediate').map({
            'Beginner': 'beginner',
            'Intermediate': 'intermediate', 
            'Expert': 'advanced'
        }).fillna('intermediate'),
        'description': df['Desc'].fillna(''),
        'calories_per_minute': np.random.uniform(3.0, 10.0, len(df)),  # Required by training script
        'calories_per_rep': np.random.randint(3, 15, len(df)),
        'met_value': np.random.uniform(3.0, 10.0, len(df))
    })
    
    # Save
    output_path = OUTPUT_DIR / "exercises.csv"
    exercises.to_csv(output_path, index=False)
    logger.info(f"Saved {len(exercises)} exercises to {output_path}")
    
    return exercises


def load_user_data():
    """Load and transform gym_members_exercise_tracking.csv to users.csv format"""
    csv_path = DATA_DIR / "gym_members_exercise_tracking.csv"
    
    if not csv_path.exists():
        logger.warning(f"User data not found at {csv_path}, generating synthetic users")
        return generate_synthetic_users(500)
    
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} users from gym_members_exercise_tracking.csv")
    
    # Map columns
    goals = ['weight_loss', 'muscle_gain', 'endurance', 'strength', 'general_fitness']
    
    users = pd.DataFrame({
        'user_id': [f"user_{i}" for i in range(len(df))],
        'age': df.get('Age', np.random.randint(18, 65, len(df))),
        'gender': df.get('Gender', np.random.choice(['male', 'female'], len(df))).map(lambda x: str(x).lower() if pd.notna(x) else 'other'),
        'weight': df.get('Weight (kg)', np.random.uniform(50, 100, len(df))),
        'height': df.get('Height (m)', np.random.uniform(1.5, 2.0, len(df))),
        'fitness_level': map_experience_to_level(df.get('Experience_Level', pd.Series([2]*len(df)))),
        'primary_goal': np.random.choice(goals, len(df)),
        'secondary_goal': np.random.choice(goals, len(df)),  # Required by training script
        'workout_days_per_week': df.get('Workout_Frequency (days/week)', np.random.randint(2, 6, len(df))),
        'session_duration': df.get('Session_Duration (hours)', np.random.uniform(0.5, 1.5, len(df))) * 60,
        'available_equipment': np.random.choice(['none', 'dumbbells', 'barbell,dumbbells', 'full_gym'], len(df)),
        'has_injuries': np.random.choice([0, 1], len(df), p=[0.85, 0.15])
    })
    
    # Save
    output_path = OUTPUT_DIR / "users.csv"
    users.to_csv(output_path, index=False)
    logger.info(f"Saved {len(users)} users to {output_path}")
    
    return users


def map_experience_to_level(experience_col):
    """Map experience level (1-3) to fitness level string"""
    if isinstance(experience_col, pd.Series):
        return experience_col.map({1: 'beginner', 2: 'intermediate', 3: 'advanced'}).fillna('intermediate')
    return 'intermediate'


def generate_synthetic_users(num_users: int) -> pd.DataFrame:
    """Fallback: Generate synthetic users"""
    np.random.seed(42)
    
    users = pd.DataFrame({
        'user_id': [f"user_{i}" for i in range(num_users)],
        'age': np.random.randint(18, 65, num_users),
        'gender': np.random.choice(['male', 'female', 'other'], num_users),
        'weight': np.random.uniform(50, 110, num_users),
        'height': np.random.uniform(1.5, 2.0, num_users),
        'fitness_level': np.random.choice(['beginner', 'intermediate', 'advanced'], num_users),
        'primary_goal': np.random.choice(['weight_loss', 'muscle_gain', 'endurance', 'strength', 'general_fitness'], num_users),
        'workout_days_per_week': np.random.randint(2, 7, num_users),
        'session_duration': np.random.choice([30, 45, 60, 90], num_users),
        'available_equipment': np.random.choice(['none', 'dumbbells', 'barbell,dumbbells', 'full_gym'], num_users),
        'has_injuries': np.random.choice([0, 1], num_users, p=[0.85, 0.15])
    })
    
    output_path = OUTPUT_DIR / "users.csv"
    users.to_csv(output_path, index=False)
    logger.info(f"Generated and saved {len(users)} synthetic users")
    
    return users


def generate_interactions(users: pd.DataFrame, exercises: pd.DataFrame, num_interactions: int = 10000):
    """Generate user-exercise interaction data"""
    np.random.seed(42)
    
    interactions = []
    user_ids = users['user_id'].tolist()
    exercise_ids = exercises['exercise_id'].tolist()
    
    for _ in range(num_interactions):
        user_id = np.random.choice(user_ids)
        exercise_id = np.random.choice(exercise_ids)
        
        user = users[users['user_id'] == user_id].iloc[0]
        exercise = exercises[exercises['exercise_id'] == exercise_id].iloc[0]
        
        base_rating = 3.0
        if user['fitness_level'] == exercise['difficulty']:
            base_rating += 1.0
        
        interactions.append({
            'interaction_id': len(interactions),
            'user_id': user_id,
            'exercise_id': exercise_id,
            'rating': min(5.0, max(1.0, base_rating + np.random.normal(0, 0.5))),
            'completed': np.random.choice([0, 1], p=[0.1, 0.9]),
            'sets': np.random.randint(2, 5),
            'reps': np.random.randint(8, 15),
            'weight': np.random.uniform(0, 100),
            'duration_minutes': np.random.randint(1, 10),
            'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 365))
        })
    
    df = pd.DataFrame(interactions)
    output_path = OUTPUT_DIR / "interactions.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Generated {len(df)} interactions to {output_path}")
    
    return df


def main():
    """Main function to process all workout data"""
    print("=" * 60)
    print("Workout Data Loader")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    exercises = load_exercise_data()
    users = load_user_data()
    
    if exercises is not None and users is not None:
        interactions = generate_interactions(users, exercises)
        
        print("\n" + "=" * 60)
        print("✅ Workout data processing complete!")
        print("=" * 60)
        print(f"\nOutput files:")
        print(f"  - exercises.csv: {len(exercises)} exercises")
        print(f"  - users.csv: {len(users)} users")
        print(f"  - interactions.csv: {len(interactions)} interactions")
        print(f"\nOutput directory: {OUTPUT_DIR}")
    else:
        print("❌ Failed to process workout data")


if __name__ == "__main__":
    main()

"""
Nutrition Data Loader
======================
Converts Food.com recipes dataset into format for training nutrition model

Input datasets:
- backend/data/Nutritional Meal Planner/PP_recipes.csv
- backend/data/Nutritional Meal Planner/RAW_interactions.csv

Output: Formatted meal data for training
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import json
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "backend" / "data" / "Nutritional Meal Planner"
OUTPUT_DIR = BASE_DIR / "backend" / "models" / "nutritional-meal-planner" / "app" / "training"

def load_recipes():
    """Load and transform PP_recipes.csv"""
    csv_path = DATA_DIR / "PP_recipes.csv"
    
    if not csv_path.exists():
        logger.error(f"Recipe data not found at {csv_path}")
        return None
    
    df = pd.read_csv(csv_path, nrows=50000)  # Limit for memory
    logger.info(f"Loaded {len(df)} recipes from PP_recipes.csv")
    
    return df


def load_interactions():
    """Load user-recipe interactions for preference learning"""
    csv_path = DATA_DIR / "RAW_interactions.csv"
    
    if not csv_path.exists():
        logger.warning(f"Interactions not found at {csv_path}")
        return None
    
    df = pd.read_csv(csv_path, nrows=100000)  # Limit for memory
    logger.info(f"Loaded {len(df)} interactions from RAW_interactions.csv")
    
    return df


def parse_nutrition(nutrition_str):
    """Parse nutrition string from dataset"""
    try:
        # Format: [calories, fat, sugar, sodium, protein, saturated_fat, carbs]
        if pd.isna(nutrition_str):
            return {}
        
        # Clean and parse list string
        nutrition_str = str(nutrition_str).replace('[', '').replace(']', '')
        values = [float(x.strip()) for x in nutrition_str.split(',') if x.strip()]
        
        if len(values) >= 7:
            return {
                'calories': values[0],
                'fat': values[1],
                'sugar': values[2],
                'sodium': values[3],
                'protein': values[4],
                'saturated_fat': values[5],
                'carbs': values[6]
            }
        return {}
    except Exception as e:
        return {}


def classify_meal_type(name, tags_str):
    """Classify recipe into meal type based on name and tags"""
    name_lower = str(name).lower()
    tags_lower = str(tags_str).lower() if pd.notna(tags_str) else ""
    
    if any(word in name_lower or word in tags_lower for word in ['breakfast', 'brunch', 'pancake', 'waffle', 'oatmeal', 'eggs']):
        return 'breakfast'
    elif any(word in name_lower or word in tags_lower for word in ['lunch', 'sandwich', 'salad', 'soup']):
        return 'lunch'
    elif any(word in name_lower or word in tags_lower for word in ['dinner', 'supper', 'main-dish', 'entree']):
        return 'dinner'
    elif any(word in name_lower or word in tags_lower for word in ['snack', 'appetizer', 'finger-food']):
        return 'snack'
    elif any(word in name_lower or word in tags_lower for word in ['dessert', 'cake', 'cookie', 'pie']):
        return 'dessert'
    else:
        return 'other'


def classify_dietary(tags_str):
    """Classify dietary restrictions from tags"""
    tags_lower = str(tags_str).lower() if pd.notna(tags_str) else ""
    
    return {
        'is_vegetarian': 'vegetarian' in tags_lower,
        'is_vegan': 'vegan' in tags_lower,
        'is_gluten_free': 'gluten-free' in tags_lower or 'gluten free' in tags_lower,
        'is_dairy_free': 'dairy-free' in tags_lower or 'dairy free' in tags_lower,
        'is_low_carb': 'low-carb' in tags_lower or 'keto' in tags_lower,
        'is_healthy': 'healthy' in tags_lower or 'low-fat' in tags_lower
    }


def process_recipes(recipes_df):
    """Process recipes into training format"""
    processed = []
    
    for idx, row in recipes_df.iterrows():
        try:
            nutrition = parse_nutrition(row.get('nutrition', ''))
            dietary = classify_dietary(row.get('tags', ''))
            meal_type = classify_meal_type(row.get('name', ''), row.get('tags', ''))
            
            processed.append({
                'recipe_id': row.get('id', idx),
                'name': row.get('name', 'Unknown Recipe'),
                'meal_type': meal_type,
                'minutes': row.get('minutes', 30),
                'n_steps': row.get('n_steps', 5),
                'n_ingredients': row.get('n_ingredients', 8),
                'calories': nutrition.get('calories', 300),
                'protein': nutrition.get('protein', 15),
                'carbs': nutrition.get('carbs', 40),
                'fat': nutrition.get('fat', 12),
                'sugar': nutrition.get('sugar', 10),
                **dietary
            })
        except Exception as e:
            continue
    
    df = pd.DataFrame(processed)
    output_path = OUTPUT_DIR / "processed_recipes.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} processed recipes to {output_path}")
    
    return df


def process_interactions(interactions_df, recipes_df):
    """Process interactions for preference learning"""
    if interactions_df is None:
        return None
    
    # Get unique users with their interaction history
    user_prefs = interactions_df.groupby('user_id').agg({
        'recipe_id': 'count',
        'rating': 'mean'
    }).reset_index()
    user_prefs.columns = ['user_id', 'num_recipes', 'avg_rating']
    
    output_path = OUTPUT_DIR / "user_preferences.csv"
    user_prefs.to_csv(output_path, index=False)
    logger.info(f"Saved {len(user_prefs)} user preferences to {output_path}")
    
    return user_prefs


def create_meal_plans_sample():
    """Create sample meal plans from processed recipes"""
    csv_path = OUTPUT_DIR / "processed_recipes.csv"
    if not csv_path.exists():
        return None
    
    recipes = pd.read_csv(csv_path)
    
    # Sample recipes for different meal types
    meal_plans = []
    for meal_type in ['breakfast', 'lunch', 'dinner', 'snack']:
        type_recipes = recipes[recipes['meal_type'] == meal_type].head(20)
        for _, recipe in type_recipes.iterrows():
            meal_plans.append({
                'name': recipe['name'],
                'meal_type': meal_type,
                'calories': recipe['calories'],
                'protein': recipe['protein'],
                'carbs': recipe['carbs'],
                'fat': recipe['fat'],
                'prep_time': recipe['minutes'],
                'is_vegetarian': recipe.get('is_vegetarian', False),
                'is_vegan': recipe.get('is_vegan', False),
                'is_gluten_free': recipe.get('is_gluten_free', False)
            })
    
    df = pd.DataFrame(meal_plans)
    output_path = OUTPUT_DIR / "sample_meal_plans.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} sample meal plans to {output_path}")
    
    return df


def main():
    """Main function to process all nutrition data"""
    print("=" * 60)
    print("Nutrition Data Loader")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    recipes = load_recipes()
    interactions = load_interactions()
    
    if recipes is not None:
        # Process recipes
        processed_recipes = process_recipes(recipes)
        
        # Process interactions
        if interactions is not None:
            process_interactions(interactions, recipes)
        
        # Create sample meal plans
        create_meal_plans_sample()
        
        print("\n" + "=" * 60)
        print("✅ Nutrition data processing complete!")
        print("=" * 60)
        print(f"\nOutput files in: {OUTPUT_DIR}")
        print("  - processed_recipes.csv")
        print("  - user_preferences.csv")
        print("  - sample_meal_plans.csv")
    else:
        print("❌ Failed to process nutrition data")


if __name__ == "__main__":
    main()

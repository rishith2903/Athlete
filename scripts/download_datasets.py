"""
Dataset Downloader for AIthlete AI Models
==========================================
Downloads all required datasets for training:
- Nutrition data from Open Food Facts API
- Pose/Exercise data already downloaded in free-exercise-db
"""

import requests
import json
import csv
import os
from pathlib import Path

def download_openfoodfacts_sample():
    """Download sample nutrition data from Open Food Facts API"""
    print("Downloading nutrition data from Open Food Facts...")
    
    # Create directory
    os.makedirs("backend/data/nutrition", exist_ok=True)
    
    categories = ["breakfast-cereals", "fruits", "vegetables", "meats", "dairy", 
                  "pasta", "rice", "fish", "legumes", "nuts-and-seeds"]
    
    all_products = []
    
    for category in categories:
        try:
            url = f"https://world.openfoodfacts.org/category/{category}.json"
            response = requests.get(url, timeout=30)
            data = response.json()
            
            for product in data.get("products", [])[:50]:  # Get 50 per category
                nutrition = product.get("nutriments", {})
                all_products.append({
                    "name": product.get("product_name", "Unknown"),
                    "category": category,
                    "calories": nutrition.get("energy-kcal_100g", 0),
                    "protein": nutrition.get("proteins_100g", 0),
                    "carbs": nutrition.get("carbohydrates_100g", 0),
                    "fat": nutrition.get("fat_100g", 0),
                    "fiber": nutrition.get("fiber_100g", 0),
                    "sugars": nutrition.get("sugars_100g", 0),
                    "sodium": nutrition.get("sodium_100g", 0),
                })
            print(f"  ✓ {category}: {len(data.get('products', [])[:50])} products")
        except Exception as e:
            print(f"  ✗ {category}: {str(e)}")
    
    # Save to CSV
    if all_products:
        with open("backend/data/nutrition/openfoodfacts_nutrition.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_products[0].keys())
            writer.writeheader()
            writer.writerows(all_products)
        print(f"\n✅ Saved {len(all_products)} products to nutrition/openfoodfacts_nutrition.csv")
    
    return all_products

def create_meal_plans_sample():
    """Create sample meal plans dataset"""
    print("\nGenerating sample meal plans...")
    
    os.makedirs("backend/data/nutrition", exist_ok=True)
    
    meals = [
        {"name": "Oatmeal with Berries", "meal_type": "breakfast", "calories": 350, "protein": 12, "carbs": 55, "fat": 8, "is_vegan": True, "is_vegetarian": True, "is_gluten_free": False},
        {"name": "Greek Yogurt Parfait", "meal_type": "breakfast", "calories": 280, "protein": 18, "carbs": 35, "fat": 6, "is_vegan": False, "is_vegetarian": True, "is_gluten_free": True},
        {"name": "Scrambled Eggs with Toast", "meal_type": "breakfast", "calories": 420, "protein": 22, "carbs": 30, "fat": 20, "is_vegan": False, "is_vegetarian": True, "is_gluten_free": False},
        {"name": "Grilled Chicken Salad", "meal_type": "lunch", "calories": 380, "protein": 35, "carbs": 15, "fat": 18, "is_vegan": False, "is_vegetarian": False, "is_gluten_free": True},
        {"name": "Quinoa Buddha Bowl", "meal_type": "lunch", "calories": 450, "protein": 15, "carbs": 60, "fat": 16, "is_vegan": True, "is_vegetarian": True, "is_gluten_free": True},
        {"name": "Turkey Sandwich", "meal_type": "lunch", "calories": 520, "protein": 28, "carbs": 45, "fat": 22, "is_vegan": False, "is_vegetarian": False, "is_gluten_free": False},
        {"name": "Salmon with Vegetables", "meal_type": "dinner", "calories": 480, "protein": 38, "carbs": 20, "fat": 24, "is_vegan": False, "is_vegetarian": False, "is_gluten_free": True},
        {"name": "Pasta Primavera", "meal_type": "dinner", "calories": 550, "protein": 18, "carbs": 75, "fat": 16, "is_vegan": True, "is_vegetarian": True, "is_gluten_free": False},
        {"name": "Steak with Sweet Potato", "meal_type": "dinner", "calories": 620, "protein": 42, "carbs": 35, "fat": 28, "is_vegan": False, "is_vegetarian": False, "is_gluten_free": True},
        {"name": "Tofu Stir Fry", "meal_type": "dinner", "calories": 380, "protein": 22, "carbs": 35, "fat": 15, "is_vegan": True, "is_vegetarian": True, "is_gluten_free": True},
        {"name": "Mixed Nuts", "meal_type": "snack", "calories": 180, "protein": 6, "carbs": 8, "fat": 14, "is_vegan": True, "is_vegetarian": True, "is_gluten_free": True},
        {"name": "Protein Shake", "meal_type": "snack", "calories": 200, "protein": 25, "carbs": 10, "fat": 5, "is_vegan": False, "is_vegetarian": True, "is_gluten_free": True},
    ]
    
    with open("backend/data/nutrition/meal_plans.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=meals[0].keys())
        writer.writeheader()
        writer.writerows(meals)
    
    print(f"✅ Saved {len(meals)} meal plans to nutrition/meal_plans.csv")
    return meals

def main():
    print("="*50)
    print("AIthlete Dataset Downloader")
    print("="*50)
    
    # Download nutrition data
    download_openfoodfacts_sample()
    
    # Create meal plans
    create_meal_plans_sample()
    
    print("\n" + "="*50)
    print("✅ All datasets downloaded successfully!")
    print("="*50)
    print("\nDatasets available:")
    print("  - backend/data/nutrition/openfoodfacts_nutrition.csv")
    print("  - backend/data/nutrition/meal_plans.csv")
    print("  - backend/models/free-exercise-db/ (exercise database)")

if __name__ == "__main__":
    main()

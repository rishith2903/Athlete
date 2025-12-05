package com.aifitness.backend.entity;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.annotation.Id;
import org.springframework.data.annotation.LastModifiedDate;
import org.springframework.data.mongodb.core.index.Indexed;
import org.springframework.data.mongodb.core.mapping.Document;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Document(collection = "meals")
public class Meal {
    
    @Id
    private String id;
    
    @Indexed
    private String userId;
    
    private String name;
    private String type; // BREAKFAST, LUNCH, DINNER, SNACK, PRE_WORKOUT, POST_WORKOUT
    private LocalDate date;
    private String time; // Approximate time like "08:00"
    
    @Builder.Default
    private List<FoodItem> foodItems = new ArrayList<>();
    
    // Nutritional summary
    private Double totalCalories;
    private Double totalProtein; // in grams
    private Double totalCarbs; // in grams
    private Double totalFat; // in grams
    private Double totalFiber; // in grams
    private Double totalSugar; // in grams
    private Double totalSodium; // in mg
    
    // Micronutrients
    private Map<String, Double> vitamins;
    private Map<String, Double> minerals;
    
    private String notes;
    private boolean consumed;
    private LocalDateTime consumedAt;
    
    // AI-generated meal metadata
    private boolean aiGenerated;
    private Map<String, Object> aiMetadata;
    private List<String> dietaryTags; // VEGAN, VEGETARIAN, GLUTEN_FREE, DAIRY_FREE, etc.
    
    private String recipeUrl;
    private String imageUrl;
    private Integer preparationTime; // in minutes
    private String difficulty; // EASY, MEDIUM, HARD
    
    @CreatedDate
    private LocalDateTime createdAt;
    
    @LastModifiedDate
    private LocalDateTime updatedAt;
    
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class FoodItem {
        private String name;
        private String category; // PROTEIN, CARB, FAT, VEGETABLE, FRUIT, DAIRY, etc.
        private Double quantity;
        private String unit; // grams, ml, cups, pieces, etc.
        
        private Double calories;
        private Double protein;
        private Double carbs;
        private Double fat;
        private Double fiber;
        private Double sugar;
        private Double sodium;
        
        private String brand;
        private String barcode;
        private boolean verified;
    }
}
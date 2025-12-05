package com.aifitness.backend.entity;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.annotation.Id;
import org.springframework.data.annotation.LastModifiedDate;
import org.springframework.data.mongodb.core.index.CompoundIndex;
import org.springframework.data.mongodb.core.index.CompoundIndexes;
import org.springframework.data.mongodb.core.index.Indexed;
import org.springframework.data.mongodb.core.mapping.Document;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Document(collection = "progress")
@CompoundIndexes({
    @CompoundIndex(name = "user_date_idx", def = "{'userId': 1, 'date': -1}")
})
public class Progress {
    
    @Id
    private String id;
    
    @Indexed
    private String userId;
    
    private LocalDate date;
    
    // Body measurements
    private Double weight; // in kg
    private Double bodyFatPercentage;
    private Double muscleMass; // in kg
    private Double bmi; // Body Mass Index
    
    // Detailed measurements
    @Builder.Default
    private Map<String, Double> measurements = new HashMap<>(); // chest, waist, hips, thighs, arms, etc.
    
    // Performance metrics
    private Integer workoutsCompleted;
    private Integer totalWorkoutMinutes;
    private Integer totalCaloriesBurned;
    private Integer stepsCount;
    private Double distanceCovered; // in km
    
    // Nutrition tracking
    private Double caloriesConsumed;
    private Double proteinConsumed; // in grams
    private Double carbsConsumed; // in grams
    private Double fatConsumed; // in grams
    private Double waterIntake; // in liters
    
    // Sleep and recovery
    private Double sleepHours;
    private Integer sleepQuality; // 1-10 scale
    private Integer restingHeartRate;
    private Integer stressLevel; // 1-10 scale
    
    // Goals progress
    private Double goalCompletionPercentage;
    private String currentPhase; // CUTTING, BULKING, MAINTAINING, RECOMPOSITION
    
    // Photos
    private String frontPhotoUrl;
    private String sidePhotoUrl;
    private String backPhotoUrl;
    
    // Personal records
    @Builder.Default
    private Map<String, PersonalRecord> personalRecords = new HashMap<>();
    
    // Notes and mood
    private String notes;
    private String mood; // EXCELLENT, GOOD, AVERAGE, POOR, TERRIBLE
    private Integer energyLevel; // 1-10 scale
    private Integer motivationLevel; // 1-10 scale
    
    @CreatedDate
    private LocalDateTime createdAt;
    
    @LastModifiedDate
    private LocalDateTime updatedAt;
    
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class PersonalRecord {
        private String exerciseName;
        private Double weight; // in kg
        private Integer reps;
        private Integer sets;
        private Double distance; // for cardio exercises
        private Integer duration; // in seconds
        private LocalDateTime achievedAt;
        private boolean isCurrent;
    }
}
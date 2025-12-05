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
import org.springframework.data.mongodb.core.mapping.DBRef;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Document(collection = "workouts")
public class Workout {
    
    @Id
    private String id;
    
    @Indexed
    private String userId;
    
    private String name;
    private String description;
    private String type; // STRENGTH, CARDIO, FLEXIBILITY, BALANCE, MIXED
    private String difficulty; // BEGINNER, INTERMEDIATE, ADVANCED, EXPERT
    
    private Integer duration; // in minutes
    private Integer caloriesBurned;
    
    @Builder.Default
    private List<Exercise> exercises = new ArrayList<>();
    
    private String status; // PLANNED, IN_PROGRESS, COMPLETED, SKIPPED
    
    private LocalDateTime scheduledFor;
    private LocalDateTime startedAt;
    private LocalDateTime completedAt;
    
    private String notes;
    private Integer rating; // 1-5 star rating
    private Integer perceivedExertion; // 1-10 scale for perceived exertion
    private String feedback;
    
    // AI-generated workout metadata
    private boolean aiGenerated;
    private Map<String, Object> aiMetadata;
    private String targetMuscleGroups;
    
    @CreatedDate
    private LocalDateTime createdAt;
    
    @LastModifiedDate
    private LocalDateTime updatedAt;
    
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Exercise {
        private String name;
        private String category; // UPPER_BODY, LOWER_BODY, CORE, FULL_BODY, CARDIO
        private String equipment;
        private String muscleGroup;
        
        private Integer sets;
        private Integer reps;
        private Double weight; // in kg
        private Integer duration; // in seconds
        private Integer restTime; // in seconds between sets
        
        private String instructions;
        private String videoUrl;
        private String imageUrl;
        
        private boolean completed;
        private Integer actualSets;
        private Integer actualReps;
        private Double actualWeight;
        
        private String formCheckResult;
        private Double formScore;
        private String formFeedback;
    }
}